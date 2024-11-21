"""
Modified from smoe/entrypoint/cpt/cpt_fpt.py
"""
import logging
import os.path
import shutil
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import accelerate
import torch
from torch.nn.parameter import Parameter
from tqdm import tqdm

from smoe.entrypoint.expert_construction.get_gates.hidden_feature_clustering import (
    prepare_model_and_data,
)
from smoe.models.mixtral import MixtralForCausalLM
from smoe.utils.config import EnhancedTrainingArguments, ModelArguments, parse_args
from smoe.utils.io import create_dir
from smoe.utils.model_operation.modify_llama_moe_v2_model import (
    llama_moe_v2_with_hidden_distribution_recording_for_alignment,
)

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    dataset_dir_or_path: str = field(
        default="data/merged",
        metadata={"help": "Path to dataset directory or a single jsonl file"},
    )
    model_max_length: int = field(
        default=4096,
    )


@dataclass
class AnalysisArguments:
    reference_distribution_file: Optional[str] = field(default=None)
    save_path: Optional[str] = field(default=None)


# fmt: off
def main():
    model_args, data_args, training_args, analysis_args = parse_args(
        ModelArguments, DataArguments, EnhancedTrainingArguments, AnalysisArguments
    )

    model, dataloader = prepare_model_and_data(model_args, data_args, training_args)

    # 🔍 model check & prepare configs
    if isinstance(model, MixtralForCausalLM):
        model.model = llama_moe_v2_with_hidden_distribution_recording_for_alignment(model.model)  # 🔍 change the forward function
        model.config.use_cache = False  # 🔍 set configuration
        model.config.add_rescale_bias = True  # 🔍 set configuration
    else:
        raise ValueError("For now the only supported model is MixtralForCausalLM!")

    if model.config.moe_type != "modulelist":
        raise ValueError("For now the only supported MoE type is modulelist!")

    num_hidden_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    use_attn_moe = model.config.use_attn_moe

    # 🔍 load distribution information from file
    # Example Format:
    # {
    #     layer_index (int): {
    #         "attn": {
    #             "number": torch.Tensor,
    #             "mean": torch.Tensor,
    #             "variance": torch.Tensor,
    #         },
    #         "mlp": {
    #             "number": torch.Tensor,
    #             "mean": torch.Tensor,
    #             "variance": torch.Tensor,
    #         }
    #     },
    #     ......
    # }
    ref_distribution: Dict = torch.load(analysis_args.reference_distribution_file)

    # 🔍 prepare accelerator
    accelerator = accelerate.Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)

    # 🔍 start aligning
    scale_factors = {}

    with torch.no_grad():
        for target_layer in range(num_hidden_layers):
            # check if is MoE
            if not accelerator.unwrap_model(model).model.layers[target_layer].is_moe:
                continue

            for target_module in ["attn", "mlp"] if use_attn_moe else ["mlp"]:
                print(f"layer {target_layer} {target_module}")

                # set the target module to align & initialize the statistics
                for layer_idx, layer in enumerate(accelerator.unwrap_model(model).model.layers):
                    if layer_idx == target_layer:
                        layer.align_module = target_module
                        layer.distribution = {
                            "number": torch.zeros((1,)),
                            "mean": torch.zeros((hidden_size,)),
                            "variance": torch.zeros((hidden_size,)),
                        }
                    else:
                        layer.align_module = None
                        layer.distribution = None

                accelerator.unwrap_model(model).model.early_stopping_layer = target_layer

                # forward for distribution of the target module
                for i, batch in tqdm(enumerate(iter(dataloader))):  # each time we create a new iter, i.e., the dataloader is reset
                    if i >= training_args.max_steps:
                        break
                    model(**batch)

                # gather results
                distribution = accelerator.unwrap_model(model).model.layers[target_layer].distribution
                print("distribution\n", distribution)

                all_num = accelerator.gather(distribution["number"])
                all_mean = accelerator.gather(distribution["mean"]).reshape(-1, hidden_size)
                all_var = accelerator.gather(distribution["variance"]).reshape(-1, hidden_size)
                final_num = all_num.sum()
                final_mean = (all_num[:, None] * all_mean).sum(0) / final_num
                final_var = (all_num[:, None] * (all_var + all_mean ** 2)).sum(0) / final_num - final_mean ** 2

                # calculate the scale factor for the current module
                ref_mean = ref_distribution[target_layer][target_module]["mean"].to(accelerator.device)
                ref_var = ref_distribution[target_layer][target_module]["variance"].to(accelerator.device)

                scale_magnitude = torch.sqrt(ref_var / final_var)
                scale_bias = ref_mean - final_mean * scale_magnitude

                if target_layer not in scale_factors:
                    scale_factors[target_layer] = {}
                scale_factors[target_layer][target_module] = {
                    "scale_bias": scale_bias.clone().cpu(),
                    "scale_magnitude": scale_magnitude.clone().cpu(),
                }

                print(f"final_num {final_num}")
                print(f"mean: {final_mean} -> {ref_mean}")
                print(f"variance: {final_var} -> {ref_var}")
                print(f"scale bias: {scale_bias}")
                print(f"scale magnitude: {scale_magnitude}")

                # perform weight rescaling
                layer = accelerator.unwrap_model(model).model.layers[target_layer]

                if target_module == "attn":
                    for expert_idx, attention_expert in enumerate(layer.self_attn.o_proj):
                        old_bias = None if attention_expert.bias is None else attention_expert.bias.data.clone()
                        old_weight = attention_expert.weight.data.clone()
                        attention_expert.bias = Parameter(scale_bias.clone())
                        attention_expert.weight *= scale_magnitude.unsqueeze(1)
                        print(f"attn expert {expert_idx} o_proj bias: {old_bias} -> {attention_expert.bias.data}")
                        print(f"attn expert {expert_idx} o_proj weight: {old_weight} -> {attention_expert.weight.data}")

                elif target_module == "mlp":
                    for expert_idx, mlp_expert in enumerate(layer.block_sparse_moe.experts):
                        old_bias = None if mlp_expert.w2.bias is None else mlp_expert.w2.bias.data.clone()
                        old_weight = mlp_expert.w2.weight.data.clone()
                        mlp_expert.w2.bias = Parameter(scale_bias.clone())
                        mlp_expert.w2.weight *= scale_magnitude.unsqueeze(1)
                        print(f"MLP expert {expert_idx} down_proj bias: {old_bias} -> {mlp_expert.w2.bias.data}")
                        print(f"MLP expert {expert_idx} down_proj weight: {old_weight} -> {mlp_expert.w2.weight.data}")

                else:
                    raise NotImplementedError

    # 🔍 save the aligned model
    if accelerator.is_main_process:
        create_dir(analysis_args.save_path)

        # scale factor (just for recording, not used when loading model)
        torch.save(scale_factors, os.path.join(analysis_args.save_path, "scale_factors.pt"))

        # model
        accelerator.unwrap_model(model).save_pretrained(analysis_args.save_path)

        # tokenizer
        tokenizer = dataloader.dataset.tokenizer
        tokenizer.save_pretrained(analysis_args.save_path)

        # code
        current_path = os.path.dirname(__file__)
        if isinstance(accelerator.unwrap_model(model), MixtralForCausalLM):
            shutil.copy(os.path.join(current_path, "../../../models/mixtral/configuration_mixtral.py"), analysis_args.save_path)
            shutil.copy(os.path.join(current_path, "../../../models/mixtral/modeling_mixtral.py"), analysis_args.save_path)
        else:
            warnings.warn(f"[WARN] unknown model type {type(accelerator.unwrap_model(model))}")

    accelerator.wait_for_everyone()
    accelerator.print("All done!")


# fmt: on

if __name__ == "__main__":
    main()
