"""
Modified from smoe/entrypoint/cpt/cpt_fpt.py
"""
import logging
import os.path
from dataclasses import dataclass, field
from typing import Optional

import accelerate
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM

from smoe.entrypoint.expert_construction.get_gates.hidden_feature_clustering import (
    prepare_model_and_data,
)
from smoe.models.mixtral import MixtralForCausalLM
from smoe.utils.config import EnhancedTrainingArguments, ModelArguments, parse_args
from smoe.utils.io import create_dir
from smoe.utils.model_operation.modify_llama_model import (
    llama_with_hidden_distribution_recording,
)
from smoe.utils.model_operation.modify_llama_moe_v2_model import (
    llama_moe_v2_with_hidden_distribution_recording,
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
    save_path: Optional[str] = field(default=None)


# fmt: off
def main():
    model_args, data_args, training_args, analysis_args = parse_args(
        ModelArguments, DataArguments, EnhancedTrainingArguments, AnalysisArguments
    )

    model, dataloader = prepare_model_and_data(model_args, data_args, training_args)

    # 🔍 model check & prepare configs
    if isinstance(model, LlamaForCausalLM):
        model.model = llama_with_hidden_distribution_recording(model.model)  # 🔍 change the forward function
    elif isinstance(model, MixtralForCausalLM):
        model.model = llama_moe_v2_with_hidden_distribution_recording(model.model)  # 🔍 change the forward function
    else:
        raise ValueError("For now the only supported model is LLaMA and LLaMA-MoE-v2!")

    num_hidden_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # 🔍 prepare accelerator
    accelerator = accelerate.Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)

    # 🔍 forward for features
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            if i >= training_args.max_steps:
                break
            model(**batch)

    # 🔍 gather results
    distribution_info = {}
    for layer_index in range(num_hidden_layers):
        print("layer", layer_index)

        attn_distribution = accelerator.unwrap_model(model).model.layers[layer_index].attn_distribution
        all_attn_num = accelerator.gather(attn_distribution["number"])
        all_attn_mean = accelerator.gather(attn_distribution["mean"]).reshape(-1, hidden_size)
        all_attn_var = accelerator.gather(attn_distribution["variance"]).reshape(-1, hidden_size)
        final_attn_num = all_attn_num.sum()
        final_attn_mean = (all_attn_num[:, None] * all_attn_mean).sum(0) / final_attn_num
        final_attn_var = (all_attn_num[:, None] * (all_attn_var + all_attn_mean ** 2)).sum(0) / final_attn_num - final_attn_mean ** 2

        print("all_attn_num", all_attn_num.shape, "\n", all_attn_num)
        print("all_attn_mean", all_attn_mean.shape, "\n", all_attn_mean)
        print("all_attn_var", all_attn_var.shape, "\n", all_attn_var)
        print("final_attn_num", final_attn_num.shape, "\n", final_attn_num)
        print("final_attn_mean", final_attn_mean.shape, "\n", final_attn_mean)
        print("final_attn_var", final_attn_var.shape, "\n", final_attn_var)

        mlp_distribution = accelerator.unwrap_model(model).model.layers[layer_index].mlp_distribution
        all_mlp_num = accelerator.gather(mlp_distribution["number"])
        all_mlp_mean = accelerator.gather(mlp_distribution["mean"]).reshape(-1, hidden_size)
        all_mlp_var = accelerator.gather(mlp_distribution["variance"]).reshape(-1, hidden_size)
        final_mlp_num = all_mlp_num.sum()
        final_mlp_mean = (all_mlp_num[:, None] * all_mlp_mean).sum(0) / final_mlp_num
        final_mlp_var = (all_mlp_num[:, None] * (all_mlp_var + all_mlp_mean ** 2)).sum(0) / final_mlp_num - final_mlp_mean ** 2

        print("all_mlp_num", all_mlp_num.shape, "\n", all_mlp_num)
        print("all_mlp_mean", all_mlp_mean.shape, "\n", all_mlp_mean)
        print("all_mlp_var", all_mlp_var.shape, "\n", all_mlp_var)
        print("final_mlp_num", final_mlp_num.shape, "\n", final_mlp_num)
        print("final_mlp_mean", final_mlp_mean.shape, "\n", final_mlp_mean)
        print("final_mlp_var", final_mlp_var.shape, "\n", final_mlp_var)

        distribution_info[layer_index] = {
            "attn": {
                "number": final_attn_num.cpu(),
                "mean": final_attn_mean.cpu(),
                "variance": final_attn_var.cpu(),
            },
            "mlp": {
                "number": final_mlp_num.cpu(),
                "mean": final_mlp_mean.cpu(),
                "variance": final_mlp_var.cpu(),
            }
        }

    # 🔍 save the distribution information
    if accelerator.is_main_process:
        create_dir(analysis_args.save_path)
        torch.save(distribution_info, os.path.join(analysis_args.save_path, "distribution.pt"))
    accelerator.wait_for_everyone()
    accelerator.print("All done!")


# fmt: on

if __name__ == "__main__":
    main()
