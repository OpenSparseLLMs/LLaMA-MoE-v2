"""
Modified from smoe/entrypoint/cpt/cpt_fpt.py
"""
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import accelerate
import torch
from tqdm import tqdm
from transformers import DynamicCache, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP

from smoe.entrypoint.expert_construction_v2.get_gates.hidden_feature_clustering import (
    prepare_model_and_data,
)
from smoe.utils.config import EnhancedTrainingArguments, ModelArguments, parse_args
from smoe.utils.gpu_mem_track import print_gpu_memory
from smoe.utils.io import create_dir

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
class SelectionArguments:
    save_path: Optional[str] = field(default=None)
    num_experts: int = field(
        default=16,
        metadata={"help": "Number of experts (clusters)."},
    )


# fmt: off
def main():
    model_args, data_args, training_args, selection_args = parse_args(
        ModelArguments, DataArguments, EnhancedTrainingArguments, SelectionArguments
    )

    model, dataloader = prepare_model_and_data(model_args, data_args, training_args)

    # 🔍 model check & prepare configs
    if not isinstance(model, LlamaForCausalLM):
        raise ValueError("For now the only supported model is LLaMA!")

    num_hidden_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    assert hidden_size % selection_args.num_experts == 0

    # 🔍 prepare accelerator
    accelerator = accelerate.Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)

    # 🔍 begin layer-wise seleciton
    all_gate_weights = {}

    with torch.no_grad():
        # 🔍 prepare forward kwargs (copied from LlamaModel)
        ## initialize temp vars
        all_hidden_states = []
        all_attn_kwargs = []
        all_padding_masks = []

        ## prepare inputs
        accelerator.print(f"Forward for embedding outputs!")

        for i, batch in tqdm(enumerate(dataloader)):
            if i >= training_args.max_steps:
                break

            ## get embedded hidden states
            hidden_states = accelerator.unwrap_model(model).model.embed_tokens(batch["input_ids"])

            ## get attention kwargs
            output_attentions = False
            use_cache = False

            past_key_values = DynamicCache.from_legacy_cache(None)
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + hidden_states.shape[1],
                device=accelerator.device,
            )
            position_ids = cache_position.unsqueeze(0)

            attention_mask = batch.get("attention_mask")
            causal_mask = accelerator.unwrap_model(model).model._update_causal_mask(
                attention_mask,
                hidden_states,
                cache_position,
                past_key_values,
                output_attentions,
            )

            input_kwargs = {
                "attention_mask": causal_mask,
                "position_ids": position_ids,
                "past_key_value": past_key_values,
                "output_attentions": output_attentions,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }

            ## add to all
            all_hidden_states.append(hidden_states.cpu())
            all_attn_kwargs.append(input_kwargs)
            all_padding_masks.append(attention_mask.bool() if attention_mask is not None else None)

        accelerator.print(f"Forward for embedding outputs done!")

        # 🔍 perform selection for layers
        for i in range(num_hidden_layers):
            ## initialize temp vars
            global this_padding_mask
            this_padding_mask = None
            all_selection_hidden_states = []

            ## hook
            @torch.no_grad()
            def _forward_hook_mlp(module, input, output):
                """This hook captures the input features to the MLP layers"""
                # WARNING: Each run here causes the GPU memory to leak!!!!!!!!!
                if this_padding_mask is None:
                    selection_hidden_state = input[0].detach()
                else:
                    selection_hidden_state = input[0].detach()[this_padding_mask]
                all_selection_hidden_states.append(selection_hidden_state)

            ## prepare layer
            layer = accelerator.unwrap_model(model).model.layers[i]
            assert isinstance(layer.mlp, LlamaMLP)
            hook = layer.mlp.up_proj.register_forward_hook(_forward_hook_mlp)

            ## get hidden states
            accelerator.print(f"Forward for layer {i} outputs!")
            print_gpu_memory(accelerator)
            for batch_id, hidden_states in tqdm(enumerate(all_hidden_states)):
                this_padding_mask = all_padding_masks[batch_id]
                all_hidden_states[batch_id] = layer(hidden_states.to(accelerator.device), **all_attn_kwargs[batch_id])[0].cpu()
            accelerator.print(f"Forward for layer {i} outputs done!")
            print_gpu_memory(accelerator)

            ## 🔍 all gather across devices
            all_selection_hidden_states = torch.cat(all_selection_hidden_states, dim=0)
            if accelerator.num_processes > 1:
                # pad to make the tensors share the same shape
                raise NotImplementedError("Please use 1 GPU to run.")
            all_selection_hidden_states = accelerator.gather(all_selection_hidden_states).cpu().float().reshape(-1, hidden_size)

            ## free memory
            hook.remove()
            layer.to("cpu")
            del layer
            del hidden_states
            gc.collect()
            torch.cuda.empty_cache()
            accelerator.print(f"Clear cache done!")
            print_gpu_memory(accelerator)

            ## perform selection
            if accelerator.is_main_process:
                accelerator.print(f"Selection for layer {i} outputs!")

                num_features = all_selection_hidden_states.shape[0]
                print("total number of features:", num_features)

                gate_weights = all_selection_hidden_states[torch.randperm(num_features)[:selection_args.num_experts]]

                all_gate_weights[i] = gate_weights  # add to all weights
                accelerator.print(f"{gate_weights.shape}, {gate_weights}")
                accelerator.print(f"Clustering for layer {i} outputs done!")

                ## check the classification results
                all_logits = torch.from_numpy(all_selection_hidden_states).to(accelerator.device) @ gate_weights.clone().to(accelerator.device).t()
                all_classes = torch.argmax(all_logits, dim=-1)
                class_counts = torch.bincount(all_classes, minlength=selection_args.num_experts)
                accelerator.print(f"Classification counts for layer {i}: {class_counts}")

                ## free memory
                gc.collect()
                torch.cuda.empty_cache()
                print_gpu_memory(accelerator)

            else:
                del all_selection_hidden_states  # only store the hidden states on the main process

            accelerator.wait_for_everyone()

    # 🔍 save selection results (gate weights)
    if accelerator.is_main_process:
        create_dir(selection_args.save_path)
        torch.save(all_gate_weights, os.path.join(selection_args.save_path, "gate_weights.pt"))
    accelerator.wait_for_everyone()
    accelerator.print("All done!")


# fmt: on

if __name__ == "__main__":
    main()
