import os.path
import types
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.utils import release_memory
from torch.optim import SGD
from tqdm import tqdm
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP

from smoe.entrypoint.expert_construction_v2.get_gates.hidden_feature_clustering import (
    prepare_model_and_data,
)
from smoe.utils.config import (
    DataArguments,
    EnhancedTrainingArguments,
    ModelArguments,
    parse_args,
)
from smoe.utils.io import create_dir
from smoe.utils.model_operation.change_llama_forward import (
    forward_llama_mlp_with_backward_hook_bug_fix,
)
from smoe.utils.operations.operation_tensor import move_tensors_to_device


@dataclass
class DataArguments:
    dataset_dir_or_path: str = field(
        default="data/merged",
        metadata={"help": "Path to dataset directory or a single jsonl file"},
    )
    model_max_length: int = field(
        default=2048,
    )


@dataclass
class SplitArguments:
    gate_weights_file: Optional[str] = field(default=None)
    save_path: Optional[str] = field(default=None)


def main():
    # fmt: off
    model_args, data_args, training_args, split_args = parse_args(
        ModelArguments, DataArguments, EnhancedTrainingArguments, SplitArguments
    )

    model, dataloader = prepare_model_and_data(model_args, data_args, training_args)
    optimizer = SGD(model.parameters())

    # validity check
    if not isinstance(model, LlamaForCausalLM):
        raise ValueError("For now the only supported model is LLaMA!")

    # replace forward func (IMPORTANT)
    for layer_id, layer in enumerate(model.model.layers):
        layer.mlp.forward = types.MethodType(
            forward_llama_mlp_with_backward_hook_bug_fix, layer.mlp
        )  # locate block by the name template

    # 🔍 prepare accelerator
    accelerator = Accelerator()
    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

    # 🔍 prepare configs
    device = accelerator.device
    layer_num = accelerator.unwrap_model(model).config.num_hidden_layers
    neuron_num = accelerator.unwrap_model(model).config.intermediate_size

    # 🔍 load gate weights
    gate_weights = torch.load(split_args.gate_weights_file)
    gate_weights = move_tensors_to_device(gate_weights, device)  # move to the current device

    # 🔍 initialize temp vars
    cached_attention_masks = None
    classified_cluster_ids = {}
    cached_features_intermediate = {}

    importance_scores = {
        i: {
            j: torch.zeros((neuron_num,), device=device)
            for j in range(len(gate_weights[i]))  # cluster id
        }
        for i in range(layer_num)  # layer id
    }

    token_count = {
        i: {
            j: 0
            for j in range(len(gate_weights[i]))  # cluster id
        }
        for i in range(layer_num)  # layer id
    }  # number of classified tokens in each cluster

    # 🔍 hooks
    def _forward_hook_input_token(module, input, output):
        """This hook captures the input features to the MLP layers and calculates the classified cluster ids for tokens"""
        hidden_states = input[0]
        if cached_attention_masks is not None:
            hidden_states = hidden_states[cached_attention_masks]

        layer_id = module.layer_id
        feature_logits = hidden_states.detach() @ gate_weights[layer_id].t().to(hidden_states.dtype)
        classified_cluster_ids[layer_id] = torch.argmax(feature_logits, dim=-1)

        # if accelerator.is_main_process and layer_id == 0:
        #     print("feature_logits", feature_logits.shape, feature_logits)

    def _forward_hook_intermediate(module, input, output):
        """This hook captures the intermediate features of the neurons"""
        hidden_states = input[0]
        if cached_attention_masks is not None:
            hidden_states = hidden_states[cached_attention_masks]

        layer_id = module.layer_id
        cached_features_intermediate[layer_id] = hidden_states.detach()

        # if accelerator.is_main_process and layer_id == 0:
        #     print("cached_features_intermediate", len(input), input[0].shape)

    def _backward_hook_intermediate(module, grad_in, grad_out):
        """This hook captures the backward gradients of the intermediate neurons, and calculates the corresponding importance scores"""
        hidden_states_grad = grad_in[0]
        if cached_attention_masks is not None:
            hidden_states_grad = hidden_states_grad[cached_attention_masks]

        layer_id = module.layer_id

        # if accelerator.is_main_process and layer_id == 0:
        #     with torch.cuda.device(device):
        #         print(f"Used GPU memory ({device}): " + str(int(torch.cuda.memory_allocated() / 1024 / 1024)) + " MB")
        #     print("up", "grad_out", len(grad_out), [grad_out[i].shape if grad_out[i] is not None else None for i in range(len(grad_out))], grad_out, sep='\n')
        #     print("up", "grad_in", len(grad_in), [grad_in[i].shape if grad_in[i] is not None else None for i in range(len(grad_in))], grad_in, sep='\n')

        # add to the score cache
        for cluster_id in range(gate_weights[layer_id].shape[0]):  # iterate over clusters
            feature_mask: torch.BoolTensor = (classified_cluster_ids[layer_id] == cluster_id)
            importance_score = hidden_states_grad[feature_mask].detach() * cached_features_intermediate[layer_id][feature_mask]

            # if accelerator.is_main_process and layer_id == layer_num - 1:
            #     print("input", cached_features_intermediate[layer_id][feature_mask].shape)
            #     print("gradient", grad_in[0][feature_mask].shape)
            #     print("importance_score", importance_score.shape)

            importance_scores[layer_id][cluster_id] += torch.sum(torch.abs(importance_score), dim=0)
            token_count[layer_id][cluster_id] += feature_mask.sum().item()

            # if accelerator.is_main_process and layer_id == layer_num - 1:
            #     print("importance_scores", importance_scores[layer_id])
            #     print("token_count", token_count[layer_id])

    # 🔍 start calculating importance scores
    ## initialization
    for layer_id, layer in enumerate(accelerator.unwrap_model(model).model.layers):  # locate block by the name template
        assert isinstance(layer.mlp, LlamaMLP)

        layer.mlp.up_proj.layer_id = layer_id
        layer.mlp.up_proj.register_forward_hook(_forward_hook_input_token)

        layer.mlp.down_proj.layer_id = layer_id
        layer.mlp.down_proj.register_forward_hook(_forward_hook_intermediate)  # input of "down_proj" <==> "up_proj * gate_proj" output
        layer.mlp.down_proj.register_backward_hook(_backward_hook_intermediate)  # grad_in of "down_proj" <==> grad of "up_proj * gate_proj" output

    ## forward
    for i, batch in tqdm(enumerate(dataloader)):
        release_memory()
        if i >= training_args.max_steps:
            break

        if accelerator.is_main_process:
            with torch.cuda.device(device):
                print(f"Used GPU memory ({device}) (before forward): " + str(int(torch.cuda.memory_allocated() / 1024 / 1024)) + " MB")

        attention_mask = batch.get("attention_mask")
        cached_attention_masks = attention_mask.bool().flatten() if attention_mask is not None else None
        outputs = model(**batch)

        if accelerator.is_main_process:
            with torch.cuda.device(device):
                print(f"Used GPU memory ({device}) (after forward): " + str(int(torch.cuda.memory_allocated() / 1024 / 1024)) + " MB")

        loss = outputs.loss
        loss.backward()
        optimizer.zero_grad()

    for layer_id in range(layer_num):
        accelerator.print(f"Layer {layer_id}:")
        accelerator.print("importance_scores", importance_scores[layer_id])
        accelerator.print("token_count", token_count[layer_id])

    # 🔍 aggregate results
    final_importance_scores = {}

    for layer_id in tqdm(range(layer_num)):
        this_layer_importance_scores = {}
        for cluster_id in range(gate_weights[layer_id].shape[0]):  # iterate over clusters
            # gather results on different devices
            gathered_this_token_count = accelerator.reduce(torch.tensor(token_count[layer_id][cluster_id], device=device), reduction="sum")
            gathered_importance_scores = accelerator.reduce(importance_scores[layer_id][cluster_id], reduction="sum")

            # get mean values
            if gathered_this_token_count > 0:
                gathered_importance_scores /= gathered_this_token_count
            this_layer_importance_scores[cluster_id] = gathered_importance_scores.cpu()

            accelerator.print(f"layer {layer_id} cluster {cluster_id}: {gathered_this_token_count} tokens, {gathered_importance_scores[:10]}")
        final_importance_scores[layer_id] = this_layer_importance_scores

    # 🔍 save to disk
    if accelerator.is_main_process:
        create_dir(split_args.save_path)
        up_filename = os.path.join(split_args.save_path, f"importance_scores.pt")
        torch.save(final_importance_scores, up_filename)
    accelerator.wait_for_everyone()

    accelerator.print("Done!")
    # fmt: on


if __name__ == "__main__":
    main()
