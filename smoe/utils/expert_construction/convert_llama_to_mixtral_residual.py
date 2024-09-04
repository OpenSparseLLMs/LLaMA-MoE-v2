import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import init
from transformers.modeling_utils import dtype_byte_size

from smoe.models.mixtral import MixtralForCausalLM
from smoe.models.mixtral.configuration_mixtral import MixtralConfig
from smoe.utils.expert_construction.convert_llama_to_mixtral import (
    FFN_TYPE_MAP,
    is_safetensors_file,
)
from smoe.utils.io import dump_json, load_json


def convert_residual_safetensors(
    model_dir,
    dump_dir,
    num_experts: int,
    intermediate_size: int,  # 🔍
    intermediate_size_residual: int,  # 🔍
    top_k: int,
    scale_factor: int = 1.0,
    moe_type: str = "modulelist",
    neuron_indices: dict = None,
    gate_weights: dict = None,
):
    # fmt: off
    assert neuron_indices is not None  # 🔍 must pass the neuron indices

    model_folder = Path(model_dir)
    dump_folder = Path(dump_dir)
    dump_folder.mkdir(parents=True, exist_ok=True)
    ffn_type_map = FFN_TYPE_MAP[moe_type]

    raw_total_size = -1
    tensor_filepaths = []
    for filepath in model_folder.glob("*"):
        if not os.path.isdir(filepath):
            if is_safetensors_file(filepath):
                tensor_filepaths.append(filepath)
            if filepath.name == "config.json":
                config = MixtralConfig.from_pretrained(filepath)
                config.architectures = ["MixtralForCausalLM"]
                config.num_experts_per_tok = top_k
                config.num_local_experts = num_experts
                config.router_aux_loss_coef = 1e-2
                config.scale_factor = scale_factor
                config.moe_type = moe_type
                config.intermediate_size = intermediate_size  # 🔍
                config.intermediate_size_residual = intermediate_size_residual  # 🔍
                config.auto_map = {
                    "AutoConfig": "configuration_mixtral.MixtralConfig",
                    "AutoModel": "modeling_mixtral.MixtralModel",
                    "AutoModelForCausalLM": "modeling_mixtral.MixtralForCausalLM",
                }
                config.save_pretrained(dump_folder)
                for filename in [
                    "configuration_mixtral.py",
                    "modeling_mixtral.py",
                ]:
                    shutil.copy2(f"smoe/models/mixtral/{filename}", dump_folder / filename)
                (dump_folder / "__init__.py").touch()
            elif filepath.name == "model.safetensors.index.json":
                raw_total_size = load_json(filepath)["metadata"]["total_size"]
            else:
                # cp to dump_dir
                shutil.copy2(filepath, dump_folder / filepath.name)

    router_records = set()
    weight_map = {}
    total_size = 0
    total_gate_size = 0
    visited_layers = set()
    scattermoe_upgate_tensors = defaultdict(dict)
    for fi, filepath in enumerate(tensor_filepaths):
        with safe_open(filepath, framework="pt", device="cpu") as f:
            tensors = {}
            contained_layers = set()
            for key in f.keys():
                tensor = f.get_tensor(key)
                if ".mlp." in key:
                    # preparation
                    layer_idx, ffn_type = re.search(
                        r"model.layers.(\d+).mlp.(gate|up|down)_proj.weight", key
                    ).groups()
                    layer_idx = int(layer_idx)

                    contained_layers.add(layer_idx)

                    if ffn_type == "down":
                        hsz, mid = tensor.shape
                        mid_idx = 1
                    else:
                        mid, hsz = tensor.shape
                        mid_idx = 0

                    # initialize gate weights
                    if layer_idx not in router_records:
                        if gate_weights is None:  # use newly initialized gate weights
                            gate_weight = torch.zeros(num_experts, hsz)
                            init.kaiming_uniform_(gate_weight, a=math.sqrt(5))
                            tensors[
                                f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
                            ] = gate_weight
                        else:  # use provided gate weights
                            print(f"Initializing layer {layer_idx} gate weights using {gate_weights[layer_idx]}...")
                            tensors[
                                f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
                            ] = gate_weights[layer_idx][:num_experts].clone()  # 🔍 limit the weight num
                        router_records.add(layer_idx)
                    new_ffn_type = ffn_type_map[ffn_type]

                    # initialize expert weights
                    this_layer_indices: dict = neuron_indices[layer_idx]  # 🔍

                    # 🔍 add residual weights
                    print(f"Initializing layer {layer_idx} residual expert {ffn_type} using neurons with indices {this_layer_indices['residual']}...")
                    if mid_idx == 0:
                        expert_tensor = tensor[this_layer_indices['residual']].clone()
                    else:
                        expert_tensor = tensor[:, this_layer_indices['residual']].clone()
                    tensors[f"model.layers.{layer_idx}.mlp_residual.{new_ffn_type}.weight"] = expert_tensor

                    # add moe weights
                    if moe_type == "megablocks":
                        states_dict_name = f"model.layers.{layer_idx}.block_sparse_moe.experts.mlp.{new_ffn_type}"
                        if mid_idx == 1:
                            tensor = tensor.view(mid, hsz)
                        expert_size = mid // num_experts
                        tensors[states_dict_name] = torch.zeros_like(tensor)
                        for expert_idx in range(num_experts):
                            print(f"Initializing layer {layer_idx} expert {expert_idx} {ffn_type} using neurons with indices {this_layer_indices[expert_idx]}...")
                            tensors[states_dict_name][expert_idx * expert_size: (expert_idx + 1) * expert_size] = tensor[this_layer_indices[expert_idx]].clone()

                    elif moe_type == "modulelist":
                        for expert_idx in range(num_experts):
                            print(f"Initializing layer {layer_idx} expert {expert_idx} {ffn_type} using neurons with indices {this_layer_indices[expert_idx]}...")
                            if mid_idx == 0:
                                expert_tensor = tensor[this_layer_indices[expert_idx]].clone()
                            else:
                                expert_tensor = tensor[:, this_layer_indices[expert_idx]].clone()
                            tensors[
                                f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{new_ffn_type}.weight"
                            ] = expert_tensor

                    elif moe_type == "scattermoe":
                        if mid_idx == 1:
                            tensor = tensor.view(mid, hsz)

                        expert_size = mid // num_experts
                        temp_tensor = torch.zeros((num_experts, expert_size, hsz), device=tensor.device, dtype=tensor.dtype)
                        for expert_idx in range(num_experts):
                            print(f"Initializing layer {layer_idx} expert {expert_idx} {ffn_type} using neurons with indices {this_layer_indices[expert_idx]}...")
                            temp_tensor[expert_idx] = tensor[this_layer_indices[expert_idx]].clone()
                        tensor = temp_tensor

                        if new_ffn_type == "output_experts":
                            tensors[
                                f"model.layers.{layer_idx}.block_sparse_moe.experts.{new_ffn_type}.weight"
                            ] = tensor.permute(0, 2, 1)
                        elif new_ffn_type == "experts#1":
                            scattermoe_upgate_tensors[layer_idx][0] = tensor
                        elif new_ffn_type == "experts#2":
                            scattermoe_upgate_tensors[layer_idx][1] = tensor
                        else:
                            raise KeyError
                    else:
                        raise NotImplementedError

                else:
                    tensors[key] = tensor

            if moe_type == "scattermoe":
                # for the last file, take all the rest of the layers
                if fi == len(tensor_filepaths) - 1:
                    contained_layers = (
                            set(scattermoe_upgate_tensors.keys()) - visited_layers
                    )

                for layer_idx in contained_layers:
                    upgate_tensors = scattermoe_upgate_tensors[layer_idx]
                    try:
                        up = upgate_tensors[0]
                        gate = upgate_tensors[1]
                        upgate = torch.cat([up, gate], dim=1)
                        tensors[
                            f"model.layers.{layer_idx}.block_sparse_moe.experts.experts.weight"
                        ] = upgate
                        visited_layers.add(layer_idx)
                    except KeyError:
                        print(
                            f"layer {layer_idx} has no up or gate tensors in {filepath.name}, skipping..."
                        )

            for key in tensors:
                tensors[key] = tensors[key].contiguous()
            save_file(tensors, dump_folder / filepath.name, metadata={"format": "pt"})
            for key, tensor in tensors.items():
                weight_size = tensor.numel() * dtype_byte_size(tensor.dtype)
                total_size += weight_size
                weight_map[key] = filepath.name
                if ".block_sparse_moe.gate." in key:
                    total_gate_size += weight_size
                print(key, tensor.shape)

    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    dump_json(index, dump_folder / "model.safetensors.index.json", indent=2)
    assert total_size - total_gate_size == raw_total_size


if __name__ == "__main__":
    num_experts = 56
    top_k = 8

    # src_model_dir = "/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct"
    # tgt_model_dir_prefix = "/mnt/petrelfs/zhutong/smoe/resources/llama-3-8b-mixtral-Residual"
    # tgt_model_dir_prefix = f"/mnt/petrelfs/shared_data/quxioaye/llama_moe_v2/converted_models/split-sequential-Top{top_k}"

    src_model_dir = "/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B"
    tgt_model_dir_prefix = f"/mnt/petrelfs/shared_data/quxioaye/llama_moe_v2/converted_models/base-split-sequential-Top{top_k}"

    # tgt_moe_types = ["modulelist", "megablocks", "scattermoe"]
    tgt_moe_types = ["modulelist"]

    neuron_indices_file = ""
    gate_weights_file = ""

    neuron_indices = torch.load(neuron_indices_file)
    intermediate_size = len(neuron_indices[0][0])
    intermediate_size_residual = len(neuron_indices[0]["residual"])

    for moe_type in tgt_moe_types:
        print(f"converting {moe_type}")
        convert_residual_safetensors(
            src_model_dir,
            f"{tgt_model_dir_prefix}-{moe_type}-{num_experts}e-top{top_k}-residual",  # 🔍
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,  # 🔍
            intermediate_size_residual=intermediate_size_residual,  # 🔍
            moe_type=moe_type,
            neuron_indices=neuron_indices,  # 🔍
            gate_weights=None
            if gate_weights_file == ""
            else torch.load(gate_weights_file),
        )

        print(f"testing {moe_type}")
        m = MixtralForCausalLM.from_pretrained(
            f"{tgt_model_dir_prefix}-{moe_type}-{num_experts}e-top{top_k}"
        ).bfloat16()

        print(f"Re-saving {moe_type}")
        m.save_pretrained(f"{tgt_model_dir_prefix}")

        print("Done")
    # fmt: on
