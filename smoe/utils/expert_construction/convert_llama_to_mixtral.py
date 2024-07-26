import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.modeling_utils import dtype_byte_size

from smoe.models.mixtral.configuration_mixtral import MixtralConfig
from smoe.models.mixtral.modeling_mixtral import MixtralForCausalLM
from smoe.utils.io import dump_json, load_json


def is_safetensors_file(filepath):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    string = filepath.name
    return re.match(r"model-\d{5}-of-\d{5}.safetensors", string) is not None


FFN_TYPE_MAP = {
    "megablocks": {
        "gate": "w1",
        "down": "w2",
        "up": "v1",
    },
    "modulelist": {
        "gate": "w1",
        "down": "w2",
        "up": "w3",
    },
    "scattermoe": {
        "gate": "experts#2",
        "down": "output_experts",
        "up": "experts#1",
    },
}


def convert_safetensors(
    model_dir,
    dump_dir,
    num_experts: int,
    top_k: int,
    moe_type: str,
    neuron_indices: dict = None,
    gate_weights: dict = None,
):
    # fmt: off
    model_folder = Path(model_dir)
    dump_folder = Path(dump_dir)
    dump_folder.mkdir(parents=True, exist_ok=True)
    ffn_type_map = FFN_TYPE_MAP[moe_type]

    raw_total_size = -1
    tensor_filepaths = []
    for filepath in model_folder.glob("*"):
        if is_safetensors_file(filepath):
            tensor_filepaths.append(filepath)
        if filepath.name == "config.json":
            config = MixtralConfig.from_pretrained(filepath)
            config.num_experts_per_tok = top_k
            config.num_local_experts = num_experts
            config.router_aux_loss_coef = 1e-2
            config.act_rescale = True
            config.moe_type = moe_type
            config.intermediate_size = config.intermediate_size // num_experts
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
                            tensors[
                                f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
                            ] = torch.zeros(num_experts, hsz)  # TODO by DDZ: all zeros may be problematic here. I suggest using random initialization, where the initialization std should be adjusted according to the std of hidden features. You can try this out if possible.
                        else:  # use provided gate weights
                            print(f"Initializing layer {layer_idx} gate weights using {gate_weights[layer_idx]}...")
                            tensors[
                                f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
                            ] = gate_weights[layer_idx]
                        router_records.add(layer_idx)
                    new_ffn_type = ffn_type_map[ffn_type]

                    # initialize expert weights
                    if moe_type == "megablocks":
                        states_dict_name = f"model.layers.{layer_idx}.block_sparse_moe.experts.mlp.{new_ffn_type}"
                        if mid_idx == 1:
                            tensor = tensor.view(mid, hsz)
                        if neuron_indices is None:  # sequential split
                            tensors[states_dict_name] = tensor
                        else:  # split according to the given indices
                            this_layer_indices: list = neuron_indices[layer_idx]
                            expert_size = mid // num_experts
                            tensors[states_dict_name] = torch.zeros_like(tensor)
                            for expert_idx in range(num_experts):
                                print(f"Initializing layer {layer_idx} expert {expert_idx} {ffn_type} using neurons with indices {this_layer_indices[expert_idx]}...")
                                tensors[states_dict_name][expert_idx * expert_size: (expert_idx + 1) * expert_size] = tensor[this_layer_indices[expert_idx]].clone()

                    elif moe_type == "modulelist":
                        expert_size = mid // num_experts
                        for expert_idx in range(num_experts):
                            if mid_idx == 0:
                                if neuron_indices is None:  # sequential split
                                    expert_tensor = tensor[expert_idx * expert_size: (expert_idx + 1) * expert_size].clone()
                                else:  # split according to the given indices
                                    this_layer_indices: list = neuron_indices[layer_idx]
                                    print(f"Initializing layer {layer_idx} expert {expert_idx} {ffn_type} using neurons with indices {this_layer_indices[expert_idx]}...")
                                    expert_tensor = tensor[this_layer_indices[expert_idx]].clone()
                            else:
                                if neuron_indices is None:  # sequential split
                                    expert_tensor = tensor[:, expert_idx * expert_size: (expert_idx + 1) * expert_size].clone()
                                else:  # split according to the given indices
                                    this_layer_indices: list = neuron_indices[layer_idx]
                                    print(f"Initializing layer {layer_idx} expert {expert_idx} {ffn_type} using neurons with indices {this_layer_indices[expert_idx]}...")
                                    expert_tensor = tensor[:, this_layer_indices[expert_idx]].clone()
                            tensors[
                                f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{new_ffn_type}.weight"
                            ] = expert_tensor

                    elif moe_type == "scattermoe":
                        if mid_idx == 1:
                            tensor = tensor.view(mid, hsz)

                        expert_size = mid // num_experts
                        if neuron_indices is None:  # sequential split
                            tensor = tensor.view(num_experts, expert_size, hsz)
                        else:  # split according to the given indices
                            this_layer_indices: list = neuron_indices[layer_idx]
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
    # fmt: on


if __name__ == "__main__":
    num_experts = 56
    top_k = 8

    src_model_dir = "/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct"
    tgt_model_dir_prefix = "/mnt/petrelfs/zhutong/smoe/resources/llama-3-8b-mixtral"
    tgt_moe_types = ["modulelist", "megablocks", "scattermoe"]

    neuron_indices_file = ""
    gate_weights_file = ""

    for moe_type in tgt_moe_types:
        print(f"converting {moe_type}")
        convert_safetensors(
            src_model_dir,
            f"{tgt_model_dir_prefix}-{moe_type}-{num_experts}e-top{top_k}",
            num_experts=num_experts,
            top_k=top_k,
            moe_type=moe_type,
            neuron_indices=None
            if neuron_indices_file == ""
            else torch.load(neuron_indices_file),
            gate_weights=None
            if gate_weights_file == ""
            else torch.load(gate_weights_file),
        )

        print(f"testing {moe_type}")
        m = MixtralForCausalLM.from_pretrained(
            f"{tgt_model_dir_prefix}-{moe_type}-{num_experts}e-top{top_k}",
        )
