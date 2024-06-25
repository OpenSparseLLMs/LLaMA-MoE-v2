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
):
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
                "__init__.py",
                "configuration_mixtral.py",
                "modeling_mixtral.py",
            ]:
                shutil.copy2(model_folder / filename, dump_folder / filename)
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
                    layer_idx, ffn_type = re.search(
                        r"model.layers.(\d+).mlp.(gate|up|down)_proj.weight", key
                    ).groups()
                    contained_layers.add(layer_idx)
                    if ffn_type == "down":
                        hsz, mid = tensor.shape
                        mid_idx = 1
                    else:
                        mid, hsz = tensor.shape
                        mid_idx = 0
                    if layer_idx not in router_records:
                        tensors[
                            f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
                        ] = torch.zeros(num_experts, hsz)
                        router_records.add(layer_idx)
                    new_ffn_type = ffn_type_map[ffn_type]
                    if moe_type == "megablocks":
                        if mid_idx == 1:
                            tensor = tensor.view(mid, hsz)
                        tensors[
                            f"model.layers.{layer_idx}.block_sparse_moe.experts.mlp.{new_ffn_type}"
                        ] = tensor
                    elif moe_type == "modulelist":
                        expert_size = mid // num_experts
                        for expert_idx in range(num_experts):
                            if mid_idx == 0:
                                expert_tensor = tensor[
                                    expert_idx
                                    * expert_size : (expert_idx + 1)  # noqa: W503,E203
                                    * expert_size  # noqa: W503
                                ].clone()
                            else:
                                expert_tensor = tensor[
                                    :,
                                    expert_idx
                                    * expert_size : (expert_idx + 1)  # noqa: W503,E203
                                    * expert_size,  # noqa: W503
                                ].clone()
                            tensors[
                                f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{new_ffn_type}.weight"
                            ] = expert_tensor
                    elif moe_type == "scattermoe":
                        if mid_idx == 1:
                            tensor = tensor.view(mid, hsz)
                        tensor = tensor.view(num_experts, mid // num_experts, hsz)

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
    for moe_type in [
        # "megablocks",
        # "modulelist",
        "scattermoe"
    ]:
        print(f"converting {moe_type}")
        convert_safetensors(
            "/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct",
            f"/mnt/petrelfs/zhutong/smoe/resources/llama-3-8b-mixtral-{moe_type}-56e-top8",
            num_experts=56,
            top_k=8,
            moe_type=moe_type,
        )

        print(f"testing {moe_type}")
        m = MixtralForCausalLM.from_pretrained(
            f"/mnt/petrelfs/zhutong/smoe/resources/llama-3-8b-mixtral-{moe_type}-56e-top8",
        )
