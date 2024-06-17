import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from smoe.models.mixtral.configuration_mixtral import MixtralConfig


def is_safetensors_file(filepath):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    string = filepath.name
    return re.match(r"model-\d{5}-of-\d{5}.safetensors", string) is not None


FFN_TYPE_MAP = {
    "gate": "w1",
    "down": "w2",
    "up": "v1",
}


def convert_safetensors(
    model_dir,
    dump_dir,
    num_experts: int,
    top_k: int,
    enable_megablocks: bool = True,
):
    model_folder = Path(model_dir)
    dump_folder = Path(dump_dir)
    dump_folder.mkdir(parents=True, exist_ok=True)

    router_records = set()
    for filepath in model_folder.glob("*"):
        if is_safetensors_file(filepath):
            with safe_open(filepath, framework="pt", device="cpu") as f:
                tensors = {}
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if ".mlp." in key:
                        layer_idx, ffn_type = re.search(
                            r"model.layers.(\d+).mlp.(gate|up|down)_proj.weight", key
                        ).groups()
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
                        new_ffn_type = FFN_TYPE_MAP[ffn_type]
                        if enable_megablocks:
                            tensors[
                                f"model.layers.{layer_idx}.block_sparse_moe.experts.mlp.{new_ffn_type}"
                            ] = tensor
                        else:
                            expert_size = mid // num_experts
                            for expert_idx in range(num_experts):
                                if mid_idx == 0:
                                    expert_tensor = tensor[
                                        expert_idx
                                        * expert_size : (expert_idx + 1)
                                        * expert_size
                                    ].clone()
                                else:
                                    expert_tensor = tensor[
                                        :,
                                        expert_idx
                                        * expert_size : (expert_idx + 1)
                                        * expert_size,
                                    ].clone()
                                tensors[
                                    f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{new_ffn_type}.weight"
                                ] = expert_tensor
                    else:
                        tensors[key] = tensor
                save_file(tensors, dump_folder / filepath.name)
        elif filepath.name == "config.json":
            config = MixtralConfig.from_pretrained(filepath)
            config.num_experts_per_tok = top_k
            config.num_local_experts = num_experts
            config.router_aux_loss_coef = 1e-2
            config.act_rescale = True
            config.enable_megablocks = enable_megablocks
            config.intermediate_size = config.intermediate_size // num_experts
            config.save_pretrained(dump_folder)
        else:
            # cp to dump_dir
            shutil.copy2(filepath, dump_folder / filepath.name)


if __name__ == "__main__":
    # convert_safetensors(
    #     "/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct",
    #     "/mnt/petrelfs/zhutong/smoe/resources/llama-3-8b-mixtral-megablocks",
    #     num_experts=64,
    #     top_k=8,
    #     enable_megablocks=True,
    # )
    convert_safetensors(
        "/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct",
        "/mnt/petrelfs/zhutong/smoe/resources/llama-3-8b-mixtral-no-megablocks",
        num_experts=64,
        top_k=8,
        enable_megablocks=False,
    )
