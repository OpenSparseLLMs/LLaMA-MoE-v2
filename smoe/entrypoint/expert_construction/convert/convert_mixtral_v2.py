import argparse

import torch

from smoe.utils.expert_construction.convert_llama_to_mixtral import convert_safetensors

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--neuron_indices_file', type=str, default=None)
    parser.add_argument('--gate_weights_file', type=str, default=None)

    parser.add_argument('--num_experts', type=int, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--scale_factor', type=float, default=None)
    parser.add_argument('--num_moe_contract_layers', type=int, default=0)
    parser.add_argument('--moe_implementation_type', type=str, default='modulelist', choices=["modulelist"])

    args = parser.parse_args()
    print(args, "\n")

    convert_safetensors(
        args.model_path,
        args.save_path,
        num_experts=args.num_experts,
        top_k=args.top_k,
        scale_factor=args.scale_factor,
        num_moe_contract_layers=args.num_moe_contract_layers,
        moe_type=args.moe_implementation_type,
        neuron_indices=None if args.neuron_indices_file is None else torch.load(args.neuron_indices_file),
        gate_weights=None if args.gate_weights_file is None else torch.load(args.gate_weights_file),
    )
    print("Done!")
# fmt: on
