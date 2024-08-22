import argparse
import os

import torch
from tqdm import tqdm
from transformers import LlamaConfig

from smoe.utils.expert_construction.expert_split_residual import GradientSplitResidual
from smoe.utils.io import create_dir
from smoe.utils.operations.operation_string import str2bool


class GradientSplitResidualV2(GradientSplitResidual):
    # Here we only use the `split` function in `GradientSplit`.
    # Other functions may raise errors as the format of `self.labels` is changed.
    def __init__(self, config, layer, score_list):
        super().__init__(config, None, layer, score_list)

    def split(
        self,
        expert_num_moe,
        expert_num_residual,
        expert_size,
        criterion="min",
        share_neurons=False,
    ):
        super().split(
            expert_num_moe, expert_num_residual, expert_size, criterion, share_neurons
        )

        # 修改一个小细节，将结果以字典形式保存
        new_labels = {}

        new_labels["residual"] = [
            i for neuron_ids in self.labels[:expert_num_residual] for i in neuron_ids
        ]
        for expert_id in range(expert_num_moe):
            new_labels[expert_id] = self.labels[expert_num_residual + expert_id]

        self.labels = new_labels
        print(self.labels)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--score_file', type=str)
    parser.add_argument('--visualization_path', type=str, default=None)
    parser.add_argument('--expert_size', type=int, default=None)
    parser.add_argument('--num_experts_moe', type=int, default=None)
    parser.add_argument('--num_experts_residual', type=int, default=None)

    parser.add_argument('--criterion', type=str, default="max", choices=("min", "max"))
    parser.add_argument('--share_neurons', type=str, default="False")

    args = parser.parse_args()
    args.share_neurons = str2bool(args.share_neurons)
    print(args, "\n")

    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)

    print("Loading importance scores...")
    all_importance_scores = torch.load(args.score_file)

    # START
    neuron_indices = {}
    for i in tqdm(range(config.num_hidden_layers)):
        # get scores
        this_layer_scores = all_importance_scores[i]
        score_list = [this_layer_scores[j] for j in range(len(this_layer_scores))]

        # check configs
        assert args.num_experts_moe + args.num_experts_residual == len(score_list)

        if args.expert_size is None:
            args.expert_size = score_list[0].numel() // (args.num_experts_moe + args.num_experts_residual)

        # start split
        split = GradientSplitResidualV2(args, i, score_list)
        split.split(args.num_experts_moe, args.num_experts_residual, args.expert_size, criterion=args.criterion, share_neurons=args.share_neurons)
        neuron_indices[i] = split.labels

    # SAVE
    create_dir(args.save_path)
    torch.save(neuron_indices, os.path.join(args.save_path, "neuron_indices.pt"))
    print("Done.")
    # fmt: on
