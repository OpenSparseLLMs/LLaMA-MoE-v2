# Expert Construction of LLaMA-MoE-V2

This documentation provides the procedures to convert a LLaMA model to LLaMA-MoE-V2.



## 1. Get Router Weights

### K-means Centroids

Get the router weights through k-means clustering on the `hidden_states` of all layer inputs by running:

```bash
sbatch scripts/expert_construction/get_gates/hidden_clustering.sh
```
Remember to change the following variables:

```shell
num_experts="" # number of experts in each MoE layer
balance_jitter_factor="" # hyper-parameter for adjusting the cluster size
distance_metric="" # metric for clustering, choices: `l2` `cos`

dataset_dir_or_path="" # path to dataset directory or a single jsonl file
model_path="" # path to the LLaMA checkpoint
output_dir="" # path to save the indices sets
```

### Random Features

Get the router weights through random selection of `hidden_states` among all layer inputs by running:

```bash
sbatch scripts/expert_construction/get_gates/random_selection.sh
```

Remember to change the following variables:

```shell
num_experts="" # number of experts in each MoE layer

dataset_dir_or_path="" # path to dataset directory or a single jsonl file
model_path="" # path to the LLaMA checkpoint
output_dir="" # path to save the indices sets
```

## 2. Split Neurons

### Gradient Split Plus

This strategy splits the neurons according to their importance scores on different token batches.

You should first run the following script to get the importance scores of all experts:

(Remember to pass the `--folder_name` argument, where the file is generated from the *Get Router Weights* step.)

```bash
sbatch scripts/expert_construction/split/split_gradient_get_grads_v2.sh
```

Remember to change the following variables:

```shell
dataset_dir_or_path="" # path to dataset directory or a single jsonl file
model_path="" # path to the LLaMA checkpoint
folder_name="" # name to router weights file
gate_weights_file="" # path to the router weights file generated above
output_dir="" # path to save the indices sets
```


Then, you can run the following scripts to get the indices splits accordingly:

(Remember to pass the `--score_file` argument, where the file is generated from the above step.)

| MoE Type | Script                                                       |
| -------- | ------------------------------------------------------------ |
| Vanilla  | `sbatch scripts/expert_construction/split/split_gradient_v2.sh` |
| Residual | `sbatch scripts/expert_construction/split/split_gradient_residual_v2.sh` |

Remember to change the following variables:

```shell
# Vanilla
model_path="" # path to the LLaMA checkpoint
num_experts="" # number of experts in each MoE layer
score_file="" # path to importance scores file generated above
output_dir="" # path to save the indices sets

# Residual
model_path="" # path to the LLaMA checkpoint
num_experts_moe="" # number of experts in each MoE layer
num_experts_residual="" # number of residual experts in each MoE layer
score_file="" # path to importance scores file generated above
output_dir="" # path to save the indices sets
```


## 3. Convert MoE

### Convert to Vanilla MLP MoE

Just run the following script:

```bash
sbatch scripts/expert_construction/convert/convert_mixtral_v2.sh
```

For vanilla MLP MoE conversion, we can 

There are some arguments that you should notice:

- **`--gate_weights_file`:** This determines the initialization strategy of routers in the converted MoE model. If not specified, the MLP gates will be initialized randomly using *kaiming initialization*.
- **`--neuron_indices_file`:** This determines the indices of neurons in the original dense model for converted MLP experts. If not specified, the MLP experts will be split sequentially & uniformly (which is a very naive strategy).

Note that if the *Gradient Split Plus* strategy is used, you must specify `--gate_weights_file` as the path to the gate weights generated in the *Get Router Weights* step, and `--neuron_indices_file` as the generated `neuron_indices.pt` file accordingly.

Remember to change the following variables:

```shell
num_experts="" # number of experts in each MoE layer
top_k="" # number of activate experts in each MoE layer
scale_factor="" # hyper-parameter for balancing the experts
num_moe_contract_layers="" # number of MoE Contract Layers

model_path="" # path to the LLaMA checkpoint
neuron_indices_file="" # path to the gate weights file generated above
gate_weights_file="" # path to the neuron indices file generated above
save_path="" # path to save the indices sets
```

### Convert to Residual MLP MoE

This is almost the same as the above. Just run the following script:

```bash
sbatch scripts/expert_construction/convert/convert_mixtral_residual_v2.sh
```

The only difference is that you should always pass both `--gate_weights_file` and `--neuron_indices_file` arguments, as this script is specifically designed for the *Gradient Split Plus* strategy.


### Convert to Attention MoE

The conversion of Attention MoE is performed on an existing converted MoE model (where the MLP has already been converted into MoE). You just need to run the following script:

```bash
sbatch scripts/expert_construction/convert/convert_mixtral_attn_moe.sh
```

Note that the argument `--model_path` should be pointed to an already converted MoE model.

Remember to change the following variables:

```shell
top_k_attn="" # number of activate experts in each attention MoE layer
scale_factor_attn="" # hyper-parameter for balancing the experts

model_path="" # path to the converted MoE checkpoint
folder_name="" # name to the converted MoE checkpoint
save_path="" # path to save the indices sets
```

