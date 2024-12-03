# Expert Construction of LLaMA-MoE-V2

This documentation provides the procedures to convert a dense LLaMA model to LLaMA-MoE-V2 models.

(1) Convert to MLP MoE, including two architectures: Vanilla MLP MoE and Residual MLP MoE.

For the Residual MLP MoE, we can use the *Gradient Split Plus* strategy to split the neurons. If you want to quickly undertand the split prrocess, we suggest using the Vanilla MLP MoE. 

(2) Convert to Attention MoE.



## 1. Convert to MLP MoE

### Convert to Vanilla MLP MoE

Just run the following script:

```bash
sbatch scripts/expert_construction/convert/convert_mixtral_v2.sh
```

Remember to change the following variables:

```shell
num_experts="" # number of experts in each MoE layer
top_k="" # number of activate experts in each MoE layer
scale_factor="" # hyper-parameter for balancing the experts
num_moe_contract_layers="" # number of MoE Contract Layers

model_path="" # path to the LLaMA checkpoint
save_path="" # path to save the indices sets
```

### Convert to Residual MLP MoE

This is almost the same as the above. Just run the following script:

```bash
sbatch scripts/expert_construction/convert/convert_mixtral_residual_v2.sh
```

There are some arguments that you should notice:

- **`--gate_weights_file`:** This determines the initialization strategy of routers in the converted MoE model. If not specified, the MLP gates will be initialized randomly using *kaiming initialization*.
- **`--neuron_indices_file`:** This determines the indices of neurons in the original dense model for converted MLP experts. If not specified, the MLP experts will be split sequentially & uniformly (which is a very naive strategy).

You should pass both `--gate_weights_file` and `--neuron_indices_file` arguments, as this script is specifically designed for the *Gradient Split Plus* strategy.

These files can be ontained with:

Get the router weights through k-means clustering on the `hidden_states` of all layer inputs by running:

```bash
sbatch scripts/expert_construction/get_gates/hidden_clustering.sh
```

Run the following script to get the importance scores of all experts:

```bash
sbatch scripts/expert_construction/split/split_gradient_get_grads_v2.sh
```


## 2. Convert to Attention MoE

We can convert an MLP MoE to further into an Attention MoE or directly convert a dense model to an Attention MoE.

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

