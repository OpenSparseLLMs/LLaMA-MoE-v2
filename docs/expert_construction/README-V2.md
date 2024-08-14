# Expert Construction of LLaMA-MoE-V2

This documentation provides the procedures to convert a LLaMA model to LLaMA-MoE-V2.

## Split

### Gradient Split Plus

First, run `scripts/expert_construction_v2/get_gates/hidden_clustering.sh` to cluster the hidden_states of all layer inputs.

Then, run `scripts/expert_construction_v2/split/split_gradient_get_grads_v2.sh` to get the intermediate neuron-wise importance scores for each cluster in each layer.

Finally, `run scripts/expert_construction_v2/split/split_gradient_v2.sh` to get the indices of all experts in all layers.

## Convert

### Convert to Vanilla LLaMA-MoE-V2

Run `scripts/expert_construction_v2/convert/convert_mixtral_v2.sh`. By default the neurons are sequentially & uniformly split to form experts.

If you want to specify the expert-wise neurons, please pass `--neuron_indices_file`.

If you want to specify the layer-wise gate weights, please pass `--gate_weights_file`.

(P.S., for *Gradient Split Plus*, both the above args should be passed.)

### Convert to Residual LLaMA-MoE-V2

Coming soon... 🙀
