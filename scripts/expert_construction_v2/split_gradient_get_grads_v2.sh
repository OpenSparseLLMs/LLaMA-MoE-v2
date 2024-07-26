#!/usr/bin/bash

#SBATCH --job-name=get_grads
#SBATCH --output=logs_split/%x-%j.log
#SBATCH --error=logs_split/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=auto

# reserved spot auto

{
  num_nodes=1        # should match with --nodes
  num_gpu_per_node=4 # should match with --gres
  export OMP_NUM_THREADS=4
  export LOGLEVEL=INFO

  export TOKENIZERS_PARALLELISM=false
  export NCCL_TIMEOUT=1800000

  model_type="llama"
  model_path=/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct
  dataset_dir_or_path="/mnt/petrelfs/share_data/quxiaoye/llama_moe_v2/OpenHermes-2.5/openhermes2_5.jsonl"

  seed=12345467
  per_device_train_batch_size=1
  model_max_length=4096
  max_steps=16

  folder_name="8experts-0.2jitter"

  gate_weights_file="/mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/outputs/v2_mixtral_gate/${folder_name}/results/gate_weights.pt"
  output_dir="/mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/outputs/v2_mixtral_gate/${folder_name}"
  save_path="${output_dir}/results"

  nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
  nodes_array=($nodes)
  head_node=${nodes_array[0]}
  head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
  echo "Node: $head_node"
  echo "Node IP: $head_node_ip"

  srun torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29518 \
    smoe/entrypoint/expert_construction_v2/split_gradient_get_grads_v2.py \
    --model_name_or_path ${model_path} \
    --model_type ${model_type} \
    --dataset_dir_or_path ${dataset_dir_or_path} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --seed ${seed} \
    --bf16 \
    --max_steps ${max_steps} \
    --model_max_length ${model_max_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --torch_dtype bfloat16 \
    --report_to none \
    --gate_weights_file ${gate_weights_file} \
    --save_path ${save_path}
}
