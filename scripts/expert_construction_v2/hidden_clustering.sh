#!/usr/bin/bash

#SBATCH --job-name=clustering
#SBATCH --output=logs_split/%x-%j.log
#SBATCH --error=logs_split/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=auto

#SBATCH -x SH-IDCA1404-10-140-54-122

# reserved spot auto

{
  num_nodes=1        # should match with --nodes
  num_cpus=16        # should match with --cpus-per-task
  num_gpu_per_node=1 # should match with --gres
  export OMP_NUM_THREADS=4
  export LOGLEVEL=INFO

  export TOKENIZERS_PARALLELISM=false
  export NCCL_TIMEOUT=3600000

  model_type="llama"
  model_path=/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct
  dataset_dir_or_path="/mnt/petrelfs/share_data/quxiaoye/llama_moe_v2/OpenHermes-2.5/openhermes2_5.jsonl"

  per_device_train_batch_size=4
  max_steps=16 # the total number of samples shouldn't be too large, as the KMeans is of n^2 complexity
  model_max_length=4096

  num_experts=16
  balance_jitter_factor=0.6 # hyper-parameter for adjusting the cluster size, will affect the initialization of gate weights. (0.0 for strictly balanced, however the performance may be worse.)
  max_iter=100
  random_state=123456789
  n_jobs=${num_cpus}

  output_dir="/mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/outputs/v2_mixtral_gate"
  output_dir="${output_dir}/${num_experts}experts-${balance_jitter_factor}jitter"
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
    smoe/entrypoint/expert_construction_v2/hidden_feature_clustering.py \
    --model_name_or_path ${model_path} \
    --model_type ${model_type} \
    --dataset_dir_or_path ${dataset_dir_or_path} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --seed ${random_state} \
    --bf16 \
    --max_steps ${max_steps} \
    --model_max_length ${model_max_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --torch_dtype bfloat16 \
    --report_to none \
    --save_path ${save_path} \
    --num_experts ${num_experts} \
    --balance_jitter_factor ${balance_jitter_factor} \
    --max_iter ${max_iter} \
    --random_state ${random_state} \
    --n_jobs ${n_jobs}
}
