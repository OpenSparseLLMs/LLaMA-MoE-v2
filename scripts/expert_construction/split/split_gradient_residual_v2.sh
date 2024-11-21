#!/usr/bin/bash

#SBATCH --job-name=split-grad-residual
#SBATCH --output=logs_split/%x-%j.log
#SBATCH --error=logs_split/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --quotatype=auto

# reserved spot auto

{
  model_path="/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct"

  num_experts_moe=7
  num_experts_residual=1

  criterion="max"
  share_neurons="False"

  #  folder_name="${num_experts_moe}experts-0.8jitter-l2"
  folder_name="${num_experts_moe}experts-0.4jitter-l2"
  #  folder_name="${num_experts_moe}experts-0.0jitter-l2"

  score_file="/mnt/petrelfs/huxuyang/push/LLaMA-MoE-v2/resources/llama_moe_v2/v2_mixtral_gate/${folder_name}/results/importance_scores.pt"
  output_dir="/mnt/petrelfs/huxuyang/push/LLaMA-MoE-v2/resources/llama_moe_v2/v2_mixtral_gate/${folder_name}"
  save_path="${output_dir}/results/split-gradient-${criterion}-Share${share_neurons}-${num_experts_residual}Residual-${num_experts_moe}MoE"

  srun python smoe/entrypoint/expert_construction/split/split_gradient_residual_v2.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --score_file ${score_file} \
    --num_experts_moe ${num_experts_moe} \
    --num_experts_residual ${num_experts_residual} \
    --criterion ${criterion} \
    --share_neurons ${share_neurons}
}
