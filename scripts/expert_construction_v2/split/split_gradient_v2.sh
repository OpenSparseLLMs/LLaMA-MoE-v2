#!/usr/bin/bash

#SBATCH --job-name=split-grad
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

  criterion="max"
  share_neurons="False"

#  folder_name="8experts-0.0jitter-l2"
#  folder_name="8experts-0.4jitter-l2"
#  folder_name="8experts-0.8jitter-l2"
#  folder_name="16experts-0.0jitter-l2"
#  folder_name="16experts-0.4jitter-l2"
#  folder_name="16experts-0.8jitter-l2"
#  folder_name="16experts-0.0jitter-cos"
  folder_name="16experts-0.4jitter-cos"
#  folder_name="16experts-0.8jitter-cos"

  score_file="/mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/outputs/v2_mixtral_gate/${folder_name}/results/importance_scores.pt"
  output_dir="/mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/outputs/v2_mixtral_gate/${folder_name}"
  save_path="${output_dir}/results/split-gradient-${criterion}-Share${share_neurons}"

  srun python smoe/entrypoint/expert_construction_v2/split/split_gradient_v2.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --score_file ${score_file} \
    --criterion ${criterion} \
    --share_neurons ${share_neurons}
}
