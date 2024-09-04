#!/usr/bin/bash

#SBATCH --job-name=convert-attn
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
#  folder_name="split-gradient-max-ShareFalse-8MoE-Top2-Scale1.0"
  folder_name="split-gradient-max-ShareFalse-1Residual-7MoE-Top2-Scale1.0"
  model_path="/mnt/petrelfs/share_data/quxiaoye/llama_moe_v2/converted_models/${folder_name}"

  top_k_attn=7
  scale_factor_attn=1.0
  save_path="/mnt/petrelfs/share_data/quxiaoye/llama_moe_v2/converted_models/${folder_name}-AttnMoE-Top${top_k_attn}-Scale${scale_factor_attn}"

  srun python smoe/entrypoint/expert_construction_v2/convert/convert_to_mixtral_attn_moe.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --top_k_attn ${top_k_attn} \
    --scale_factor_attn ${scale_factor_attn}
}
