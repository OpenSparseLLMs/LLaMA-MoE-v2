#!/usr/bin/bash

#SBATCH --job-name=dpo_mb_64e_top8
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=auto
# reserved spot auto

num_nodes=1        # should match with --nodes
num_gpu_per_node=8 # should match with --gres
export OMP_NUM_THREADS=4
export LOGLEVEL=INFO

{
  output_router_logits=False # do not add gate loss
  router_aux_loss_coef=0.001
  freeze_gate=True

  beta=0.1 # DOP hyparam
  learning_rate=8e-6

  per_device_train_batch_size=1
  per_device_eval_batch_size=1
  gradient_accumulation_steps=16
  num_train_epochs=1

  #  dataset_dir_or_path="argilla/ultrafeedback-binarized-preferences-cleaned"
  dataset_dir_or_path="/mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/data/combined"

  #  model_name_or_path="/mnt/petrelfs/share_data/quxiaoye/checkpoint-11000-sft"
  model_name_or_path="/mnt/petrelfs/huxuyang/LLaMA-MoE-v2/outputs/v2_mixtral/mb_16e_top4/3357283/checkpoint-3000"

  task_name=$SLURM_JOB_NAME
  model_type="auto"
  base_dir="outputs/v2_mixtral"
  output_dir="${base_dir}/${task_name}-beta${beta}-lr${learning_rate}/$SLURM_JOB_ID"
  comment="llama-3-8b-instruct to mixtral-no-megablocks, 64 experts, top8"

  mkdir -p $output_dir
  scontrol write batch_script $SLURM_JOBID $output_dir/sbatch.sh
  git diff >$output_dir/diff.patch
  env >$output_dir/env
  echo -e "Job ID: ${SLURM_JOB_ID}\n\nLog: logs/${task_name}-$SLURM_JOB_ID.log\n\nGit commit: $(git log -1 --oneline)\n\nGit branch: $(git branch | grep "*")\n\nComment: ${comment}" >$output_dir/comment.txt
  ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $output_dir/log.log
  echo "$SLURM_JOB_ID" >$base_dir/latest.jobid
  ln -snf $output_dir $base_dir/latest.dir
  ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $base_dir/latest.log

  nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
  nodes_array=($nodes)
  head_node=${nodes_array[0]}
  echo "Node: $head_node"

  srun torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29522 \
    -m smoe.entrypoint.dpo.train_dpo \
    --output_router_logits ${output_router_logits} \
    --router_aux_loss_coef ${router_aux_loss_coef} \
    --freeze_gate ${freeze_gate} \
    --evaluation_strategy no \
    --run_name $task_name \
    --beta ${beta} \
    --model_type $model_type \
    --model_name_or_path $model_name_or_path \
    --dataset_name $dataset_dir_or_path \
    --output_dir $output_dir \
    --deepspeed conf/deepspeed/bf16_zero2_default.json \
    --seed 114514 \
    --bf16 True \
    --tf32 True \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${num_train_epochs} \
    --save_strategy no \
    --save_steps 2333333333333333333333333333333333 \
    --save_total_limit 0 \
    --learning_rate ${learning_rate} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --logging_first_step \
    --no_remove_unused_columns
}

#    --deepspeed conf/deepspeed/bf16_zero1.json \
#    --deepspeed conf/deepspeed/bf16_zero2_default.json \
#    --deepspeed conf/deepspeed/bf16_zero3.json \

#    --save_strategy no \
#    --save_steps 2333333333333333333333333333333333 \
#    --save_total_limit 0 \

#    --save_strategy steps \
#    --save_steps 1000 \
#    --save_total_limit 1 \
