#!/usr/bin/bash

#SBATCH --job-name=moe-res
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=llm_s
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved

# export WANDB_PROJECT="v2-mixtral"

{
    task_name=$SLURM_JOB_NAME
    # model_type="v2_mixtral"
    model_type="v2_mixtral_residual"

    # dataset_dir_or_path="/mnt/petrelfs/quxiaoye/models/OpenHermes-2.5/openhermes2_5.jsonl"
    dataset_dir_or_path="/mnt/petrelfs/quxiaoye/models/sft-v2/data.jsonl"

    # model_name_or_path="/mnt/petrelfs/quxiaoye/models/llama-3-8b-mixtral-modulelist-8e-top2"  # base version
    model_name_or_path="/mnt/petrelfs/quxiaoye/models/split-gradient-max-ShareFalse-1Residual-7MoE-Top1" # residual version
    # model_name_or_path="/mnt/petrelfs/quxiaoye/models/llama-3-8b-instruct-mixtral-modulelist-8e-top2"  # instruct version
    # model_name_or_path="/mnt/petrelfs/quxiaoye/models/llama-3-8b-instruct-mixtral-uniform-modulelist-8e-top2"  # uniform instruct version
    # model_name_or_path="/mnt/petrelfs/quxiaoye/models/llama-3-8b-instruct-mixtral-kaiming-modulelist-8e-top2"  # kaiming instruct version

    comment="llama-3-8b to mixtral-no-megablocks, 8 experts, top2"
    base_dir="outputs/v2_mixtral"
    output_dir="${base_dir}/${task_name}/$SLURM_JOB_ID"
    mkdir -p $output_dir
    scontrol write batch_script $SLURM_JOBID $output_dir/sbatch.sh
    git diff > $output_dir/diff.patch
    env > $output_dir/env
    echo -e "Job ID: ${SLURM_JOB_ID}\n\nLog: logs/${task_name}-$SLURM_JOB_ID.log\n\nGit commit: $(git log -1 --oneline)\n\nGit branch: $(git branch | grep "*")\n\nComment: ${comment}" > $output_dir/comment.txt
    ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $output_dir/log.log
    echo "$SLURM_JOB_ID" > $base_dir/latest.jobid
    ln -snf $output_dir $base_dir/latest.dir
    ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $base_dir/latest.log

    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    echo "Node: $head_node"

    srun torchrun \
    --nnodes 4 \
    --nproc_per_node 8 \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29522 \
        -m smoe.entrypoint.sft.train_sft_llama3 \
            --do_train \
            --freeze_gate False \
            --evaluation_strategy no \
            --run_name $task_name \
            --model_type $model_type \
            --model_name_or_path $model_name_or_path \
            --dataset_dir_or_path $dataset_dir_or_path \
            --output_dir $output_dir \
            --deepspeed conf/deepspeed/bf16_zero1.json \
            --seed 1227 \
            --bf16 True \
            --tf32 True \
            --torch_dtype bfloat16 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 2 \
            --num_train_epochs 3 \
            --save_strategy steps \
            --save_steps 1000 \
            --save_total_limit 12 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type cosine \
            --logging_steps 10 \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --save_only_model True \
            # --max_grad_norm 1.0  # Add this for nan gradient.
            # --report_to wandb

}

            # --gradient_accumulation_steps 8 \

            # --deepspeed conf/deepspeed/bf16_zero3.json \
            # --save_strategy steps \
            # --save_steps 9999999999999 \
