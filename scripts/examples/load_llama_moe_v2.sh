#!/usr/bin/bash

#folder_name="split-gradient-max-ShareFalse-16MoE-Top4-Scale1.0"
folder_name="split-gradient-max-ShareFalse-1Residual-7MoE-Top2-Scale1.0"
model_path="/mnt/petrelfs/share_data/quxiaoye/llama_moe_v2/converted_models/${folder_name}"

gpus=0
cpus=8
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=test --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.examples.load_llama_moe_v2 \
  --tokenizer_path ${model_path} \
  --model_path ${model_path}
