#!/usr/bin/bash

#SBATCH --job-name=get_grads
#SBATCH --output=logs_split/%x-%j.log
#SBATCH --error=logs_split/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --quotatype=reserved

# reserved spot auto

num_nodes=1        # should match with --nodes
num_gpu_per_node=2 # should match with --gres
export OMP_NUM_THREADS=4
export LOGLEVEL=INFO

{
  # @Desc 此脚本用于获取一个指定区间且未被占用的随机端口号
  # @Author Hellxz <hellxz001@foxmail.com>

  function Listening { #判断当前端口是否被占用，没被占用返回0，反之1
    TCPListeningnum=$(netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l)
    UDPListeningnum=$(netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l)
    ((Listeningnum = TCPListeningnum + UDPListeningnum))
    if [ $Listeningnum == 0 ]; then
      echo "0"
    else
      echo "1"
    fi
  }

  function get_random_port { #得到随机端口
    PORT=0
    while [ $PORT == 0 ]; do
      temp_port=$(shuf -i $1-$2 -n1) #指定区间随机数
      if [ $(Listening $temp_port) == 0 ]; then
        PORT=$temp_port
      fi
    done
    echo "$PORT"
  }

  port=$(get_random_port 29500 29600) #任取一个未占用端口号
  echo "Port: $port"

  nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
  nodes_array=($nodes)
  head_node=${nodes_array[0]}
  head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
  echo "Node: $head_node"
  echo "Node IP: $head_node_ip"
}

{
  model_type="llama"
  model_path="/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct"
  dataset_dir_or_path="/mnt/petrelfs/share_data/quxiaoye/llama_moe_v2/OpenHermes-2.5/openhermes2_5.jsonl"

  seed=2147483648
  per_device_train_batch_size=1
  model_max_length=4096
  max_steps=256

  #  folder_name="8experts-0.0jitter-l2"
  #  folder_name="8experts-0.2jitter-l2"
  #  folder_name="8experts-0.4jitter-l2"
  #  folder_name="8experts-0.6jitter-l2"

  #  folder_name="16experts-0.0jitter-l2"
  #  folder_name="16experts-0.2jitter-l2"
  #  folder_name="16experts-0.4jitter-l2"
  folder_name="16experts-0.6jitter-l2"

  #  folder_name="16experts-0.0jitter-cos"
  #  folder_name="16experts-0.2jitter-cos"
  #  folder_name="16experts-0.4jitter-cos"
  #  folder_name="16experts-0.6jitter-cos"

  gate_weights_file="/mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/outputs/v2_mixtral_gate/${folder_name}/results/gate_weights.pt"
  output_dir="/mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/outputs/v2_mixtral_gate/${folder_name}"
  save_path="${output_dir}/results"

  srun torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:$port \
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
