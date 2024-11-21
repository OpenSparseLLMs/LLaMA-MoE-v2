#!/usr/bin/bash

#SBATCH --job-name=distribution
#SBATCH --output=logs_align/%x-%j.log
#SBATCH --error=logs_align/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=auto

# reserved spot auto

num_nodes=1        # should match with --nodes
num_cpus=16        # should match with --cpus-per-task
num_gpu_per_node=1 # should match with --gres
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
  #######################################################################################
  # This part is for getting metrics of dense LLaMA models, which acts as the reference for the alignment algorithm.

  model_type="llama"
  folder_name="Meta-Llama-3-8B-Instruct"
  model_path="/mnt/petrelfs/share_data/quxiaoye/models/${folder_name}"

  #######################################################################################
  # This part is for validating the mean & variance of aligned models.

  #  model_type="mixtral"
  #  folder_name="split-gradient-max-ShareFalse-8MoE-Top2-Scale1.0-Dense0-Aligned"
  #  folder_name="split-gradient-max-ShareFalse-8MoE-Top2-Scale1.0-Dense1-Aligned"
  #  folder_name="split-gradient-max-ShareFalse-1Residual-7MoE-Top1-Scale1.0-Dense0-Aligned"
  #  folder_name="split-gradient-max-ShareFalse-1Residual-7MoE-Top1-Scale1.0-Dense1-Aligned"
  #  folder_name="split-gradient-max-ShareFalse-8MoE-Top2-Scale1.0-Dense0-AttnMoE-Top7-Scale1.0-Aligned"
  #  folder_name="split-gradient-max-ShareFalse-1Residual-7MoE-Top1-Scale1.0-Dense0-AttnMoE-Top7-Scale1.0-Aligned"
  #  model_path="/mnt/petrelfs/share_data/quxiaoye/llama_moe_v2/converted_models/${folder_name}"
  #######################################################################################

  dataset_dir_or_path="/mnt/petrelfs/share_data/quxiaoye/llama_moe_v2/OpenHermes-2.5/openhermes2_5.jsonl"

  per_device_train_batch_size=4
  max_steps=500
  model_max_length=4096

  echo "Maximum number of possible tokens: $((${num_gpu_per_node} * ${per_device_train_batch_size} * ${max_steps} * ${model_max_length})) (paddings are taken into account here)"

  output_dir="/mnt/petrelfs/share_data/quxiaoye/llama_moe_v2/v2_mixtral_alignment"
  output_dir="${output_dir}/distribution"
  save_path="${output_dir}/${folder_name}"

  srun torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:$port \
    smoe/entrypoint/expert_construction/align/get_hidden_distribution.py \
    --model_name_or_path ${model_path} \
    --model_type ${model_type} \
    --dataset_dir_or_path ${dataset_dir_or_path} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --bf16 \
    --max_steps ${max_steps} \
    --model_max_length ${model_max_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --torch_dtype bfloat16 \
    --report_to none \
    --save_path ${save_path} \
    --attn_implementation "eager"
  #    --attn_implementation "flash_attention_2"
}
