set -x

multi_eval() {
  id_name=$1
  result_name=$2
  model_path=$3

  mkdir logs/$id_name
  mkdir results/$id_name
  for task_name in extend arc mmlu; do
    sleep 1
    nohup srun -p MoE --gres gpu:1 -J "$task_name" bash eval.sh $task_name $model_path True results/$id_name/$result_name 1>logs/$id_name/$result_name-$task_name.log 2>&1 &
  done
}

single_eval() {
  task_name=$1
  id_name=$2
  result_name=$3
  model_path=$4

  mkdir logs/$id_name
  mkdir results/$id_name
  nohup srun -p MoE --gres gpu:1 -J "$task_name" bash eval.sh $task_name $model_path True results/$id_name/$result_name 1>logs/$id_name/$result_name-$task_name.log 2>&1 &
}

for lr in "8e-5" "4e-5" "2e-5" "1e-5" "8e-6" "4e-6" "2e-6" "1e-6"; do
  ##############################################################
  folder_name="dpo_mb_64e_top8-beta0.1-lr${lr}"
  run_id="3361280"

  result_name=${folder_name}
  model_path="/mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/outputs/v2_mixtral/${folder_name}/${run_id}"

  ##############################################################
  #  result_name="baseline-sft"
  #  model_path="/mnt/petrelfs/share_data/quxiaoye/checkpoint-11000-sft"

  ##############################################################

  log_dir="/mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/logs_eval"
  mkdir -p ${log_dir}

  for task_name in extend arc; do
    OMP_NUM_THREADS=8 srun --quotatype=auto --partition=MoE --job-name=eval --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 -c 16 --kill-on-bad-exit=1 \
      bash /mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/scripts/v2_mixtral/eval/eval.sh \
      $task_name \
      $model_path True \
      ${log_dir}/${result_name} &
    sleep 1
  done
done
