set -x

mkdir logs
mkdir results


queue_eval() {
    result_name=$1
    model_path=$2
    nohup srun -p MoE --gres gpu:1 bash eval.sh smoe $model_path True results/$result_name 1>logs/$result_name.log 2>&1 &
}

multi_eval() {
    result_name=$1
    model_path=$2
    for task_name in extend arc mmlu;
    do
        sleep 1
        nohup srun -p MoE --gres gpu:1 -J "$task_name" bash eval.sh $task_name $model_path True results/$result_name 1>logs/$result_name-$task_name.log 2>&1 &
    done
}

single_eval() {
    task_name=$1
    result_name=$2
    model_path=$3
    nohup srun -p MoE --gres gpu:1 -J "$task_name" bash eval.sh $task_name $model_path True results/$result_name 1>logs/$result_name-$task_name.log 2>&1 &
}

{
    # multi_eval "56e_top8_1k_3245958" /mnt/petrelfs/zhutong/smoe/outputs/v2_mixtral/mb_64e_top8/3245958/checkpoint-1000
    # single_eval hellaswag "56e_top8_1k_3245958" /mnt/petrelfs/zhutong/smoe/outputs/v2_mixtral/mb_64e_top8/3245958/checkpoint-1000
    # multi_eval "llama-3-8b-instruct" /mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct
    # multi_eval "56e_top8_raw_split" /mnt/petrelfs/zhutong/smoe/resources/llama-3-8b-mixtral-no-megablocks-56e-top8
    # single_eval extend "56e_top8_2k_3245958" /mnt/petrelfs/zhutong/smoe/outputs/v2_mixtral/mb_64e_top8/3245958/checkpoint-2000
    # multi_eval "56e_top8_3k_3245958" /mnt/petrelfs/zhutong/smoe/outputs/v2_mixtral/mb_64e_top8/3245958/checkpoint-3000
    # single_eval arc "56e_top8_2k_3245958" /mnt/petrelfs/zhutong/smoe/outputs/v2_mixtral/mb_64e_top8/3245958/checkpoint-2000
    # single_eval mmlu "56e_top8_2k_3245958" /mnt/petrelfs/zhutong/smoe/outputs/v2_mixtral/mb_64e_top8/3245958/checkpoint-2000
    # multi_eval "scattermoe_56e_top8_2k_3274796" /mnt/petrelfs/zhutong/smoe/outputs/v2_mixtral/mb_64e_top8/3274796/checkpoint-2000
    # multi_eval "megablocks_56e_top8_3k_3277430" "/mnt/petrelfs/zhutong/smoe/outputs/v2_mixtral/mb_64e_top8/3277430/checkpoint-3000"
    # single_eval arc "megablocks_56e_top8_3k_3277430" "/mnt/petrelfs/zhutong/smoe/outputs/v2_mixtral/mb_64e_top8/3277430/checkpoint-3000"
    # single_eval extend "megablocks_56e_top8_3k_3277430" "/mnt/petrelfs/zhutong/smoe/outputs/v2_mixtral/mb_64e_top8/3277430/checkpoint-3000"
    single_eval extend "megablocks_56e_top8_4k_3277430" "/mnt/petrelfs/zhutong/smoe/outputs/v2_mixtral/mb_64e_top8/3277430/checkpoint-4000"
}
