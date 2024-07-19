# 🌴 Setup

dependencies:

cuda: 11.8
python: 3.11.4

Just follow the installation guide in [README.md](..%2F..%2FREADME.md), which can be simplified as:

```bash
conda create -n smoe python=3.11
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
git clone git@github.com:pjlab-sys4nlp/llama-moe.git
cd llama-moe
pip install -e .[dev]
```

For special packages that are exclusive in v2, please follow:

1. Install `megablocks` and `stanford-stk`, the installation has to be done in the computing cluster: `srun -p MoE pip install megablocks==0.5.1`.
2. Install `scattermoe`: `git clone https://github.com/shawntan/scattermoe.git` and follow the instruction on the [official website](https://github.com/shawntan/scattermoe).

Finally, ensure that you environments satisfy:

```
deepspeed==0.14.4
flash-attn==2.6.1
megablocks==0.5.1
scattermoe==0.0.0 (installed locally)
torch==2.3.1
triton==2.3.1
transformers==4.42.4
```

For disabling `wandb` during training, you may run the following command:

```bash
$ wandb disable
```

## 🗃️ Data Preparation

Run the following commands, and the data would be prepared in `resources/OpenHermes-2.5/openhermes2_5.jsonl` .

```bash
$ huggingface-cli download teknium/OpenHermes-2.5 --repo-type dataset --local-dir resources/OpenHermes-2.5 --local-dir-use-symlinks False
$ srun -p MoE python resources/OpenHermes-2.5/json2jsonl.py
```

## 🧃 Model Preparation (Converting dense models to MoE)

Check and change the `num_experts`, `top_k`, `src_model_dir`, `tgt_model_dir_prefix`, and `tgt_moe_types` according to your settings.

After all settings are ready, run: `srun -p MoE python smoe/utils/expert_construction/convert_llama_to_mixtral.py` .

`tgt_moe_types`:
- modulelist: the raw and original implementation of Mixtral without 3rd-party training accelerations
- megablocks: enables the megablocks implementation: http://arxiv.org/abs/2211.15841
- scattermoe: the scattermoe implementation: http://arxiv.org/abs/2403.08245

## 🚀 Training

Check the settings in `scripts/v2_mixtral/mb_64e_top8.sh` and run `sbatch scripts/v2_mixtral/mb_64e_top8.sh` .

- `model_type` must be `auto` for enabling `trust_remote_code=True`.
- `##SBATCH` means the slurm setting is not activated.

## DPO

先跑`python smoe/entrypoint/dpo/merge_datasets.py`，合并数据集。注意数据集应该有相同的columns名称，不然合并时会报错。(这里应该也有更好的方法，但时间有限，我就先这样简单写了，你们后面正式搞的时候，应该要对每种数据都做下分别处理)

跑训练的话，就直接`bash scripts/v2_mixtral/dpo/mb_64e_top8_dpo.sh`。可以调以下参数：

```
output_router_logits: 控制是否将balance loss进行backward。现在只能为False，即不加balance loss。为True会报错，应该要同步改Trainer，这个需要你们来实现。
freeze_gate: 要不要冻结gate参数。现在为True，来弥补不加balance loss对平衡产生的影响。
beta: DPO的一个超参，一般在0.1到0.5之间，我没调这个，用的默认的0.1，不过也可以调下。
learning_rate: 一般要小于等于微调阶段的最终学习率，目前搜出来8e-6比较好，如果后面微调阶段学习率有修改，可以以这个值为中心再搜索一波DPO学习率。
```

## 🛫 Evaluation

```bash
$ git clone https://github.com/EleutherAI/lm-evaluation-harness
$ cd lm-evaluation-harness
$ git checkout d14b36e81aea4cef
$ pip install -e .
# copy the scripts in smoe - `scripts/v2_mixtral/eval` to here
# change the model dir and evaluation settings
$ bash run.sh
```

## 🗺️ Roadmap & Instructions

- **balance loss**: to enable the balance loss, change the `output_router_logits` in a model's `config.json` to `true` (e.g. `resources/llama-3-8b-mixtral-megablocks-56e-top8/config.json`)
- **sequence length**: try to increase the `model_max_length` to `4096` as you can
- **megablocks & scattermoe**: there may be bugs and the evaluation results are bad than `modulelist`, but the training process is available with 2.6x acceleration and the loss goes down correctly
- **DPO Alignment**: Done
- **More Split Strategies**
- **Attention MoE**
- **More diversified & powerful data**
