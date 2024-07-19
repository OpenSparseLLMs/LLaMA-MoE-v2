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
