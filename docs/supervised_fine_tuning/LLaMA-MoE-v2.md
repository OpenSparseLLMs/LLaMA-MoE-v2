# 🌴 Setup

dependencies:

cuda: 11.8
python: 3.11.10

Just follow the installation guide in [README.md](..%2F..%2FREADME.md), which can be simplified as:

```bash
conda create -n smoe python=3.11
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
git git@github.com:LLaMA-MoE/LLaMA-MoE-v2.git
cd LLaMA-MoE-v2
pip install -e .[dev]
```

Finally, ensure that you environments satisfy:

```
deepspeed==0.14.4
flash-attn==2.6.1
torch==2.3.1
triton==2.3.1
transformers==4.42.4
```

For disabling `wandb` during training, you may run the following command:

```bash
$ wandb disabled
```

## 🗃️ Data Preparation

Run the following commands, and the data would be prepared in `resources/OpenHermes-2.5/openhermes2_5.jsonl` .

```bash
$ mkdir resources
$ huggingface-cli download teknium/OpenHermes-2.5 --repo-type dataset --local-dir resources/OpenHermes-2.5 --local-dir-use-symlinks False
```
Create a file named `json2jsonl.py` in the `resources/OpenHermes-2.5/` directory with the following content:
```python
from smoe.utils.io import load_json, dump_jsonlines

data = load_json("resources/OpenHermes-2.5/openhermes2_5.json")
dump_jsonlines(data, "resources/OpenHermes-2.5/openhermes2_5.jsonl")
```
Use the following command to run the script:
```bash
$ srun -p MoE python resources/OpenHermes-2.5/json2jsonl.py
```

## 🧃 Model Preparation (Converting dense models to MoE)

- Vanilla LLaMA-MoE-v2: `sbatch scripts/expert_construction/convert/convert_mixtral_v2.sh`
- Residual LLaMA-MoE-v2: `sbatch scripts/expert_construction/convert/convert_mixtral_residual_v2.sh`

For more information, please refer to [Expert Construction docs](../expert_construction/README.md).

## 🚀 Training

- packedpad
 Check the settings in `scripts/sft/sft_8e_top2_base_res_pad.sh` and run `sbatch scripts/sft/sft_8e_top2_base_res_pad.sh` .
- nopad
 Check the settings in `scripts/sft/sft_8e_top2_base_res.sh` and run `sbatch scripts/sft/sft_8e_top2_base_res.sh` .

`##SBATCH` means the slurm setting is not activated.

## 🛫 Evaluation

```bash
$ git clone https://github.com/EleutherAI/lm-evaluation-harness
$ cd lm-evaluation-harness
$ git checkout d14b36e81aea4cef
$ pip install -e .
# copy the scripts in smoe - `scripts/sft/eval` to here
# change the model dir and evaluation settings
$ bash run.sh
```
