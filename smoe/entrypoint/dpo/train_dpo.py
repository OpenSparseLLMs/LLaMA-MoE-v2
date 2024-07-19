# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/train_dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns
"""
import copy
import logging
import multiprocessing
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer, RichProgressCallback
from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser

from smoe.entrypoint.sft.train_sft_llama3 import ModelArguments, get_model_and_tokenizer, trainer_save_model_safe
from smoe.utils.operations.operation_string import str2bool

TRL_USE_RICH = str2bool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


@dataclass
class DPOMoEArguments(DPOConfig):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    freeze_gate: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the gate during training."},
    )
    save_final_ckpt: bool = field(
        default=True,
        metadata={"help": "Whether to save final checkpoint."},
    )
    # For MoE Gate Loss
    output_router_logits: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to add gate loss for backward."},
    )  # this toggle controls whether to add the gate loss to the final loss. reference see: https://huggingface.co/docs/trl/dpo_trainer#for-mixture-of-experts-models-enabling-the-auxiliary-loss
    router_aux_loss_coef: Optional[float] = field(
        default=None,
        metadata={"help": "MoE gate loss weight."},
    )


if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOMoEArguments, ModelArguments))
    args, training_args, model_args = parser.parse_args_and_config()

    # copy & set configs
    training_args.max_length = training_args.model_max_length
    training_args.max_prompt_length = 128  # default

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################################################################
    # Model & Tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_args.model_type,
        model_args.model_name_or_path,
        tokenizer_path=model_args.tokenizer_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side=model_args.padding_side,
        torch_dtype=model_args.torch_dtype,
        additional_config=model_args.additional_config,
        attn_impl=model_args.attn_impl,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    if training_args.freeze_gate:
        for name, param in model.named_parameters():
            if ".gate." in name:
                param.requires_grad = False
    if args.ignore_bias_buffers:  # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool]

    # Set config
    if training_args.output_router_logits is not None:
        model.config.output_router_logits = training_args.output_router_logits
    if training_args.router_aux_loss_coef is not None:
        model.config.router_aux_loss_coef = training_args.router_aux_loss_coef

    # Reference Model
    ref_model = copy.deepcopy(model)

    ###############################################################
    # Optional rich context managers
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = nullcontext() if not TRL_USE_RICH else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")

    ###############################################################
    # Dataset
    ds = load_dataset(args.dataset_name)
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))


    def process(row):
        row["prompt"] = tokenizer.apply_chat_template(row["chosen"][:-1], tokenize=False)
        row["chosen"] = tokenizer.apply_chat_template([row["chosen"][-1]], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template([row["rejected"][-1]], tokenize=False)
        return row


    ds = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
    )
    print(list(ds.keys()))
    train_dataset = ds[args.dataset_train_split]
    eval_dataset = ds[args.dataset_test_split] if "test" in ds.keys() else None

    ##############################################################
    # Training
    with init_context:
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    # Save model
    if training_args.save_final_ckpt:
        with save_context:
            trainer.accelerator.print("training finished, dumping model")
            model.config.use_cache = True
            trainer.save_state()
            if trainer.is_deepspeed_enabled:
                trainer.save_model()
            else:
                trainer_save_model_safe(trainer)
