"""
Modified from smoe/entrypoint/cpt/cpt_fpt.py
"""
import logging
import os
import pathlib
import socket
import sys
from dataclasses import dataclass, field
from typing import Optional

import accelerate
import datasets
import torch
import transformers
from k_means_constrained import KMeansConstrained
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    LlamaForCausalLM,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from smoe.data.collate_fn import fault_tolerance_data_collator
from smoe.entrypoint.cpt.cpt_fpt import MODEL_MAP
from smoe.entrypoint.sft.train_sft_llama3 import CachedJsonlDataset
from smoe.models.llama_moe.modeling_llama_moe import LlamaMoEForCausalLM
from smoe.models.llama_moe_residual import LlamaMoEResidualForCausalLM
from smoe.models.mixtral.modeling_mixtral import MixtralForCausalLM
from smoe.utils.config import EnhancedTrainingArguments, ModelArguments, parse_args
from smoe.utils.expert_construction.k_means_constrained_cos import KMeansConstrainedCos
from smoe.utils.io import create_dir
from smoe.utils.param import get_trainable_parameters

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    dataset_dir_or_path: str = field(
        default="data/merged",
        metadata={"help": "Path to dataset directory or a single jsonl file"},
    )
    model_max_length: int = field(
        default=2048,
    )


@dataclass
class ClusteringArguments:
    save_path: Optional[str] = field(default=None)
    num_experts: int = field(
        default=16,
        metadata={"help": "Number of experts (clusters)."},
    )
    balance_jitter_factor: float = field(
        default=0.1,
        metadata={
            "help": "The maximum tolerance of the size of clusters compared to the absolutely balanced situation."
        },
    )
    distance_metric: str = field(
        default="l2",
        metadata={
            "choices": ("l2", "cos"),
            "help": "Metric to calculate the distance between features.",
        },
    )
    max_iter: int = field(
        default=500,
        metadata={"help": "Number of iterations for K-Means to run."},
    )
    random_state: int = field(default=114514)
    n_jobs: int = field(
        default=-1,
        metadata={
            "help": "Number of CPUs to use for multiple initializations of KMeans clustering. (-1 denotes all)"
        },
    )


def prepare_model_and_data(model_args, data_args, training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    hostname = socket.gethostname()
    logger.warning(
        f"Global rank: {training_args.process_index}, "
        f"Host: {hostname}, IP: {socket.gethostbyname(hostname)}, "
        f"Process local rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"fp16 training: {training_args.fp16}, "
        f"bf16 training: {training_args.bf16}"
    )
    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args.to_json_string()}")

    # Set seed before initializing model.
    logger.info(f"Seed set to: {training_args.seed}")
    set_seed(training_args.seed)

    # model
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    if training_args.gradient_checkpointing:
        config.use_cache = False

    # tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "model_max_length": data_args.model_max_length,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "legacy": True if model_args.use_legacy_tokenizer else False,
        "trust_remote_code": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, **tokenizer_kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token

    # dataset
    datapath = pathlib.Path(data_args.dataset_dir_or_path)
    if not datapath.exists():
        raise ValueError(f"Dataset path {datapath} not found")
    elif datapath.is_file():
        logger.info(f"CachedJsonlDataset: {datapath}")
        train_dataset = CachedJsonlDataset(
            data_args.dataset_dir_or_path,
            tokenizer,
        )
    else:
        raise ValueError(f"Unknown dataset path type: {datapath}")
    logger.info("train dataset ready")

    logger.info("training example:")
    if hasattr(train_dataset, "take"):
        res = tokenizer.decode([x["input_ids"] for x in train_dataset.take(1)][0])
    else:
        for x in train_dataset:
            input_ids = x["input_ids"]
            break
        res = tokenizer.decode(input_ids)
    logger.info(res)

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        ModelClass = MODEL_MAP[model_args.model_type]

        # config._attn_implementation = "eager"
        # config._attn_implementation = "flash_attention_2"

        model: (
            LlamaForCausalLM
            | LlamaMoEForCausalLM
            | LlamaMoEResidualForCausalLM
            | MixtralForCausalLM
        ) = ModelClass.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params"
        )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model_vocab_size = model.get_output_embeddings().weight.size(0)
    if model_vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        raise ValueError(
            f"The model's vocab size ({model_vocab_size}) does not match with the"
            f" tokenizer ({len(tokenizer)})"
        )

    trainable_params, _ = get_trainable_parameters(model, verbose=True)
    training_args.num_training_params = trainable_params

    # 🔍 prepare data loader
    dataloader_params = {
        "batch_size": training_args.per_device_train_batch_size,
        "collate_fn": fault_tolerance_data_collator,
        "num_workers": 0,
        "pin_memory": True,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = RandomSampler(train_dataset)
        dataloader_params["drop_last"] = True
        dataloader_params["worker_init_fn"] = seed_worker

    dataloader = DataLoader(train_dataset, **dataloader_params)

    return model, dataloader


def main():
    model_args, data_args, training_args, clustering_args = parse_args(
        ModelArguments, DataArguments, EnhancedTrainingArguments, ClusteringArguments
    )

    model, dataloader = prepare_model_and_data(model_args, data_args, training_args)

    # 🔍 model check & prepare configs
    if not isinstance(model, LlamaForCausalLM):
        raise ValueError("For now the only supported model is LLaMA!")

    num_hidden_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    assert hidden_size % clustering_args.num_experts == 0
    balance_jitter_factor = max(0.0, clustering_args.balance_jitter_factor)

    # 🔍 prepare accelerator
    accelerator = accelerate.Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)

    # 🔍 begin layer-wise clustering
    all_gate_weights = {}

    with torch.no_grad():
        # 🔍 embedding layer
        ## get hidden states
        all_hidden_states = []
        all_gathered_hidden_states = []
        accelerator.print(f"Forward for embedding outputs!")

        for i, batch in tqdm(enumerate(dataloader)):
            if i >= training_args.max_steps:
                break
            hidden_states = accelerator.unwrap_model(model).model.embed_tokens(
                batch["input_ids"]
            )
            gathered_hidden_states = (
                accelerator.gather(hidden_states).cpu().float()
            )  # 🔍 all gather across devices
            gathered_hidden_states = gathered_hidden_states.reshape(-1, hidden_size)
            if not accelerator.is_main_process:
                gathered_hidden_states = (
                    None  # only store the hidden states on the main process
                )
            hidden_states = hidden_states.cpu()

            all_hidden_states.append(hidden_states)
            all_gathered_hidden_states.append(gathered_hidden_states)

        accelerator.print(f"Forward for embedding outputs done!")

        ## perform clustering
        if accelerator.is_main_process:
            accelerator.print(f"Clustering for embedding outputs!")
            all_gathered_hidden_states = torch.cat(all_gathered_hidden_states, dim=0)
            all_gathered_hidden_states = all_gathered_hidden_states.numpy()

            num_features = all_gathered_hidden_states.shape[0]
            balanced_num_features = num_features // clustering_args.num_experts
            min_cluster_size = max(
                0, int(balanced_num_features * (1 - balance_jitter_factor))
            )
            max_cluster_size = min(
                num_features, int(balanced_num_features * (1 + balance_jitter_factor))
            )

            print("ideally balanced cluster size:", balanced_num_features)
            print("min cluster size:", min_cluster_size)
            print("max cluster size:", max_cluster_size)

            if clustering_args.distance_metric == "l2":
                kmeans = KMeansConstrained(
                    n_clusters=clustering_args.num_experts,
                    size_min=min_cluster_size,
                    size_max=max_cluster_size,
                    tol=1e-1,
                    max_iter=clustering_args.max_iter,
                    random_state=clustering_args.random_state,
                    n_jobs=clustering_args.n_jobs,
                    verbose=True,
                ).fit(all_gathered_hidden_states, None)
            elif clustering_args.distance_metric == "cos":
                kmeans = KMeansConstrainedCos(
                    n_clusters=clustering_args.num_experts,
                    size_min=min_cluster_size,
                    size_max=max_cluster_size,
                    tol=1e-1,
                    max_iter=clustering_args.max_iter,
                    random_state=clustering_args.random_state,
                    n_jobs=clustering_args.n_jobs,
                    verbose=True,
                ).fit(all_gathered_hidden_states, None)
            gate_weights = torch.from_numpy(kmeans.cluster_centers_)

            all_gate_weights[0] = gate_weights  # add to all weights
            accelerator.print(f"{gate_weights.shape}")

            accelerator.print(f"Clustering for embedding outputs done!")
        accelerator.wait_for_everyone()

        # 🔍 prepare forward kwargs (copied from LlamaModel)
        all_attn_kwargs = []

        for hidden_states in all_hidden_states:
            hidden_states.to(accelerator.device)

            output_attentions = False
            use_cache = False

            past_key_values = DynamicCache.from_legacy_cache(None)
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + hidden_states.shape[1],
                device=accelerator.device,
            )
            position_ids = cache_position.unsqueeze(0)

            causal_mask = accelerator.unwrap_model(model).model._update_causal_mask(
                batch.get("attention_mask"),
                hidden_states.to(accelerator.device),
                cache_position,
                past_key_values,
                output_attentions,
            )

            input_kwargs = {
                "attention_mask": causal_mask,
                "position_ids": position_ids,
                "past_key_value": past_key_values,
                "output_attentions": output_attentions,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
            all_attn_kwargs.append(input_kwargs)

            hidden_states.cpu()

        # 🔍 remaining layers
        for i in range(num_hidden_layers - 1):
            layer = accelerator.unwrap_model(model).model.layers[i]

            ## get hidden states
            all_gathered_hidden_states = []
            accelerator.print(f"Forward for layer {i} outputs!")

            for j, hidden_states in tqdm(enumerate(all_hidden_states)):
                hidden_states = hidden_states.to(accelerator.device)
                hidden_states = layer(hidden_states, **all_attn_kwargs[j])[0]
                gathered_hidden_states = (
                    accelerator.gather(hidden_states).cpu().float()
                )  # 🔍 all gather across devices
                gathered_hidden_states = gathered_hidden_states.reshape(-1, hidden_size)
                if not accelerator.is_main_process:
                    gathered_hidden_states = (
                        None  # only store the hidden states on the main process
                    )

                all_hidden_states[j] = hidden_states.cpu()
                all_gathered_hidden_states.append(gathered_hidden_states)

            accelerator.print(f"Forward for layer {i} outputs done!")

            ## perform clustering
            if accelerator.is_main_process:
                accelerator.print(f"Clustering for layer {i} outputs!")
                all_gathered_hidden_states = torch.cat(
                    all_gathered_hidden_states, dim=0
                )
                all_gathered_hidden_states = all_gathered_hidden_states.numpy()

                num_features = all_gathered_hidden_states.shape[0]
                balanced_num_features = num_features // clustering_args.num_experts
                min_cluster_size = max(
                    0, int(balanced_num_features * (1 - balance_jitter_factor))
                )
                max_cluster_size = min(
                    num_features,
                    int(balanced_num_features * (1 + balance_jitter_factor)),
                )

                print("ideally balanced cluster size:", balanced_num_features)
                print("min cluster size:", min_cluster_size)
                print("max cluster size:", max_cluster_size)

                if clustering_args.distance_metric == "l2":
                    kmeans = KMeansConstrained(
                        n_clusters=clustering_args.num_experts,
                        size_min=min_cluster_size,
                        size_max=max_cluster_size,
                        tol=1e-1,
                        max_iter=clustering_args.max_iter,
                        random_state=clustering_args.random_state,
                        n_jobs=clustering_args.n_jobs,
                        verbose=True,
                    ).fit(all_gathered_hidden_states, None)
                elif clustering_args.distance_metric == "cos":
                    kmeans = KMeansConstrainedCos(
                        n_clusters=clustering_args.num_experts,
                        size_min=min_cluster_size,
                        size_max=max_cluster_size,
                        tol=1e-1,
                        max_iter=clustering_args.max_iter,
                        random_state=clustering_args.random_state,
                        n_jobs=clustering_args.n_jobs,
                        verbose=True,
                    ).fit(all_gathered_hidden_states, None)
                gate_weights = torch.from_numpy(kmeans.cluster_centers_)

                all_gate_weights[i + 1] = gate_weights  # add to all weights
                accelerator.print(f"{gate_weights.shape}")

                accelerator.print(f"Clustering for layer {i} outputs done!")
            accelerator.wait_for_everyone()

    # 🔍 save clustering results (gate weights)
    if accelerator.is_main_process:
        create_dir(clustering_args.save_path)
        torch.save(
            all_gate_weights, os.path.join(clustering_args.save_path, "gate_weights.pt")
        )
    accelerator.wait_for_everyone()
    accelerator.print("All done!")


if __name__ == "__main__":
    main()
