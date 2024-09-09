import os
import math
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import transformers
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, Trainer
from transformers.trainer_pt_utils import LabelSmoother

import smoe.models.mixtral.modeling_mixtral as ModelingMixtralResidual
from smoe.models.mixtral import MixtralConfig, MixtralForCausalLM
from smoe.utils.conversation import Llama3ConversationTemplate
from smoe.utils.io import load_json, load_jsonlines, get_pathname_from_name_or_path

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    model_type: str = field(
        default="auto", metadata={"help": "Model type: `moe` or `mixtral` or `auto`"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "Torch dtype: `float32` or `bfloat16`"},
    )
    additional_config: str = field(
        default=None,
        metadata={"help": "Additional config file (in json) to load"},
    )
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={
            "help": "attention implementation, choice from [eager, flash_attention_2, sdpa] (default: `flash_attention_2`)"
        },
    )

    def __post_init__(self):
        if hasattr(torch, self.torch_dtype):
            self.torch_dtype = getattr(torch, self.torch_dtype)
        if self.additional_config is not None:
            if not pathlib.Path(self.additional_config).exists():
                raise ValueError(
                    f"Additional config file {self.additional_config} not found"
                )
            self.additional_config = load_json(self.additional_config)


@dataclass
class DataArguments:
    eval_data_dir: str = field(
        default=None, metadata={"help": "Path to the evaluation data folder."}
    )
    dataset_dir_or_path: str = field(
        default="data/merged",
        metadata={"help": "Path to dataset directory or a single jsonl file"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    freeze_gate: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the gate during training."},
    )
    save_final_ckpt: bool = field(
        default=True,
        metadata={"help": "Whether to save final checkpoint."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm."},
    )
    save_only_model: bool = field(
        default=False,
        metadata={"help": "Whether to save optimizer."},
    )


def trainer_save_model_safe(trainer):
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def simple_fault_tolerance_data_collator(features: list) -> dict[str, Any]:
    batch = {}
    first = features[0]
    assert all(key in first for key in ["input_ids", "labels", "attention_mask"])
    max_len = max([len(feature["input_ids"]) for feature in features])
    for feature in features:
        # Simple for llama3, we directly use '<|eot_id|>' (128009) for pad token. You should change for other models.
        feature["input_ids"] = torch.cat(
            [
                feature["input_ids"],
                torch.tensor(
                    [128009] * (max_len - len(feature["input_ids"])), dtype=torch.long
                ),
            ]
        )
        feature["labels"] = torch.cat(
            [
                feature["labels"],
                torch.tensor(
                    [-100] * (max_len - len(feature["labels"])), dtype=torch.long
                ),
            ]
        )
        feature["attention_mask"] = torch.cat(
            [
                feature["attention_mask"],
                torch.tensor(
                    [0] * (max_len - len(feature["attention_mask"])), dtype=torch.long
                ),
            ]
        )

    for k, v in first.items():
        batch[k] = torch.stack([f[k] for f in features])

    return batch


def fault_tolerance_data_collator(features: list) -> dict[str, Any]:
    if not isinstance(features[0], Mapping):
        try:
            features = [vars(f) for f in features]
        except TypeError:
            print(len(features), type(features[0]), features[0])
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


class TokenizedSupervisedDataset(Dataset):
    """Tokenized dataset for supervised fine-tuning.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        (HF) tokenizer. `None` for empty dataset.
    input_ids : list[torch.Tensor], default: []
        List of input token ID sequences.
    labels : list[torch.Tensor], default: []
        List of label sequences.
    attention_mask : list[torch.Tensor], default: []
        List of attention mask sequences.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        input_ids: list[torch.Tensor] = None,
        labels: list[torch.Tensor] = None,
        attention_mask: list[torch.Tensor] = None,
    ):

        Dataset.__init__(self)

        if input_ids is None:
            input_ids = []
        if labels is None:
            labels = []
        if attention_mask is None:
            attention_mask = []

        self.tokenizer = tokenizer

        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask


    @staticmethod
    def load_from_raw_dset(
        tokenizer: PreTrainedTokenizer,
        data_path: str,
    ) -> "TokenizedSupervisedDataset":
        """Load a dataset from a file and tokenize it.

        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer
            (HF) tokenizer.
        data_path : str
            Dataset ID or path.

        Returns
        -------
        TokenizedSupervisedDataset
        """
        logger.info("Loading from raw dataset ...")
        dset = load_jsonlines(data_path)
        conversations=[conv['conversations'] for conv in dset]

        prompt, source_part = Llama3ConversationTemplate.parse_list(conversations, skip_system=True)
        
        logger.info("Generating inputs ...")
        # Tokenize conversations
        res_conv = tokenizer(
            prompt,
            # return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        
        input_ids = [torch.tensor(inputs, dtype=torch.int64) for inputs in res_conv["input_ids"]]
        import copy

        targets = copy.deepcopy(input_ids)

        logger.info("Generating labels ...")
        res_source = tokenizer(
            source_part,
            # return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
    
        for i in range(len(targets)):
            source_length = len(res_source["input_ids"][i])
            targets[i][:source_length] = -100  

        logger.info("Generating masks ...")
        attention_masks = [input_id_seq.ne(-100) for input_id_seq in input_ids]

        return TokenizedSupervisedDataset(
            tokenizer=tokenizer,
            input_ids=input_ids,
            labels=targets,
            attention_mask=attention_masks
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(
        self,
        i: int,
    ) -> dict[str, torch.Tensor]:
        """Get a data point.

        Parameters
        ----------
        i : int
            `dataset[i]`

        Returns
        -------
        dict[str, torch.Tensor]
            `{"input_ids": input_ids[i], "labels": labels[i], "attention_mask": attention_mask[i]}`
        """
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attention_mask[i],
        }

    def concat(
        self,
        datasets: list["TokenizedSupervisedDataset"],
    ) -> None:
        """Concatenate `TokenizedSupervisedDataset` instances to the current dataset.
        datasets : list[TokenizedSupervisedDataset]
            List of tokenized datasets to concatenate.
            Each dataset should have the following fields at least: `"input_ids"`, `"labels"`, and `"attention_mask"`.
        """
        self.input_ids += sum([ds.input_ids for ds in datasets], [])
        self.labels += sum([ds.labels for ds in datasets], [])
        self.attention_mask += sum([ds.attention_mask for ds in datasets], [])

    def shuffle(self, seed: int = 42) -> None:
        """Shuffle the dataset."""
        random.seed(seed)
        rand_idxs = list(range(len(self)))
        random.shuffle(rand_idxs)
        logger.debug(f"rand_indices[:10] = {rand_idxs[:10]}")
        logger.debug(f"rand_indices[-10:] = {rand_idxs[-10:]}")
        self.input_ids = [self.input_ids[i] for i in rand_idxs]
        self.labels = [self.labels[i] for i in rand_idxs]
        self.attention_mask = [self.attention_mask[i] for i in rand_idxs]

    def pad(self) -> None:
        """Pad the dataset to the same length of the longest data point."""
        max_len = max([len(input_id) for input_id in self.input_ids])
        for i in range(len(self.input_ids)):
            pad_len = max_len - len(self.input_ids[i])
            self.input_ids[i] = torch.cat(
                [
                    self.input_ids[i],
                    torch.tensor([self.tokenizer.pad_token_id] * pad_len),
                ]
            )
            self.labels[i] = torch.cat(
                [self.labels[i], torch.tensor([-100] * pad_len)]
            )
            self.attention_mask[i] = torch.cat(
                [self.attention_mask[i], torch.tensor([0] * pad_len)]
            )


class PackedDataset(Dataset):
    """Packed dataset containing computation sequences.

    Parameters
    ----------
    dataset : Dataset | HFDataset
        Original tokenized dataset, which should have the following fields at least: `"input_ids"`, `"labels"`, and `"attention_mask"`.
    tokenizer : transformers.PreTrainedTokenizer
        (HF) tokenizer.
    pack_len : int
        Maximum length of packed compuation sequence in token.
    shuffle_seed : int, default: 42
        Seed for shuffling the dataset before packing. `None` / Negative values mean no shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        pack_len: int,
        seed: int = 1227,
    ):
        Dataset.__init__(self)

        self.pack_len = pack_len
        self.tokenizer = tokenizer

        self.lens = []
        self.dps = []

        dp_idxs = list(range(len(dataset)))
        if seed is not None and seed >= 0:
            random.seed(seed)
            random.shuffle(dp_idxs)
            logger.debug(
                f"Shuffled dataset with seed = {seed}, getting {dp_idxs[:5]}..."
            )

        for i_dp in dp_idxs:
            raw_dp = dataset[i_dp]
            input_len = torch.sum(raw_dp["attention_mask"]).item()

            raw_dp["input_ids"] = PackedDataset.extract_ids(
                raw_dp["input_ids"], input_len, tokenizer.padding_side
            )

            if "labels" not in raw_dp:  # Create labels if not existed
                labels = raw_dp["input_ids"].clone()
                # Mask pad_token
                labels[labels == tokenizer.pad_token_id] = -100
                raw_dp["labels"] = labels.tolist()
            else:  # Extract labels
                raw_dp["labels"] = PackedDataset.extract_ids(
                    raw_dp["labels"], input_len, tokenizer.padding_side
                )

            self.dps.append(raw_dp)
            self.lens.append(input_len)

        max_input_len = max(self.lens)
        assert (
            self.pack_len >= max_input_len
        ), f"pack_len must be >= max(input lens), found pack_len={self.pack_len}, max_input_len={max_input_len}"
        self.groups = PackedDataset.pack_dps_by_len(self.lens, self.pack_len)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self.groups)))]
        group = self.groups[idx]
        group_dps = [self.dps[index] for index in group]
        return PackedDataset.pack_dps_FA(group_dps)

    @staticmethod
    def extract_ids(
        ids: list[int],
        input_len: int,
        padding_side: str,
    ) -> dict[str, Any]:
        """Extract `input_ids` and `labels` from a padded data point

        Parameters
        ----------
        ids : list[int]
            (Padded) list of token IDs.
        input_len : int
            Length of input.
        padding_side : str
            Padding side of the tokenizer. Must be 'left' or 'right'.

        Returns
        -------
        dict[str, Any]
            Extracted token IDs.
        """
        assert padding_side in [
            "left",
            "right",
        ], "padding_side must be 'left' or 'right'"
        return ids[:input_len] if padding_side == "right" else ids[-input_len:]

    @staticmethod
    def pack_dps_by_len(lens: list[int], pack_len: int) -> list[list[int]]:
        """Pack data points into groups (each group is a new data point), will be used by PackedDataset, to reduce number of data points in training.
        Given lens of data points, we pack them into groups such that the sum of lens
        in each group is less than `pack_len`. Each group will be considered as a data point (packed data point)
        This is known as: https://en.wikipedia.org/wiki/Bin_packing_problem
        There are many algorithms to implement this, but here we use the simple algorithm.
        We will pack/merge a consecutive list of data points until reaching the `pack_len`

        Parameters
        ----------
        lens : list[int]
            Lengths of data points.
        pack_len : int
            Maximum length of packed compuation sequence in token.

        Returns
        -------
        list[list[int]]
            Length groups of packed data points.
        """
        groups = []
        current_packed_len = 0
        current_group = []
        for i in range(len(lens)):
            cur_len = lens[i]
            if cur_len + current_packed_len <= pack_len:
                current_packed_len += lens[i]
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
                current_packed_len = cur_len
        if len(current_group) > 0:
            groups.append(current_group)
        return groups

    @staticmethod
    def pack_dps_FA(
        dps: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        """Pack data points (for Flash Attention)

        Parameters
        ----------
        dps : list[dict[str, list[int]]]
            Data points, each of which should have the following fields at least: `"input_ids"`, `"labels"`, `"attention_mask"`.
        tokenizer : transformers.PreTrainedTokenizer
            (HF) tokenizer.
        pack_len : int
            Maximum length of packed compuation sequence in token.

        Returns
        -------
        dict[str, torch.Tensor]
            Packed data point tensors.
        """
        input_ids = []
        lens = []
        label_ids = []
        attention_mask = []

        for index, item in enumerate(dps):
            input_ids += item["input_ids"]

            labels = list(item["labels"])
            # The first token should not be used to compute loss
            labels[0] = -100
            label_ids += labels
            lens.append(len(item["input_ids"]))
            attention_mask += [index + 1 for _ in range(len(item["input_ids"]))]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(label_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
    

    def stat(self) -> None:
        """Print out the statistics of the packed dataset.
        Original -> Packed:
        1. Number of data/computation sequences;
        2. Average effective length of compution sequences.
        """
        print(
            f"Number of sequences: {len(self.dps)} data/computation sequences -> {len(self.groups)} computation sequences"
        )
        original_avg_len = sum(self.lens) / len(self.lens)

        packed_lens = []
        for group in self.groups:
            lens = [self.lens[index] for index in group]
            packed_lens.append(sum(lens))

        avg_packed_len = sum(packed_lens) / len(packed_lens)
        print(
            f"Average effective length of compution sequences: {original_avg_len} -> {avg_packed_len}"
        )


def make_supervised_dset(
    tokenizer: PreTrainedTokenizer,
    data_path: str,
    shuffle_seed: int = 42,
    pack_len: int = None,
) -> PackedDataset:
    """Make dataset for supervised fine-tuning.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        (HF) tokenizer.
    data_path : str | list[str]
        Dataset ID or path.
    shuffle_seed : int, default: 42
        Seed for shuffling the dataset before packing. None or negative means no shuffling.
    pack_len : int, default: None
        Maximum length of packed computation sequence in token. None / Non-positive means no packing.
    Returns
    -------
    PackedDataset
        Dataset ready for input to `Trainer`, containing the following fields at least: `"input_ids"`, `"labels"`, and `"attention_mask"`.
    """

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    parent_path = os.path.dirname(data_path)
    ds_pathname =  get_pathname_from_name_or_path(data_path)
    tokenizer_pathname = get_pathname_from_name_or_path(tokenizer.name_or_path)
    # data_cache_path = os.path.join(
    #     parent_path,
    #     f"{ds_pathname}-{tokenizer_pathname}-tokenized.pt"
    # )
    data_cache_path = os.path.join(
        parent_path,
        f"{ds_pathname}-tokenized.pt"
    )
    if not os.path.exists(data_cache_path):
        from torch.distributed import barrier
        # If this is not rank 0, stay here, wait for rank 0 to process the data
        if local_rank != 0:
            print(
                f"[Process {local_rank}] Waiting for main process to prepare the training data"
            )
            barrier()  # When TORCH_NCCL_BLOCKING_WAIT is set, the process will block and wait for this timeout.
            logger.info(
                f"[Process {local_rank}] Loading data from {data_cache_path}"
            )
            train_dataset = torch.load(data_cache_path)
        else:  # Rank 0 processes the data and saves to `data_cache_path`
            # The way we read dataset is:
            # Rank 0 will process the dataset and save the result to data_cache_path, other ranks will read from the data_cache_path
            train_dataset = (
                TokenizedSupervisedDataset.load_from_raw_dset(
                    tokenizer=tokenizer,
                    data_path=data_path
                )
            )
            torch.save(train_dataset, data_cache_path)

            logger.info(
                f"process: {local_rank} finishes processing data and saves to {data_cache_path}"
            )
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if world_size > 1:
                barrier()
    else:  # Cache existing
        logger.info(f"[Process {local_rank}] Loading data from {data_cache_path}")
        train_dataset = torch.load(data_cache_path)

    # Shuffle the dataset if necessary
    logger.debug(f"Shuffle seed ...")
    if shuffle_seed is not None and shuffle_seed >= 0:
        train_dataset.shuffle(seed=shuffle_seed)

    # Pack the dataset if specified
    logger.debug(f"Packing dataset with pack_len = {pack_len} ...")
    train_dataset = PackedDataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        pack_len=pack_len,
    )

    # For consistency checking
    logger.info(f"len(train_dataset): {len(train_dataset)}")
    logger.info(f"train_dataset[-1]: {train_dataset[-1]}")

    return train_dataset


def get_max_seqlen_in_batch(attention_mask):
    max_num = torch.max(attention_mask)
    # attention_mask: B x N
    counts = []
    for i in range(1, max_num + 1):
        counts.append(
            torch.sum(attention_mask == i, axis=-1)
        )  # shape: B, count length of data point maksed with i
    result = torch.stack(counts, axis=1)
    result = result.flatten()
    return result[result.nonzero()].squeeze(-1).to(dtype=torch.int32)


def get_unpad_data(attention_mask):
    seqlens_in_batch = get_max_seqlen_in_batch(
        attention_mask
    )  # attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def get_tokenizer(
    model_name_or_path,
    cache_dir: str = None,
    model_max_length: int = 2048,
    padding_side: str = "right",
    use_fast: bool = False,
    trust_remote_code: bool = False,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"tokenizer ready, pad_token: {tokenizer.pad_token}")
    return tokenizer


def get_model(
    model_type: str,
    model_name_or_path: str,
    torch_dtype: str = "auto",
    model_max_length: int = 2048,
    attn_impl: str = "flash_attention_2",
    cache_dir: str = None,
    trust_remote_code: bool = False,
    additional_config: dict = None,
):
    logger.info(f"Model type: {model_type}")
    if model_type == "auto":
        ConfigClass = transformers.AutoConfig
        ModelClass = transformers.AutoModelForCausalLM
    elif model_type == "v2_mixtral":
        ConfigClass = MixtralConfig
        ModelClass = MixtralForCausalLM
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Set RoPE scaling factor
    config = ConfigClass.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    config._attn_implementation = attn_impl
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False
    if additional_config is not None:
        config.update(additional_config)
    logger.info("Config ready")

    # Load model and tokenizer
    model = ModelClass.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    ModelingMixtralResidual._get_unpad_data = get_unpad_data

    # model.to('cuda')
    # model = ModelClass(config)
    logger.info("model ready")

    return model


def get_model_and_tokenizer(
    model_type: str,
    model_name_or_path: str,
    tokenizer_path: str = None,
    torch_dtype: str = "auto",
    model_max_length: int = 2048,
    attn_impl: str = "flash_attention_2",
    cache_dir: str = None,
    trust_remote_code: bool = False,
    padding_side: str = "right",
    additional_config: dict = None,
    use_fast: bool = False,
) -> tuple:
    if tokenizer_path is None:
        tokenizer_path = model_name_or_path
    tokenizer = get_tokenizer(
        tokenizer_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    model = get_model(
        model_type,
        model_name_or_path,
        torch_dtype=torch_dtype,
        model_max_length=model_max_length,
        attn_impl=attn_impl,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        additional_config=additional_config,
    )

    return model, tokenizer


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    # donot report to tensorboard, may speed up training
    # training_args: TrainingArguments
    # if "tensorboard" not in training_args.report_to:
    #     training_args.report_to.append("tensorboard")
    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

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
    tot_params = 0
    for name, param in model.named_parameters():
        # logger.info(name, param.shape, param.numel())
        tot_params += param.numel()
    logger.info(f"Total model params: {tot_params}")

    # train_dataset = None
    datapath = pathlib.Path(data_args.dataset_dir_or_path)
    if not datapath.exists():
        raise ValueError(f"Dataset path {datapath} not found")
    elif datapath.is_file():
        # logger.info(f"CachedJsonlDataset: {datapath}")
        supervied_dset = make_supervised_dset(
            tokenizer=tokenizer,
            data_path=data_args.dataset_dir_or_path,
            shuffle_seed=training_args.seed,
            pack_len=training_args.model_max_length,
        )
    else:
        raise ValueError(f"Unknown dataset path type: {datapath}")
    logger.info("train dataset ready")

    # print("starting memory tracking...")
    # torch.cuda.memory._record_memory_history(enabled=True, trace_alloc_record_context=True, _enable_expensive_cpp=True)
    # print("starting memory tracking...ok")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=supervied_dset,
        data_collator=simple_fault_tolerance_data_collator,
        # data_collator=fault_tolerance_data_collator,
        # num_processes=1 # for flash_attention_2
    )
    logger.info("trainer ready")

    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logger.info("resume training from ckpt")
            trainer.train(resume_from_checkpoint=True)
        else:
            logger.info("start training")
            trainer.train()

    # Save model
    if training_args.save_final_ckpt:
        logger.info("training finished, dumping model")
        model.config.use_cache = True
        trainer.save_state()  # for debug, not save
        if trainer.is_deepspeed_enabled:
            trainer.save_model()
        else:
            trainer_save_model_safe(trainer)

    logger.info("🎉 All done~")


if __name__ == "__main__":
    
    train()

    # from tqdm import tqdm
    # tokenizer = get_tokenizer("resources/llama-3-8b-mixtral-megablocks-56e-top8")
    # train_dataset = CachedJsonlDataset(
    #     "resources/OpenHermes-2.5/openhermes2_5.jsonl",
    #     tokenizer,
    #     seed=1227,
    # )
    # tok_lens = []
    # for ins in tqdm(train_dataset):
    #     tok_lens.append(ins["input_ids"].shape[0])
    # print(np.mean(tok_lens), np.std(tok_lens))
    # import matplotlib.pyplot as plt
    # plt.hist(tok_lens, bins=100)
    # plt.savefig("tok_lens.png")
    # plt.close()
