import __main__
from smoe.utils.conversation import Llama3ConversationTemplate
import pathlib
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, Trainer
import random
from smoe.utils.io import load_jsonlines
import transformers
from typing import Any, Dict, Mapping, Optional
from loguru import logger
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

def preprocess(
    instances,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # Apply prompt templates
    # logger.info(f"Instances: {instances}")  
    conversations = []
    for i, ins in enumerate(instances):
        prompt = Llama3ConversationTemplate.parse(ins["conversations"])
        conversations.append(prompt)

    res = tokenizer(
        conversations,
        return_tensors="pt",
        # padding="max_length",
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    input_ids = res["input_ids"]
    # print("length of input_ids: ", len(input_ids[0]))

    return input_ids

class CachedJsonlDataset(Dataset):
    def __init__(
        self,
        datapath: str,
        tokenizer: PreTrainedTokenizer,
    ) -> None:

        super().__init__()
        self.datapath = datapath
        self.data = load_jsonlines(datapath)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        ins = self.data[index]
        processed = preprocess([ins], self.tokenizer)

        return processed

if __name__ == "__main__":
    dataset_dir_or_path="/mnt/petrelfs/quxiaoye/models/sft-v2/data.jsonl"

    model_name_or_path = "/mnt/petrelfs/quxiaoye/LLaMA-MoE-v2/outputs/v2_mixtral/moe-res/3773156/checkpoint-2000"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        model_max_length=10000,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    datapath = pathlib.Path(dataset_dir_or_path)
    if not datapath.exists():
        raise ValueError(f"Dataset path {datapath} not found")
    elif datapath.is_file():
        logger.info(f"CachedJsonlDataset: {datapath}")
        # train_dataset = CachedJsonlDataset(
        #     dataset_dir_or_path,
        #     tokenizer
        # )

        train_dataset = CachedJsonlDataset(dataset_dir_or_path, tokenizer)
        # first_item = train_dataset[0]
        # logger.info(f"first_item: {first_item}")
        length_distribution = {}
        for i in tqdm(range(len(train_dataset)), desc="Processing items"):
            input_ids = train_dataset[i]
            length = len(input_ids[0])  # 获取 input_ids 的长度
            # print("length: ", length)
            if length not in length_distribution:
                length_distribution[length] = 0
            length_distribution[length] += 1
        
        # 按照键值升序排列
        sorted_length_distribution = dict(sorted(length_distribution.items()))

        # 保存到 JSON 文件
        with open('length_distribution.json', 'w') as f:
            json.dump(sorted_length_distribution, f)

        print("Length distribution saved to 'length_distribution.json'.")

        # 可视化长度分布
        lengths = list(length_distribution.keys())
        counts = list(length_distribution.values())

        plt.figure(figsize=(10, 6))
        plt.bar(lengths, counts, color='skyblue')
        plt.xlabel('Input IDs Length')
        plt.ylabel('Frequency')
        plt.title('Input IDs Length Distribution')
        plt.xticks(lengths)
        plt.grid(axis='y')
        # plt.show()    
        plt.savefig('input_ids_length_distribution.png', bbox_inches='tight')
        logger.info("Length distribution plot saved as 'input_ids_length_distribution.png'")    

    else:
        raise ValueError(f"Unknown dataset path type: {datapath}")
    logger.info("train dataset ready")






