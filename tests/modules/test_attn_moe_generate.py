import time

import torch
import torch.nn.utils.rnn as rnn_utils
from transformers import GenerationConfig

from smoe.models.mixtral import MixtralConfig
from smoe.models.mixtral.modeling_mixtral import MixtralForCausalLM


class tensor_stack_padding_collater:  # 拼接tensor，并padding到最大长度
    """
    examples: list of tensors.
    input: [tensor1, tensor2, ..., tensorN]
    output: padded_tensor
    """

    def __init__(self, padding_id, padding_position="right", return_padding_mask=True):
        assert padding_position in ("left", "right")
        self.padding_id = padding_id
        self.padding_position = padding_position
        self.return_padding_mask = return_padding_mask

    def __call__(self, examples):
        dtype = examples[0].dtype
        if self.padding_position == "right":
            padded_examples = rnn_utils.pad_sequence(
                examples, batch_first=True, padding_value=self.padding_id
            )
        elif (
            self.padding_position == "left"
        ):  # This will take about twice the time compared to right padding
            flipped_examples = [torch.flip(tensor, dims=[0]) for tensor in examples]
            padded_examples_flip = rnn_utils.pad_sequence(
                flipped_examples, batch_first=True, padding_value=self.padding_id
            )
            padded_examples = torch.flip(padded_examples_flip, dims=[1])
        else:
            raise NotImplementedError
        padded_examples = padded_examples.to(dtype)

        if self.return_padding_mask:
            padding_mask = padded_examples != self.padding_id
            return padded_examples, padding_mask
        else:
            return padded_examples


"""basic config"""
seed = 2

torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# MLP MoE
num_experts_per_tok = 2
num_local_experts = 8

# Attention MoE
num_attention_heads = 32
num_key_value_heads = 8
use_attn_moe = True  # 🔍
top_k_attn = 2  # 🔍
scale_factor_attn = 1.0  # 🔍

config = MixtralConfig(
    vocab_size=10,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_experts_per_tok=num_experts_per_tok,
    num_local_experts=num_local_experts,
    num_attention_heads=num_attention_heads,
    num_key_value_heads=num_key_value_heads,
    use_attn_moe=use_attn_moe,
    top_k_attn=top_k_attn,
    scale_factor_attn=scale_factor_attn,
)

"""model"""
model: MixtralForCausalLM = MixtralForCausalLM(config).to(device)
model.eval()

"""input"""
collator = tensor_stack_padding_collater(padding_id=0, padding_position="left")
tensors = [torch.randperm(i) + 1 for i in [2, 5, 6, 4, 3]]
input_ids, attention_mask = collator(tensors)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

"""configs"""
max_length = 1024
repetition_penalty = 3.0

"""generate with cache"""
generation_config = GenerationConfig(
    pad_token_id=0,
    eos_token_id=0,
    max_length=max_length,
    output_attentions=False,  # False
    do_sample=False,
    use_cache=True,  # 🔍
    repetition_penalty=repetition_penalty,
)

begin_time = time.time()
result_cache = model.generate(
    inputs=input_ids,
    attention_mask=attention_mask,
    generation_config=generation_config,
)
end_time = time.time()

time_cache = end_time - begin_time
print(result_cache)

"""generate w/o cache"""
generation_config.use_cache = False  # 🔍

begin_time = time.time()
result_normal = model.generate(
    inputs=input_ids,
    attention_mask=attention_mask,
    generation_config=generation_config,
)
end_time = time.time()

time_normal = end_time - begin_time
print(result_normal)

print("time_cache", time_cache)
print("time_normal", time_normal)

"""check"""
if torch.equal(result_normal, result_cache):
    print("Passed!")
else:
    print("Failed!")

print("Done!")
