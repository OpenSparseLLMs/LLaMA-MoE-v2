import torch
import torch.nn.utils.rnn as rnn_utils
from torch import nn

from smoe.models.mixtral_for_test import MixtralConfig
from smoe.models.mixtral_for_test.modeling_mixtral import (
    MixtralAttention,
    MixtralAttentionMoE,
)
from smoe.utils.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


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
hidden_size = 256
intermediate_size = 768
num_attention_heads = 32
num_key_value_heads = 8

top_k_attn = 2  # 🔍
scale_factor_attn = 1.0  # 🔍

config = MixtralConfig(
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_attention_heads=num_attention_heads,
    num_key_value_heads=num_key_value_heads,
)

"""model"""
embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, 0)
attn = MixtralAttention(config, layer_idx=0)
attn_moe = MixtralAttentionMoE.from_vanilla_attention(
    attn, top_k_attn=top_k_attn, scale_factor_attn=scale_factor_attn
)

"""input"""

collator = tensor_stack_padding_collater(padding_id=0, padding_position="right")
tensors = [torch.randperm(i) + 1 for i in [2, 5, 6, 7, 5]]
input_ids, attention_mask = collator(tensors)

"""attn input"""
batch_size, seq_length = input_ids.shape

past_key_value = None
output_attentions = False
use_cache = False

hidden_states = embed_tokens(input_ids)

past_key_values_length = 0
position_ids = torch.arange(
    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long
)
position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

"""forward attn"""
# 4d mask is passed through the layers
attn_attention_mask = _prepare_4d_causal_attention_mask(
    attention_mask,
    (batch_size, seq_length),
    hidden_states,
    past_key_values_length,
    sliding_window=config.sliding_window,
)

attn_hidden_states, attn_self_attn_weights, attn_present_key_value = attn(
    hidden_states=hidden_states,
    attention_mask=attn_attention_mask,
    position_ids=position_ids,
    past_key_value=past_key_value,
    output_attentions=output_attentions,
    use_cache=use_cache,
)

"""forward attn moe"""
# 2d mask is passed through the layers
attn_moe_attention_mask = (
    attention_mask if (attention_mask is not None and 0 in attention_mask) else None
)

(
    attn_moe_hidden_states,
    attn_moe_self_attn_weights,
    attn_moe_present_key_value,
) = attn_moe(
    hidden_states=hidden_states,
    attention_mask=attn_moe_attention_mask,
    position_ids=position_ids,
    past_key_value=past_key_value,
    output_attentions=output_attentions,
    use_cache=use_cache,
)

print("Done!")
