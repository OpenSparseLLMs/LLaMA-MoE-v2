from typing import List, Optional, Tuple, Union

import torch
from transformers import DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from smoe.models.mixtral.modeling_mixtral import (
    MoECache,
    MoeModelOutputWithPast,
    logger,
)
from smoe.utils.cache_utils import Cache


def forward_llama_moe_v2_decoder_with_hidden_states_distribution_recording(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    output_router_logits: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    # fmt: off
    """transformers 4.42.4"""
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    ##########################################################
    # exclude padding tokens
    if attention_mask.ndim == 2:
        padding_mask = attention_mask.clone().bool()
    elif attention_mask.ndim == 4:
        padding_mask = ~attention_mask[:, 0, :, :].all(dim=-2)
    else:
        raise ValueError("padding_mask must be either 2 or 4 dimensional")
    ##########################################################

    # 🔍 Self Attention
    if self.use_attn_moe:
        (
            hidden_states,
            self_attn_weights,
            present_key_value,
            attn_router_logits,
        ) = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
    else:
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_router_logits = None

    ##########################################################
    # record distribution for attention
    if self.align_module is not None and self.align_module == "attn":
        non_padding_hidden_states = hidden_states[padding_mask]

        this_num = torch.tensor((non_padding_hidden_states.shape[0],), device=hidden_states.device)
        this_mean = non_padding_hidden_states.mean(dim=0)
        this_var = non_padding_hidden_states.var(dim=0)

        old_num = self.distribution["number"].to(hidden_states.device)
        old_mean = self.distribution["mean"].to(hidden_states.device)
        old_var = self.distribution["variance"].to(hidden_states.device)

        self.distribution["number"] = old_num + this_num
        self.distribution["mean"] = (old_num * old_mean + this_num * this_mean) / (old_num + this_num)
        self.distribution["variance"] = (
            old_num * old_var
            + this_num * this_var
            + old_num * this_num / (old_num + this_num) * (old_mean - this_mean) ** 2
        ) / (old_num + this_num)

        # print(f'({hidden_states.device}) (Layer {self.layer_idx}) Attn Number: {old_num} -> {self.distribution["number"]}')
        # print(f'({hidden_states.device}) (Layer {self.layer_idx}) Attn Mean: {old_mean} -> {self.distribution["mean"]}')
        # print(f'({hidden_states.device}) (Layer {self.layer_idx}) Attn Variance: {old_var} -> {self.distribution["variance"]}')

        # skip the MLP calculation to save time
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (None, attn_router_logits)

        return outputs
    ##########################################################

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states, router_logits = self.block_sparse_moe(hidden_states)

    ##########################################################
    # record distribution for MLP
    if self.align_module is not None and self.align_module == "mlp":
        non_padding_hidden_states = hidden_states[padding_mask]

        this_num = torch.tensor((non_padding_hidden_states.shape[0],), device=hidden_states.device)
        this_mean = non_padding_hidden_states.mean(dim=0)
        this_var = non_padding_hidden_states.var(dim=0)

        old_num = self.distribution["number"].to(hidden_states.device)
        old_mean = self.distribution["mean"].to(hidden_states.device)
        old_var = self.distribution["variance"].to(hidden_states.device)

        self.distribution["number"] = old_num + this_num
        self.distribution["mean"] = (old_num * old_mean + this_num * this_mean) / (old_num + this_num)
        self.distribution["variance"] = (
            old_num * old_var
            + this_num * this_var
            + old_num * this_num / (old_num + this_num) * (old_mean - this_mean) ** 2
        ) / (old_num + this_num)

        # print(f'({hidden_states.device}) (Layer {self.layer_idx}) Attn Number: {old_num} -> {self.distribution["number"]}')
        # print(f'({hidden_states.device}) (Layer {self.layer_idx}) Attn Mean: {old_mean} -> {self.distribution["mean"]}')
        # print(f'({hidden_states.device}) (Layer {self.layer_idx}) Attn Variance: {old_var} -> {self.distribution["variance"]}')
    ##########################################################

    if self.mlp_residual is not None:
        hidden_states += self.mlp_residual(hidden_states)  # 🔍
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    if output_router_logits:
        outputs += (router_logits, attn_router_logits)  # 🔍

    return outputs
    # fmt: on


def forward_llama_moe_v2_model_with_early_stopping(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, MoeModelOutputWithPast]:
    """transformers 4.42.4"""
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    past_key_values_length = 0

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            if self.config.use_attn_moe:  # 🔍
                past_key_values = MoECache.from_legacy_cache(past_key_values)
            else:  # 🔍
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

        # 🔍 add total seen tokens, this is VERY important for getting correct `past_key_values_length`!
        if self.config.use_attn_moe:
            past_key_values.add_seen_tokens_total(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if attention_mask is not None and self._use_flash_attention_2 and use_cache:
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Mixtral. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if (
        self._use_flash_attention_2 or self.config.use_attn_moe
    ):  # 🔍 added special case for attention MoE
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_router_logits = () if output_router_logits else None
    all_attn_router_logits = () if output_router_logits else None  # 🔍
    next_decoder_cache = None

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs)

                return custom_forward

            layer_outputs: tuple = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                output_router_logits,
                use_cache,
            )

        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        ##################################################################
        if layer_idx == self.early_stopping_layer:
            break
        ##################################################################

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if output_router_logits:
            all_router_logits += (layer_outputs[-2],)
            all_attn_router_logits += (layer_outputs[-1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                all_router_logits,
            ]
            if v is not None
        )
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=all_router_logits,
        attn_router_logits=all_attn_router_logits,  # 🔍
    )
