import os
import pickle
from typing import List, Optional, Tuple, Union

import torch
from transformers import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)


def forward_llama_mlp_with_backward_hook_bug_fix(self, x):
    # fmt: off
    batch_size, seq_len, hidden_size = x.shape
    x = x.reshape(batch_size * seq_len, hidden_size)  # ---- reshape -----

    gate_proj_output = self.act_fn(self.gate_proj(x))
    up_proj_output = self.up_proj(x)
    gate_up_mm_output = gate_proj_output * up_proj_output
    down_proj_output = self.down_proj(gate_up_mm_output)

    down_proj_output = down_proj_output.reshape(batch_size, seq_len, hidden_size)  # ---- reshape -----
    return down_proj_output
    # fmt: on


def forward_llama_mlp_with_feature_dumping(self, x, padding_mask):
    # fmt: off
    self.now_epoch += 1
    self.hidden_inputs.append(x.detach().half()[padding_mask])  # exclude padding features

    if self.now_epoch % self.save_interval == (self.save_interval - 1):
        save_path = os.path.join(self.save_path_hidden_inputs, str(self.device_id) + "_" + str(self.now_epoch // self.save_interval) + ".pth")
        torch.save(torch.cat(self.hidden_inputs, dim=0).reshape(-1, self.hidden_dim).half().cpu(), save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        self.hidden_inputs = []

    gate_proj_output = self.act_fn(self.gate_proj(x))
    up_proj_output = self.up_proj(x)
    gate_up_mm_output = gate_proj_output * up_proj_output
    down_proj_output = self.down_proj(gate_up_mm_output)

    if "gate_proj" in self.template:
        self.hidden_outputs.append(gate_proj_output[padding_mask].detach().half())
    elif "up_proj" in self.template:
        self.hidden_outputs.append(gate_up_mm_output[padding_mask].detach().half())

    if self.now_epoch % self.save_interval == (self.save_interval - 1):
        save_path = os.path.join(self.save_path_hidden_outputs, str(self.device_id) + "_" + str(self.now_epoch // self.save_interval) + ".pth")
        torch.save(torch.cat(self.hidden_outputs, dim=0).reshape(-1, self.hidden_neurons).half().cpu(), save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        self.hidden_outputs = []

    return down_proj_output
    # fmt: on


def forward_llama_decoder_with_hidden_states_scale_recording(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)

    ###########################################################
    self.mlp_outputs.append(
        torch.abs(hidden_states.detach().clone().float()).sum(2).flatten()
    )
    self.mlp_residuals.append(
        torch.abs(residual.detach().clone().float()).sum(2).flatten()
    )
    ###########################################################

    ###########################################################
    # self.mlp_outputs.append((hidden_states * hidden_states).detach().clone().float().sum(2).flatten())
    # self.mlp_residuals.append((residual * residual).detach().clone().float().sum(2).flatten())
    ###########################################################

    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def forward_llama_decoder_with_hidden_states_distribution_recording(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
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

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    ##########################################################
    # record distribution for attention
    non_padding_hidden_states = hidden_states[padding_mask]

    this_num = torch.tensor((non_padding_hidden_states.shape[0],), device=hidden_states.device)
    this_mean = non_padding_hidden_states.mean(dim=0)
    this_var = non_padding_hidden_states.var(dim=0)

    old_num = self.attn_distribution["number"].to(hidden_states.device)
    old_mean = self.attn_distribution["mean"].to(hidden_states.device)
    old_var = self.attn_distribution["variance"].to(hidden_states.device)

    self.attn_distribution["number"] = old_num + this_num
    self.attn_distribution["mean"] = (old_num * old_mean + this_num * this_mean) / (old_num + this_num)
    self.attn_distribution["variance"] = (
         old_num * old_var
         + this_num * this_var
         + old_num * this_num / (old_num + this_num) * (old_mean - this_mean) ** 2
     ) / (old_num + this_num)

    print(f'({hidden_states.device}) (Layer {self.layer_idx}) Attn Number: {old_num} -> {self.attn_distribution["number"]}')
    print(f'({hidden_states.device}) (Layer {self.layer_idx}) Attn Mean: {old_mean} -> {self.attn_distribution["mean"]}')
    print(f'({hidden_states.device}) (Layer {self.layer_idx}) Attn Variance: {old_var} -> {self.attn_distribution["variance"]}')
    ##########################################################

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)

    ###########################################################
    # record distribution for MLP
    non_padding_hidden_states = hidden_states[padding_mask]

    this_num = torch.tensor((non_padding_hidden_states.shape[0],), device=hidden_states.device)
    this_mean = non_padding_hidden_states.mean(dim=0)
    this_var = non_padding_hidden_states.var(dim=0)

    old_num = self.mlp_distribution["number"].to(hidden_states.device)
    old_mean = self.mlp_distribution["mean"].to(hidden_states.device)
    old_var = self.mlp_distribution["variance"].to(hidden_states.device)

    self.mlp_distribution["number"] = old_num + this_num
    self.mlp_distribution["mean"] = (old_num * old_mean + this_num * this_mean) / (old_num + this_num)
    self.mlp_distribution["variance"] = (
        old_num * old_var
        + this_num * this_var
        + old_num * this_num / (old_num + this_num) * (old_mean - this_mean) ** 2
    ) / (old_num + this_num)

    print(f'({hidden_states.device}) (Layer {self.layer_idx}) MLP Number: {old_num} -> {self.mlp_distribution["number"]}')
    print(f'({hidden_states.device}) (Layer {self.layer_idx}) MLP Mean: {old_mean} -> {self.mlp_distribution["mean"]}')
    print(f'({hidden_states.device}) (Layer {self.layer_idx}) MLP Variance: {old_var} -> {self.mlp_distribution["variance"]}')
    ###########################################################

    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs
    # fmt: on


def forward_llama_decoder_with_padding_mask(
    self,
    hidden_states,
    padding_mask,  # ----- add padding_mask -----
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    ###########################################################
    # ----- add padding_mask -----
    hidden_states = self.mlp(hidden_states, padding_mask)
    ###########################################################
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)

    return outputs


def forward_llama_model_with_padding_mask(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
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

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

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

    # embed positions
    ###########################################################
    padding_mask = attention_mask.bool()  # ----- add padding_mask -----
    ###########################################################
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            ###########################################################
            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                padding_mask,  # ----- add padding_mask -----
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                padding_mask,  # ----- add padding_mask -----
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            ###########################################################

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
