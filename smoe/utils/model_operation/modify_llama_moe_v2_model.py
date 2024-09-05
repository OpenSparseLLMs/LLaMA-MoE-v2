import types

import torch

from smoe.models.mixtral import MixtralModel
from smoe.utils.model_operation.change_llama_moe_v2_forward import (
    forward_llama_moe_v2_decoder_with_hidden_states_distribution_recording,
    forward_llama_moe_v2_decoder_with_hidden_states_distribution_recording_for_alignment,
    forward_llama_moe_v2_model_with_early_stopping,
)


def llama_moe_v2_with_hidden_distribution_recording(model):
    # fmt: off
    assert isinstance(model, MixtralModel)
    hidden_size = model.config.hidden_size

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        layer.layer_idx = layer_idx
        layer.attn_distribution = {
            "number": torch.zeros((1,)),
            "mean": torch.zeros((hidden_size,)),
            "variance": torch.zeros((hidden_size,)),
        }
        layer.mlp_distribution = {
            "number": torch.zeros((1,)),
            "mean": torch.zeros((hidden_size,)),
            "variance": torch.zeros((hidden_size,)),
        }
        layer.forward = types.MethodType(forward_llama_moe_v2_decoder_with_hidden_states_distribution_recording, layer)  # change forward function for LlamaDecoderLayer

    return model
    # fmt: on


def llama_moe_v2_with_hidden_distribution_recording_for_alignment(model):
    # fmt: off
    assert isinstance(model, MixtralModel)

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        layer.layer_idx = layer_idx
        layer.forward = types.MethodType(forward_llama_moe_v2_decoder_with_hidden_states_distribution_recording_for_alignment, layer)  # change forward function for MixtralDecoderLayer

    model.forward = types.MethodType(forward_llama_moe_v2_model_with_early_stopping, model)  # change forward function for MixtralModel

    return model
    # fmt: on
