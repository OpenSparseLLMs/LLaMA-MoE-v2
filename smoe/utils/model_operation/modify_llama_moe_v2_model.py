import types

from smoe.models.mixtral import MixtralModel
from smoe.utils.model_operation.change_llama_moe_v2_forward import (
    forward_llama_moe_v2_decoder_with_hidden_states_distribution_recording,
    forward_llama_moe_v2_model_with_early_stopping,
)


def llama_moe_v2_with_hidden_distribution_recording(model):
    # fmt: off
    assert isinstance(model, MixtralModel)

    for layer_idx, layer in enumerate(model.layers):  # locate block by the name template
        layer.layer_idx = layer_idx
        layer.forward = types.MethodType(forward_llama_moe_v2_decoder_with_hidden_states_distribution_recording, layer)  # change forward function for MixtralDecoderLayer

    model.forward = types.MethodType(forward_llama_moe_v2_model_with_early_stopping, model)  # change forward function for MixtralModel

    return model
    # fmt: on
