import argparse
import os
import shutil
import warnings

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from smoe.models.mixtral import MixtralConfig, MixtralForCausalLM, MixtralModel
from smoe.models.mixtral.modeling_mixtral import MISTRAL_ATTENTION_MOE_CLASSES
from smoe.models.mixtral_residual.configuration_mixtral_residual import (
    MixtralResidualConfig,
)
from smoe.models.mixtral_residual.modeling_mixtral_residual import (
    MixtralResidualForCausalLM,
    MixtralResidualModel,
)

# fmt: off

# 🔍
AutoConfig.register("mixtral", MixtralConfig, exist_ok=True)
AutoConfig.register("mixtral_residual", MixtralResidualConfig, exist_ok=True)

AutoModel.register(MixtralConfig, MixtralModel, exist_ok=True)
AutoModel.register(MixtralResidualConfig, MixtralResidualModel, exist_ok=True)

AutoModelForCausalLM.register(MixtralConfig, MixtralForCausalLM, exist_ok=True)
AutoModelForCausalLM.register(MixtralResidualConfig, MixtralResidualForCausalLM, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str)

    parser.add_argument('--top_k_attn', type=int, default=None)
    parser.add_argument('--scale_factor_attn', type=float, default=None)

    args = parser.parse_args()
    print(args, "\n")

    """model"""
    # load
    model = AutoModelForCausalLM.from_pretrained(args.model_path).bfloat16()
    config = model.config

    # change config
    config.use_attn_moe = True
    config.top_k_attn = args.top_k_attn
    config.scale_factor_attn = args.scale_factor_attn

    # init new model
    model_attn_moe = AutoModelForCausalLM.from_config(config).bfloat16()

    # change weight
    model_attn_moe.load_state_dict(model.state_dict(), strict=False)

    for i, layer in enumerate(model_attn_moe.model.layers):
        layer.self_attn = MISTRAL_ATTENTION_MOE_CLASSES[config._attn_implementation].from_vanilla_attention(
            model.model.layers[i].self_attn,
            config.top_k_attn,
            config.scale_factor_attn,
        )

    # save
    model_attn_moe.save_pretrained(args.save_path)

    """tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.save_pretrained(args.save_path)

    """code"""
    current_path = os.path.dirname(__file__)

    if isinstance(model, MixtralForCausalLM):
        shutil.copy(os.path.join(current_path, "../../../models/mixtral/configuration_mixtral.py"), args.save_path)
        shutil.copy(os.path.join(current_path, "../../../models/mixtral/modeling_mixtral.py"), args.save_path)
    elif isinstance(model, MixtralResidualForCausalLM):
        shutil.copy(os.path.join(current_path, "../../../models/mixtral_residual/configuration_mixtral_residual.py"), args.save_path)
        shutil.copy(os.path.join(current_path, "../../../models/mixtral_residual/modeling_mixtral_residual.py"), args.save_path)
    else:
        warnings.warn(f"[WARN] unknown model type {type(model)}")

    print("Done!")

# fmt: on
