"""
Adapter to expose MegaBlocks package, if available.

Ref: https://github.com/stanford-futuredata/Megatron-LM/blob/3a9e3d8de308e6f6398b59d16a8bd7177374f121/megatron/model/megablocks_utils.py
"""

try:
    import megablocks
except ImportError:
    megablocks = None


def megablocks_is_available():
    return megablocks is not None


def assert_megablocks_is_available():
    assert (
        megablocks_is_available()
    ), "MegaBlocks not available. Please run `pip install megablocks`."


def param_is_expert_model_parallel(param):
    if megablocks_is_available():
        return megablocks.layers.mpu.param_is_expert_model_parallel(param)
    return False


def copy_expert_model_parallel_attributes(destination_tensor, source_tensor):
    if not megablocks_is_available():
        return
    megablocks.layers.mpu.copy_expert_model_parallel_attributes(
        destination_tensor, source_tensor
    )


MoE = megablocks.layers.moe if megablocks_is_available() else None
dMoE = megablocks.layers.dmoe if megablocks_is_available() else None
arguments = megablocks.layers.arguments if megablocks_is_available() else None
