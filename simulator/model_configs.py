"""
Centralized model configuration registry.

All LLM model architectures are defined here so that analysis scripts
and the simulator can reference them by name.

Usage:
    from model_configs import get_model_config, list_models

    model = get_model_config("OPT-6.7B")
    print(list_models())
"""

from simulator import ModelConfig


# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY: dict[str, ModelConfig] = {}


def _register(name: str, config: ModelConfig):
    """Register a model config under *name*."""
    MODEL_REGISTRY[name] = config


# ---------- OPT family (MHA, dense) ----------

_register("OPT-125M", ModelConfig(
    num_layers=12, num_heads=12, num_kv_heads=12,
    d_model=768, d_ffn=3072, head_dim=64,
))

_register("OPT-350M", ModelConfig(
    num_layers=24, num_heads=16, num_kv_heads=16,
    d_model=1024, d_ffn=4096, head_dim=64,
))

_register("OPT-1.3B", ModelConfig(
    num_layers=24, num_heads=32, num_kv_heads=32,
    d_model=2048, d_ffn=8192, head_dim=64,
))

_register("OPT-2.7B", ModelConfig(
    num_layers=32, num_heads=32, num_kv_heads=32,
    d_model=2560, d_ffn=10240, head_dim=80,
))

_register("OPT-6.7B", ModelConfig(
    num_layers=32, num_heads=32, num_kv_heads=32,
    d_model=4096, d_ffn=16384, head_dim=128,
))

_register("OPT-13B", ModelConfig(
    num_layers=40, num_heads=40, num_kv_heads=40,
    d_model=5120, d_ffn=20480, head_dim=128,
))

_register("OPT-30B", ModelConfig(
    num_layers=48, num_heads=56, num_kv_heads=56,
    d_model=7168, d_ffn=28672, head_dim=128,
))

_register("OPT-66B", ModelConfig(
    num_layers=64, num_heads=72, num_kv_heads=72,
    d_model=9216, d_ffn=36864, head_dim=128,
))

# ---------- LLaMA-2 family (MHA, dense) ----------

_register("LLaMA-2-7B", ModelConfig(
    num_layers=32, num_heads=32, num_kv_heads=32,
    d_model=4096, d_ffn=11008, head_dim=128,
))

_register("LLaMA-2-13B", ModelConfig(
    num_layers=40, num_heads=40, num_kv_heads=40,
    d_model=5120, d_ffn=13824, head_dim=128,
))

_register("LLaMA-2-70B", ModelConfig(
    num_layers=80, num_heads=64, num_kv_heads=8,   # GQA
    d_model=8192, d_ffn=28672, head_dim=128,
))

# ---------- LLaMA-3 family (GQA, dense) ----------

_register("LLaMA-3-8B", ModelConfig(
    num_layers=32, num_heads=32, num_kv_heads=8,    # GQA
    d_model=4096, d_ffn=14336, head_dim=128,
))

_register("LLaMA-3-70B", ModelConfig(
    num_layers=80, num_heads=64, num_kv_heads=8,    # GQA
    d_model=8192, d_ffn=28672, head_dim=128,
))

# ---------- Mixtral (GQA + MoE) ----------

_register("Mixtral-8x7B", ModelConfig(
    num_layers=32, num_heads=32, num_kv_heads=8,    # GQA
    d_model=4096, d_ffn=14336, head_dim=128,
    num_experts=8, num_active_experts=2,
))

_register("Mixtral-8x22B", ModelConfig(
    num_layers=56, num_heads=48, num_kv_heads=8,    # GQA
    d_model=6144, d_ffn=16384, head_dim=128,
    num_experts=8, num_active_experts=2,
))

# ---------- Qwen-3 family ----------

_register("Qwen3-8B", ModelConfig(
    num_layers=36, num_heads=32, num_kv_heads=8,    # GQA
    d_model=4096, d_ffn=12288, head_dim=128,
))

_register("Qwen3-14B", ModelConfig(
    num_layers=40, num_heads=40, num_kv_heads=8,    # GQA
    d_model=5120, d_ffn=17408, head_dim=128,
))

_register("Qwen3-30B-A3B", ModelConfig(
    num_layers=48, num_heads=32, num_kv_heads=4,    # GQA + MoE
    d_model=2048, d_ffn=768, head_dim=128,          # d_ffn = moe_intermediate_size
    num_experts=128, num_active_experts=8,
))


# ============================================================================
# Public API
# ============================================================================

def get_model_config(name: str) -> ModelConfig:
    """Look up a model by name.  Raises KeyError if not found."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    return sorted(MODEL_REGISTRY.keys())
