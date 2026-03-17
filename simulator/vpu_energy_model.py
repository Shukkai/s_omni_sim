"""
VPU Energy Model for Non-GEMM Operations

This module defines energy costs for vector processing unit (VPU) operations
used in LLM inference pipelines. The VPU handles non-GEMM operations like:
  - Element-wise arithmetic (add, mul, sub, div)
  - Transcendental functions (exp, log, sqrt, rsqrt)
  - Activation functions (sigmoid, tanh, gelu, silu)
  - Reduction operations (sum, max, mean)
  - Special operations (sin, cos for RoPE)

Energy values are in Joules (J) per operation.
All values should be filled in based on your hardware characterization.

VPU Configuration:
  - Width: 128 lanes (processing 128 FP16 elements per cycle)
  - Frequency: Matches main accelerator (default 500 MHz)
"""

from dataclasses import dataclass


@dataclass
class VPUEnergyConfig:
    """
    Energy cost per VPU operation (in Joules).
    
    Each value represents the energy consumed to perform that operation
    on a single element. The total energy is:
        energy = num_elements * energy_per_element
    
    TODO: Fill in these values based on your hardware characterization.
    Current values are placeholders (set to 0).
    """
    
    # ========================================================================
    # Basic Arithmetic Operations (per element, FP16)
    # ========================================================================
    
    # Vector addition: a + b
    vec_add: float = 13.3076/100*1e-9  # J per element
    
    # Vector subtraction: a - b
    vec_sub: float = vec_add # J per element
    
    # Vector multiplication: a * b
    vec_mul: float = 12.9986/100*1e-9  # J per element
    
    # Vector division: a / b
    vec_div: float = (17.2698+12.9986)/100*1e-9  # J per element
    
    # Vector multiply-add: a * b + c (fused)
    vec_fma: float = vec_mul + vec_add  # J per element
    
    # ========================================================================
    # Comparison & Selection Operations (per element)
    # ========================================================================
    
    # Vector maximum: max(a, b)
    vec_max: float = 10.098/100*1e-9  # J per element
    
    # Vector minimum: min(a, b)
    vec_min: float = vec_max  # J per element
    
    # ========================================================================
    # Transcendental Functions (per element, FP16)
    # These are typically more expensive than basic arithmetic
    # ========================================================================
    
    # Exponential: exp(x)
    vec_exp: float = 16.3068/100*1e-9  # J per element
    
    # Natural logarithm: log(x)
    vec_log: float = 0.0  # J per element
    
    # Square root: sqrt(x)
    vec_sqrt: float = 16.6492/100*1e-9  # J per element
    
    # Reciprocal square root: 1/sqrt(x) = rsqrt(x)
    vec_rsqrt: float = 16.6492/100*1e-9  # J per element
    
    # Reciprocal: 1/x
    vec_recip: float = 17.2698/100*1e-9  # J per element
    
    # ========================================================================
    # Trigonometric Functions (per element, for RoPE)
    # ========================================================================
    
    # Sine: sin(x)
    vec_sin: float = 17.869/100*1e-9  # J per element
    
    # Cosine: cos(x)
    vec_cos: float = vec_sin  # J per element
    
    # ========================================================================
    # Activation Functions (per element)
    # These may be implemented as compositions or dedicated units
    # ========================================================================
    
    # Sigmoid: 1 / (1 + exp(-x))
    # If dedicated unit, use this. Otherwise, compute as: exp + add + recip
    vec_sigmoid: float = 0.0  # J per element
    
    # Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    vec_tanh: float = 0.0  # J per element
    
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Or approximate: x * sigmoid(1.702 * x)
    vec_gelu: float = 0.0  # J per element
    
    # SiLU (Swish): x * sigmoid(x)
    vec_silu: float = 0.0  # J per element
    
    # ReLU: max(0, x)
    vec_relu: float = 0.0  # J per element
    
    # ========================================================================
    # Reduction Operations (per element participating in reduction)
    # ========================================================================
    
    # Sum reduction: reduce_sum(x)
    vec_reduce_sum: float = 13.046/100*1e-9  # J per element
    
    # Max reduction: reduce_max(x)
    vec_reduce_max: float = 10.098/100*1e-9  # J per element
    
    # ========================================================================
    # Special Operations
    # ========================================================================
    
    # Square: x * x (can use vec_mul, but may have dedicated path)
    vec_square: float = vec_mul  # J per element
    
    # Absolute value: |x|
    vec_abs: float = 0.0  # J per element
    
    # Negate: -x
    vec_neg: float = vec_add  # J per element


# Default configuration instance (with placeholder values)
DEFAULT_VPU_ENERGY = VPUEnergyConfig()


def get_composite_op_energy(config: VPUEnergyConfig) -> dict:
    """
    Return energy cost for composite operations built from primitives.
    
    This helps understand the energy breakdown for complex operations
    if you don't have dedicated hardware units for them.
    
    Returns:
        Dict mapping composite operation names to energy per element.
    """
    return {
        # RMSNorm per element: square + reduce_sum/N + rsqrt + mul
        # Note: reduce_sum is amortized across d_model elements
        "rmsnorm_square": config.vec_square,
        "rmsnorm_rsqrt": config.vec_rsqrt,
        "rmsnorm_scale": config.vec_mul,
        
        # LayerNorm per element: sub (mean) + square + rsqrt + mul + add
        "layernorm_sub_mean": config.vec_sub,
        "layernorm_square": config.vec_square,
        "layernorm_rsqrt": config.vec_rsqrt,
        "layernorm_scale": config.vec_mul,
        "layernorm_shift": config.vec_add,
        
        # Softmax per element: reduce_max + sub + exp + reduce_sum + div
        "softmax_reduce_max": config.vec_reduce_max,
        "softmax_sub": config.vec_sub,
        "softmax_exp": config.vec_exp,
        "softmax_reduce_sum": config.vec_reduce_sum,
        "softmax_div": config.vec_div,
        
        # SiLU per element: sigmoid + mul (or exp + add + recip + mul)
        "silu_sigmoid": config.vec_sigmoid,
        "silu_mul": config.vec_mul,
        
        # GELU approximate per element: mul(1.702) + sigmoid + mul
        "gelu_scale": config.vec_mul,
        "gelu_sigmoid": config.vec_sigmoid,
        "gelu_mul": config.vec_mul,
        
        # RoPE per element: sin + cos + mul + mul + add (complex rotation)
        "rope_sin": config.vec_sin,
        "rope_cos": config.vec_cos,
        "rope_mul1": config.vec_mul,
        "rope_mul2": config.vec_mul,
        "rope_add": config.vec_add,
        
        # Residual add per element
        "residual_add": config.vec_add,
        
        # Attention scaling per element: mul by 1/sqrt(d_k)
        "attn_scale": config.vec_mul,
    }


# ============================================================================
# Energy Calculation Functions
# ============================================================================

def rmsnorm_energy(num_elements: int, config: VPUEnergyConfig = DEFAULT_VPU_ENERGY) -> float:
    """
    Calculate energy for RMSNorm operation.
    
    RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
    
    Operations per element:
      - square: 1
      - reduce_sum: 1 (for mean computation)
      - rsqrt: 1 (shared across elements, but amortized)
      - mul: 2 (scale by rsqrt result, then by gamma)
    
    Args:
        num_elements: Total number of elements (e.g., batch * seq_len * d_model)
        config: VPU energy configuration
    
    Returns:
        Total energy in Joules
    """
    energy = (
        num_elements * config.vec_square +      # x^2
        num_elements * config.vec_reduce_sum +  # sum for mean
        num_elements * config.vec_rsqrt +       # rsqrt (amortized)
        num_elements * config.vec_mul * 2       # scale by rsqrt, then gamma
    )
    return energy


def layernorm_energy(num_elements: int, config: VPUEnergyConfig = DEFAULT_VPU_ENERGY) -> float:
    """
    Calculate energy for LayerNorm operation.
    
    LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
    
    Operations per element:
      - reduce_sum: 1 (for mean)
      - sub: 1 (x - mean)
      - square: 1 (for variance)
      - reduce_sum: 1 (for variance sum)
      - rsqrt: 1 (amortized)
      - mul: 1 (scale)
      - add: 1 (shift)
    """
    energy = (
        num_elements * config.vec_reduce_sum +  # mean
        num_elements * config.vec_sub +         # x - mean
        num_elements * config.vec_square +      # (x - mean)^2
        num_elements * config.vec_reduce_sum +  # var sum
        num_elements * config.vec_rsqrt +       # 1/sqrt(var + eps)
        num_elements * config.vec_mul +         # scale by gamma
        num_elements * config.vec_add           # shift by beta
    )
    return energy


def softmax_energy(num_elements: int, seq_len: int, 
                   config: VPUEnergyConfig = DEFAULT_VPU_ENERGY) -> float:
    """
    Calculate energy for Softmax operation.
    
    Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    Operations per element:
      - reduce_max: 1 (find max for numerical stability)
      - sub: 1 (x - max)
      - exp: 1
      - reduce_sum: 1 (sum of exp)
      - div: 1 (normalize)
    
    Args:
        num_elements: Total number of elements
        seq_len: Sequence length (softmax dimension size)
    """
    energy = (
        num_elements * config.vec_reduce_max +  # max reduction
        num_elements * config.vec_sub +         # subtract max
        num_elements * config.vec_exp +         # exponential
        num_elements * config.vec_reduce_sum +  # sum reduction
        num_elements * config.vec_div           # normalize
    )
    return energy


def silu_energy(num_elements: int, config: VPUEnergyConfig = DEFAULT_VPU_ENERGY) -> float:
    """
    Calculate energy for SiLU (Swish) activation.
    
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    If dedicated sigmoid unit:
      - sigmoid: 1
      - mul: 1
    
    If composed from primitives:
      - neg: 1
      - exp: 1
      - add: 1
      - recip: 1
      - mul: 1
    """
    if config.vec_sigmoid > 0:
        # Use dedicated sigmoid
        energy = num_elements * (config.vec_sigmoid + config.vec_mul)
    else:
        # Composed implementation
        energy = num_elements * (
            config.vec_neg +    # -x
            config.vec_exp +    # exp(-x)
            config.vec_add +    # 1 + exp(-x)
            config.vec_recip +  # 1 / (1 + exp(-x))
            config.vec_mul      # x * sigmoid(x)
        )
    return energy


def gelu_energy(num_elements: int, config: VPUEnergyConfig = DEFAULT_VPU_ENERGY) -> float:
    """
    Calculate energy for GELU activation (approximate version).
    
    GELU(x) ≈ x * sigmoid(1.702 * x)
    
    Operations:
      - mul: 1 (1.702 * x)
      - sigmoid: 1
      - mul: 1 (x * sigmoid(...))
    """
    if config.vec_gelu > 0:
        # Dedicated GELU unit
        return num_elements * config.vec_gelu
    else:
        # Composed implementation
        return num_elements * (
            config.vec_mul +      # 1.702 * x
            config.vec_sigmoid +  # sigmoid
            config.vec_mul        # final mul
        )


def rope_energy(num_elements: int, config: VPUEnergyConfig = DEFAULT_VPU_ENERGY) -> float:
    """
    Calculate energy for Rotary Position Embedding (RoPE).
    
    RoPE applies complex rotation to pairs of elements:
      x_rot = x * cos(θ) + rotate(x) * sin(θ)
    
    For each pair of elements:
      - sin: 1 (can be precomputed/cached)
      - cos: 1 (can be precomputed/cached)
      - mul: 2 (x * cos, rotate(x) * sin)
      - add: 1 (sum the two terms)
    
    Note: sin/cos are often precomputed, so energy may be just memory access.
    This function assumes they need to be computed.
    """
    energy = num_elements * (
        config.vec_sin +   # sin(θ)
        config.vec_cos +   # cos(θ)
        config.vec_mul +   # x * cos(θ)
        config.vec_mul +   # rotate(x) * sin(θ)
        config.vec_add     # sum
    )
    return energy


def residual_add_energy(num_elements: int, 
                        config: VPUEnergyConfig = DEFAULT_VPU_ENERGY) -> float:
    """
    Calculate energy for residual addition.
    
    residual = x + sublayer_output
    """
    return num_elements * config.vec_add


def attention_scale_energy(num_elements: int, 
                           config: VPUEnergyConfig = DEFAULT_VPU_ENERGY) -> float:
    """
    Calculate energy for attention scaling (Q * 1/sqrt(d_k)).
    """
    return num_elements * config.vec_mul


def elementwise_mul_energy(num_elements: int,
                           config: VPUEnergyConfig = DEFAULT_VPU_ENERGY) -> float:
    """
    Calculate energy for element-wise multiplication (e.g., gated FFN).
    
    Used in: gate * up projection (SwiGLU, etc.)
    """
    return num_elements * config.vec_mul


# ============================================================================
# Summary Function
# ============================================================================

def print_required_energy_values():
    """Print all energy values that need to be characterized."""
    print("=" * 70)
    print("VPU Energy Values Required for Full Pipeline Simulation")
    print("=" * 70)
    print("\nAll values should be in Joules (J) per FP16 element operation.")
    print("Fill these values in VPUEnergyConfig based on hardware characterization.\n")
    
    fields = [
        ("vec_add", "Vector addition: a + b"),
        ("vec_sub", "Vector subtraction: a - b"),
        ("vec_mul", "Vector multiplication: a * b"),
        ("vec_div", "Vector division: a / b"),
        ("vec_fma", "Vector fused multiply-add: a * b + c"),
        ("vec_max", "Vector maximum: max(a, b)"),
        ("vec_min", "Vector minimum: min(a, b)"),
        ("vec_exp", "Exponential: exp(x)"),
        ("vec_log", "Natural logarithm: log(x)"),
        ("vec_sqrt", "Square root: sqrt(x)"),
        ("vec_rsqrt", "Reciprocal square root: rsqrt(x)"),
        ("vec_recip", "Reciprocal: 1/x"),
        ("vec_sin", "Sine: sin(x) [for RoPE]"),
        ("vec_cos", "Cosine: cos(x) [for RoPE]"),
        ("vec_sigmoid", "Sigmoid: 1/(1+exp(-x))"),
        ("vec_tanh", "Tanh: tanh(x)"),
        ("vec_gelu", "GELU activation [optional, can compose]"),
        ("vec_silu", "SiLU/Swish: x*sigmoid(x) [optional, can compose]"),
        ("vec_relu", "ReLU: max(0, x)"),
        ("vec_reduce_sum", "Sum reduction (per element contribution)"),
        ("vec_reduce_max", "Max reduction (per element contribution)"),
        ("vec_square", "Square: x^2"),
        ("vec_abs", "Absolute value: |x|"),
        ("vec_neg", "Negate: -x"),
    ]
    
    print("Required Energy Values:")
    print("-" * 70)
    for field_name, description in fields:
        print(f"  {field_name:20s} : {description}")
    
    print("\n" + "=" * 70)
    print("These operations are used in the following LLM components:")
    print("=" * 70)
    print("""
  RMSNorm:
    - vec_square, vec_reduce_sum, vec_rsqrt, vec_mul (x2)
    
  LayerNorm:
    - vec_reduce_sum (x2), vec_sub, vec_square, vec_rsqrt, vec_mul, vec_add
    
  Softmax:
    - vec_reduce_max, vec_sub, vec_exp, vec_reduce_sum, vec_div
    
  SiLU Activation:
    - vec_sigmoid + vec_mul  OR  vec_neg + vec_exp + vec_add + vec_recip + vec_mul
    
  GELU Activation:
    - vec_mul + vec_sigmoid + vec_mul  OR  dedicated vec_gelu
    
  RoPE (Rotary Position Embedding):
    - vec_sin, vec_cos, vec_mul (x2), vec_add
    
  Residual Addition:
    - vec_add
    
  Attention Scaling:
    - vec_mul (Q * 1/sqrt(d_k))
    
  Gated FFN (e.g., SwiGLU):
    - vec_mul (gate * up_projection)
""")


if __name__ == "__main__":
    print_required_energy_values()
