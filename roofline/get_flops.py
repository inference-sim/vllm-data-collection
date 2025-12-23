import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareCalib:
    """Hardware configuration data structure"""
    tflops_eff: float # e.g., 250.0 for H100 BF16
    bw_eff_bytes_s: float # e.g., 1.2e12 for 1.2 TB/s effective
    t_overhead_micros: float = 0.05

@dataclass
class ModelConfig:
    """Model configuration data structure."""
    num_params: int  # actual number of parameters
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    inferred_precision: str = "fp16"

    def __post_init__(self):
        # Default num_kv_heads to num_heads if not specified
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

hardware_list = {
    "H100": HardwareCalib(
        tflops_eff=989.5,
        bw_eff_bytes_s=3.35e12,
        t_overhead_micros=50
    )
}

def calculate_transformer_flops(
    hf_config,
    sequence_length: int,
    include_attention: bool = True,
    include_mlp: bool = True,
) -> dict[str, float]:
    """
    Calculate FLOPS for transformer components based on roofline analysis methodology.

    This function implements the detailed FLOPS calculation from academic literature,
    accounting for both linear operations (MLP, projections) and quadratic attention terms.

    References:
    - "Attention Is All You Need" (Vaswani et al.)
    - JAX Scaling Book: https://jax-ml.github.io/scaling-book/transformers/
    - OpenAI Scaling Laws paper

    Args:
        hf_config: Model configuration with architecture details
        sequence_length: Input sequence length (T)
        include_attention: Whether to include attention FLOPS (default True)
        include_mlp: Whether to include MLP/feedforward FLOPS (default True)

    Returns:
        Dictionary with FLOPS breakdown:
        - 'attention_qkv': Query/Key/Value projection FLOPS
        - 'attention_scores': QK^T matmul FLOPS
        - 'attention_softmax': Softmax operation FLOPS
        - 'attention_output': Attention-Value matmul FLOPS
        - 'attention_proj': Output projection FLOPS
        - 'mlp': Feed-forward network FLOPS
        - 'total': Total FLOPS per token
    """
    # Model parameters
    d_model = hf_config.hidden_dim  # Hidden dimension
    n_layers = hf_config.num_layers  # Number of transformer layers
    n_heads = hf_config.num_heads    # Number of attention heads
    n_kv_heads = hf_config.num_kv_heads  # Number of KV heads (for GQA/MQA)
    d_head = d_model // n_heads         # Dimension per head

    # Calculate intermediate dimension (typically 4x hidden dim in standard transformers)
    # We'll estimate it from total parameters if not directly available
    # Standard transformer: d_ff ≈ 4 * d_model
    d_ff = 4 * d_model  # Feed-forward intermediate dimension

    flops_breakdown = {}

    if include_attention:
        # 1. QKV Projections: Linear transformations to create queries, keys, values
        # Q: [B, T, d_model] @ [d_model, d_model] -> [B, T, d_model]
        # K,V: [B, T, d_model] @ [d_model, d_kv] -> [B, T, d_kv] where d_kv = n_kv_heads * d_head
        d_kv = n_kv_heads * d_head
        qkv_flops = 2 * sequence_length * (d_model * d_model + 2 * d_model * d_kv)
        flops_breakdown['attention_qkv'] = qkv_flops * n_layers

        # 2. Attention Scores: Q @ K^T
        # [B, n_heads, T, d_head] @ [B, n_heads, d_head, T] -> [B, n_heads, T, T]
        # Note: For GQA/MQA, keys/values are broadcast across query heads
        qk_flops = 2 * n_heads * sequence_length * sequence_length * d_head
        flops_breakdown['attention_scores'] = qk_flops * n_layers

        # 3. Softmax: Applied to attention scores
        # Includes exponential, sum reduction, and division operations
        # Approximation: ~3 ops per element for softmax
        softmax_flops = 3 * n_heads * sequence_length * sequence_length
        flops_breakdown['attention_softmax'] = softmax_flops * n_layers

        # 4. Attention Output: Softmax @ V
        # [B, n_heads, T, T] @ [B, n_heads, T, d_head] -> [B, n_heads, T, d_head]
        av_flops = 2 * n_heads * sequence_length * sequence_length * d_head
        flops_breakdown['attention_output'] = av_flops * n_layers

        # 5. Output Projection: Final linear layer
        # [B, T, d_model] @ [d_model, d_model] -> [B, T, d_model]
        proj_flops = 2 * sequence_length * d_model * d_model
        flops_breakdown['attention_proj'] = proj_flops * n_layers
    else:
        for key in ['attention_qkv', 'attention_scores', 'attention_softmax',
                   'attention_output', 'attention_proj']:
            flops_breakdown[key] = 0.0

    if include_mlp:
        # MLP/Feed-Forward Network: Two linear transformations with activation
        # Up projection: [B, T, d_model] @ [d_model, d_ff] -> [B, T, d_ff]
        # Down projection: [B, T, d_ff] @ [d_ff, d_model] -> [B, T, d_model]
        # Note: Modern transformers often use SwiGLU which has additional complexity
        mlp_flops = 2 * sequence_length * (d_model * d_ff + d_ff * d_model)
        flops_breakdown['mlp'] = mlp_flops * n_layers
    else:
        flops_breakdown['mlp'] = 0.0

    # Total FLOPS per token
    flops_breakdown['total'] = sum(flops_breakdown.values())

    return flops_breakdown


def calculate_memory_access_bytes(
    hf_config: "ModelConfig",
    sequence_length: int,
    batch_size: int = 1,
    bytes_per_param: int = 2,  # FP16 = 2 bytes
    include_kv_cache: bool = True,
) -> dict[str, float]:
    """
    Calculate memory access patterns for transformer inference following roofline methodology.

    This function computes the actual bytes that need to be moved from memory during
    transformer inference, which is crucial for determining arithmetic intensity and
    whether operations are memory-bound or compute-bound.

    References:
    - Roofline analysis papers and methodology
    - "LLM Inference Unveiled: Survey and Roofline Model Insights" (arXiv:2402.16363)

    Args:
        hf_config: Model architecture configuration
        sequence_length: Current sequence length (including generated tokens)
        batch_size: Number of concurrent sequences
        bytes_per_param: Bytes per parameter (2 for FP16, 1 for FP8, etc.)
        include_kv_cache: Whether to include KV cache access in calculation

    Returns:
        Dictionary with memory access breakdown:
        - 'model_weights': Bytes for loading model parameters
        - 'kv_cache': Bytes for KV cache access
        - 'activations': Bytes for intermediate activations
        - 'total': Total bytes accessed per token
    """
    d_model = hf_config.hidden_dim
    n_layers = hf_config.num_layers
    n_kv_heads = hf_config.num_kv_heads
    d_head = d_model // hf_config.num_heads

    memory_breakdown = {}

    # 1. Model Weights Access
    # During decode, we need to load:
    # - QKV projection weights: d_model * (d_model + 2*n_kv_heads*d_head) per layer
    # - Output projection weights: d_model * d_model per layer
    # - MLP weights: d_model * d_ff * 2 per layer (up and down projections)
    d_kv = n_kv_heads * d_head
    d_ff = 4 * d_model  # Estimated feed-forward dimension

    weights_per_layer = (
        d_model * (d_model + 2 * d_kv) +  # QKV projections
        d_model * d_model +                # Output projection
        d_model * d_ff * 2                 # MLP up/down projections
    )

    total_model_weights = weights_per_layer * n_layers * bytes_per_param
    memory_breakdown['model_weights'] = total_model_weights

    # 2. KV Cache Access
    if include_kv_cache:
        # KV cache size per token: 2 (K+V) * n_layers * n_kv_heads * d_head * bytes_per_param
        kv_per_token = 2 * n_layers * n_kv_heads * d_head * bytes_per_param

        # During decode: we access all previous tokens' KV cache
        # During prefill: we write to KV cache (similar access pattern)
        total_kv_access = kv_per_token * sequence_length * batch_size
        memory_breakdown['kv_cache'] = total_kv_access
    else:
        memory_breakdown['kv_cache'] = 0.0

    # 3. Activations
    # Intermediate activations during forward pass
    # This includes attention scores, MLP activations, etc.
    # Approximation based on sequence length and model dimensions
    activations_size = (
        batch_size * sequence_length * d_model * bytes_per_param +  # Input/output activations
        batch_size * hf_config.num_heads * sequence_length * sequence_length * bytes_per_param  # Attention matrices
    )
    memory_breakdown['activations'] = activations_size

    # Total memory access per token generation
    memory_breakdown['total'] = sum(memory_breakdown.values())

    return memory_breakdown

def get_hf_config_and_precision_from_hf(model_id: str, hf_config: dict, step_config: dict, model_config: dict, GPU: str) -> ModelConfig:
    """
    Extracts model configuration with inferred precision.

    Args:
        model_id: HuggingFace model id
        hf_config: HuggingFace config.json dict
        step_config: BLIS-generated step config dict
        {
        "prefill_requests": [
            {
            "progress_index": 256,
            "num_new_prefill_tokens": 128
            },
            {
            ...
            }
        ],
        "decode_requests": [
            {
            "progress_index": 270,
            "num_new_decode_tokens": 1
            },
            {
            ...
            }
        ]
        }

    Returns:
        Step time: int in microseconds
    """
    hw = hardware_list[GPU]
    try:
        # Extract basic config parameters
        h = hf_config["hidden_size"]
        n_layers = hf_config["num_hidden_layers"]
        v = hf_config["vocab_size"]
        n_heads = hf_config.get("num_attention_heads", 0)
        n_kv_heads = hf_config.get("num_key_value_heads", n_heads)        

        hf_config = ModelConfig(
            num_params=model_config["total_params"],
            num_layers=n_layers,
            hidden_dim=h,
            vocab_size=v,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            inferred_precision=model_config["precision"],
        )

    except KeyError as e:
        raise KeyError(f"Could not find required key {e} in config.json for {model_id}")
    
    t_compute_s = 0
    t_memory_s = 0

    for request in step_config["prefill_requests"]:
        input_length = request["progress_index"]
        concurrency = len(step_config["prefill_requests"]) + len(step_config["decode_requests"])
        prefill_flops_breakdown = calculate_transformer_flops(
            hf_config=hf_config,
            sequence_length=input_length,
            include_attention=True,
            include_mlp=True,
        )

        prefill_flops_per_token = prefill_flops_breakdown['total']
        prefill_memory_breakdown = calculate_memory_access_bytes(
            hf_config=hf_config,
            sequence_length=input_length,
            batch_size=concurrency,
            bytes_per_param=model_config["bytes_per_param"],
            include_kv_cache=True,
        )

        prefill_memory_bytes = prefill_memory_breakdown['total']
        t_compute_s += (prefill_flops_per_token * request["num_new_prefill_tokens"])
        t_memory_s += (prefill_memory_bytes / hw.bw_eff_bytes_s)

    for request in step_config["decode_requests"]:
        decode_flops_breakdown = calculate_transformer_flops(
            hf_config=hf_config,
            sequence_length=1,
            include_attention=True,
            include_mlp=True,
        )
        decode_flops_per_token = decode_flops_breakdown['total']

        decode_memory_breakdown = calculate_memory_access_bytes(
            hf_config=hf_config,
            sequence_length=request["progress_index"],  # Access all previous tokens in KV cache
            batch_size=len(step_config["prefill_requests"]) + len(step_config["decode_requests"]),
            bytes_per_param=model_config["bytes_per_param"],
            include_kv_cache=True,
        )
        decode_memory_bytes = decode_memory_breakdown['total']
        t_compute_s += decode_flops_per_token * request["num_new_decode_tokens"]
        t_memory_s += (decode_memory_bytes / hw.bw_eff_bytes_s)
    
    t_compute_micros = int((t_compute_s / (hw.tflops_eff * 1e12)) * 1e6)
    t_memory_micros = int(t_memory_s * 1e6)
    return max(t_compute_micros, t_memory_micros) + hw.t_overhead_micros

def get_param_configs(model_id, model_config):
    from llm_optimizer.common import get_precision_bytes_per_param, infer_precision_from_config, calculate_model_parameters_from_config
    precision = infer_precision_from_config(model_config, model_id)
    bytes_per_param = get_precision_bytes_per_param(precision)
    total_params = calculate_model_parameters_from_config(model_config)
    print(precision, bytes_per_param, total_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--model-id", 
                        help="LLM name")
    parser.add_argument("--model-config-folder",
                        help="Model config folder")
    args = parser.parse_args()
    hf_config_path = os.path.join(args.model_config_folder, "config.json")
    with open(hf_config_path, 'r+') as f:
        hf_config = json.load(f)
    get_param_configs(args.model_id, hf_config)
