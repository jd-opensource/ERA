"""Memory-efficient GQA eager attention for the openpi Gemma stack.

The stock eager path (and transformers 4.53's sdpa path) materializes
``repeat_kv`` copies of K/V ([b, kv_heads, s, d] -> [b, heads, s, d]) as
matmul inputs, so autograd saves the expanded copies for backward. For the
Pi0 expert that is the dominant training activation: ~978-token prefix KV
expanded 1 -> 8 heads x 18 layers x 4 flow steps ~= 37 GiB per 64-sample
microbatch.

Fused SDPA kernels cannot replace it here: Pi0's head_dim=256 + arbitrary 4D
mask is rejected by flash / cuDNN / memory-efficient backends on torch 2.6
(probed on H-series; only the math fallback runs, which holds *more* memory
than eager).

This implementation instead folds the query-head groups into the row
dimension (``[b, h, q, d] -> [b, kv_heads, groups*q, d]``) so both matmuls
run directly against the unexpanded K/V. Autograd then saves only views /
the original tensors. Same GEMMs, same op order as the patched
``eager_attention_forward`` -> bit-exact outputs, ~17x less graph-held
memory per layer call (measured 553 -> 32 MiB at mbs=64).

Registered under the custom key ``"gqa_eager"``; the patched openpi Gemma
dispatches through ``ALL_ATTENTION_FUNCTIONS[config._attn_implementation]``
and passes pre-built 4D masks, which transformers' mask pipeline returns
as-is for custom implementations.
"""
from __future__ import annotations

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

GQA_EAGER_IMPL = "gqa_eager"


def gqa_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Bit-exact eager attention without materializing repeat_kv copies."""
    batch, num_heads, q_len, head_dim = query.shape
    kv_heads = key.shape[1]
    groups = num_heads // kv_heads
    if scaling is None:
        scaling = head_dim**-0.5

    # Fold head groups into rows: per-(batch, kv_head) GEMM against the
    # unexpanded K/V. Row-wise dot products are unchanged -> bit-exact.
    query_grouped = query.reshape(batch, kv_heads, groups * q_len, head_dim)
    attn_weights = (
        torch.matmul(query_grouped, key.transpose(2, 3)).reshape(
            batch, num_heads, q_len, -1
        )
        * scaling
    )
    if attention_mask is not None:
        # openpi passes 4D additive masks (0 keep / -2.38e38 drop, fp32).
        attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    weights_grouped = attn_weights.reshape(batch, kv_heads, groups * q_len, -1)
    attn_output = torch.matmul(weights_grouped, value).reshape(
        batch, num_heads, q_len, head_dim
    )
    return attn_output.transpose(1, 2).contiguous(), None


def register_gqa_eager() -> None:
    if GQA_EAGER_IMPL not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(GQA_EAGER_IMPL, gqa_eager_attention_forward)


def resolve_attn_implementation(name: str) -> str:
    """Map the config value to a transformers attention key.

    "eager" is passed through; "gqa_eager" registers the implementation
    above. Fused SDPA/flash are deliberately not offered: Pi0's
    head_dim=256 + arbitrary mask has no usable fused kernel on torch 2.6.
    """
    impl = str(name).lower().strip()
    if impl == "eager":
        return "eager"
    if impl == "gqa_eager":
        register_gqa_eager()
        return GQA_EAGER_IMPL
    raise ValueError(
        f"attn_implementation must be 'eager' or 'gqa_eager', got {name!r}"
    )
