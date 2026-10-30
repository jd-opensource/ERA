"""gqa_eager attention: equivalence with eager reference + registration."""
import pytest
import torch

from joyrlinf.models.gqa_attention import (
    GQA_EAGER_IMPL,
    gqa_eager_attention_forward,
    register_gqa_eager,
    resolve_attn_implementation,
)


class _FakeAttnModule(torch.nn.Module):
    def __init__(self, num_key_value_groups: int):
        super().__init__()
        self.num_key_value_groups = num_key_value_groups
        self.is_causal = False


def _eager_reference(module, query, key, value, attention_mask, scaling):
    """Mirror of openpi's patched eager_attention_forward (repeat_kv path)."""
    n_rep = query.shape[1] // key.shape[1]
    key_states = key.repeat_interleave(n_rep, dim=1)
    value_states = value.repeat_interleave(n_rep, dim=1)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(
        attn_weights, p=0.0, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous()


def _additive_mask(batch, q_len, kv_len):
    """openpi-style 4D additive mask: 0 keep / -2.38e38 drop (fp32)."""
    keep = torch.rand(batch, 1, q_len, kv_len) > 0.2
    keep[..., 0] = True  # never mask out a full row
    return torch.where(keep, 0.0, -2.3819763e38)


class TestGqaEagerNumerics:
    @pytest.mark.parametrize("kv_heads", [1, 2, 8])
    def test_matches_eager_reference(self, kv_heads):
        torch.manual_seed(0)
        b, heads, q_len, kv_len, d = 3, 8, 10, 64, 32
        q = torch.randn(b, heads, q_len, d)
        k = torch.randn(b, kv_heads, kv_len, d)
        v = torch.randn(b, kv_heads, kv_len, d)
        mask = _additive_mask(b, q_len, kv_len)
        scaling = d**-0.5

        module = _FakeAttnModule(heads // kv_heads)
        out, weights = gqa_eager_attention_forward(
            module, q, k, v, mask, scaling=scaling, dropout=0.0,
        )
        ref = _eager_reference(module, q, k, v, mask, scaling)

        assert weights is None
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)

    def test_bf16_matches_eager_reference(self):
        torch.manual_seed(1)
        b, heads, q_len, kv_len, d = 2, 8, 6, 48, 32
        q = torch.randn(b, heads, q_len, d, dtype=torch.bfloat16)
        k = torch.randn(b, 1, kv_len, d, dtype=torch.bfloat16)
        v = torch.randn(b, 1, kv_len, d, dtype=torch.bfloat16)
        mask = _additive_mask(b, q_len, kv_len)  # fp32, promotes like eager

        module = _FakeAttnModule(heads)
        out, _ = gqa_eager_attention_forward(
            module, q, k, v, mask, scaling=d**-0.5, dropout=0.0,
        )
        ref = _eager_reference(module, q, k, v, mask, d**-0.5)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_no_expansion_saved_for_backward(self):
        """Grads must land on the unexpanded [b, kv_heads, ...] leaves and the
        graph must not retain repeat_kv-sized buffers."""
        torch.manual_seed(2)
        b, heads, q_len, kv_len, d = 2, 8, 4, 16, 8
        q = torch.randn(b, heads, q_len, d, requires_grad=True)
        k = torch.randn(b, 1, kv_len, d, requires_grad=True)
        v = torch.randn(b, 1, kv_len, d, requires_grad=True)
        mask = _additive_mask(b, q_len, kv_len)

        module = _FakeAttnModule(heads)
        out, _ = gqa_eager_attention_forward(
            module, q, k, v, mask, scaling=d**-0.5, dropout=0.0,
        )
        out.sum().backward()
        assert k.grad is not None and k.grad.shape == k.shape
        assert v.grad is not None and v.grad.shape == v.shape

        # Gradient correctness vs the repeat_kv reference.
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        _eager_reference(module, q2, k2, v2, mask, d**-0.5).sum().backward()
        torch.testing.assert_close(q.grad, q2.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(k.grad, k2.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(v.grad, v2.grad, rtol=1e-5, atol=1e-6)

    def test_mask_sliced_to_kv_length(self):
        """4D masks longer than kv (static-cache style) are right-trimmed."""
        b, heads, q_len, kv_len, d = 2, 8, 4, 16, 8
        q = torch.randn(b, heads, q_len, d)
        k = torch.randn(b, 1, kv_len, d)
        v = torch.randn(b, 1, kv_len, d)
        mask = _additive_mask(b, q_len, kv_len + 5)

        module = _FakeAttnModule(heads)
        out, _ = gqa_eager_attention_forward(
            module, q, k, v, mask, scaling=d**-0.5, dropout=0.0,
        )
        ref = _eager_reference(module, q, k, v, mask[..., :kv_len], d**-0.5)
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


class TestResolveAttnImplementation:
    def test_eager_passthrough(self):
        assert resolve_attn_implementation("eager") == "eager"
        assert resolve_attn_implementation(" Eager ") == "eager"

    def test_gqa_eager_registers_custom_key(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        assert resolve_attn_implementation("gqa_eager") == GQA_EAGER_IMPL
        assert (
            ALL_ATTENTION_FUNCTIONS[GQA_EAGER_IMPL] is gqa_eager_attention_forward
        )

    def test_register_is_idempotent(self):
        register_gqa_eager()
        register_gqa_eager()

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="attn_implementation"):
            resolve_attn_implementation("sdpa")
