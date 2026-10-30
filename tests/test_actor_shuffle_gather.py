"""Tests for index-only actor microbatch gather (no full-tensor shuffle copy)."""

from __future__ import annotations

import torch

from joyrlinf.workers.actor_worker import _gather_microbatch


def test_gather_microbatch_matches_materialized_shuffle():
    total, mbs, dim = 16, 4, 8
    flat = {
        "prev_logprobs": torch.arange(total * dim, dtype=torch.float32).reshape(total, dim),
        "prefix_kv": torch.arange(total * 3, dtype=torch.float32).reshape(total, 3),
        "scalar": torch.tensor(1.5),  # 0-dim — not row-aligned
    }
    perm = torch.randperm(total)
    # Materialized reference (old path).
    ref = {}
    for k, v in flat.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == total:
            ref[k] = v[perm]
        else:
            ref[k] = v

    for start in range(0, total, mbs):
        end = min(start + mbs, total)
        mb = _gather_microbatch(flat, perm, start, end)
        assert torch.equal(mb["prev_logprobs"], ref["prev_logprobs"][start:end])
        assert torch.equal(mb["prefix_kv"], ref["prefix_kv"][start:end])
        assert mb["scalar"] is flat["scalar"]


def test_gather_does_not_alias_full_storage_for_indexed_rows():
    """Gather must own its rows (safe to free/overwrite source later)."""
    total = 8
    src = torch.randn(total, 4)
    perm = torch.arange(total - 1, -1, -1)
    mb = _gather_microbatch({"x": src}, perm, 0, 2)
    src.zero_()
    assert not torch.allclose(mb["x"], torch.zeros_like(mb["x"]))
