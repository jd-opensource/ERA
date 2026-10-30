"""Tests for per-chunk rollout buffer D2H + PrefixKvHostPool (Phase2)."""

from __future__ import annotations

import pytest
import torch

from joyrlinf.core.types import (
    ChunkStepResult,
    EmbodiedRolloutBatch,
    _stack_dict_list,
    async_d2h_tensor_tree,
)
from joyrlinf.workers.rollout_worker import _resolve_rollout_buffer_d2h


class TestResolveRolloutBufferD2h:
    def test_default_end(self):
        assert _resolve_rollout_buffer_d2h({}) == "end"
        assert (
            _resolve_rollout_buffer_d2h(
                {"actor": {"model": {"openpi": {}}}}
            )
            == "end"
        )

    def test_per_chunk(self):
        assert (
            _resolve_rollout_buffer_d2h(
                {
                    "actor": {
                        "model": {"openpi": {"rollout_buffer_d2h": "per_chunk"}}
                    }
                }
            )
            == "per_chunk"
        )

    def test_rejects_bad_mode(self):
        with pytest.raises(ValueError, match="rollout_buffer_d2h"):
            _resolve_rollout_buffer_d2h(
                {"actor": {"model": {"openpi": {"rollout_buffer_d2h": "nope"}}}}
            )


class TestAsyncD2hTensorTree:
    def test_cpu_tree_detach_and_nested(self):
        t = torch.randn(2, 3, requires_grad=True)
        out = async_d2h_tensor_tree(
            {"a": t, "nested": {"b": t * 2}}, pin=False,
        )
        assert out["a"].device.type == "cpu"
        assert not out["a"].requires_grad
        assert out["nested"]["b"].device.type == "cpu"
        assert torch.allclose(out["a"], t.detach())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_to_pinned_cpu(self):
        t = torch.randn(4, 8, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out = async_d2h_tensor_tree({"x": t, "y": {"z": t * 2}}, pin=True)
        stream.synchronize()
        assert out["x"].device.type == "cpu"
        assert out["x"].is_pinned()
        assert out["y"]["z"].device.type == "cpu"
        assert torch.allclose(out["x"], t.cpu())


class TestCpuStackToDict:
    def test_cpu_resident_chunks_stack_on_cpu(self):
        buffer = EmbodiedRolloutBatch(rollout_epochs=1)
        for _ in range(3):
            buffer.append(
                ChunkStepResult(
                    prev_logprobs=torch.randn(2, 5, 7),
                    rewards=torch.randn(2, 5),
                    dones=torch.zeros(2, 5, dtype=torch.bool),
                    forward_inputs={
                        "prefix_kv": torch.randn(2, 4, 2, 8, 16),
                        "tokens": torch.randint(0, 10, (2, 8)),
                    },
                )
            )
        assert all(t.device.type == "cpu" for t in buffer.prev_logprobs)
        stacked_fi = _stack_dict_list(buffer.forward_inputs, dim=0)
        assert stacked_fi["prefix_kv"].device.type == "cpu"
        assert stacked_fi["prefix_kv"].shape[0] == 3

        batch = buffer.to_dict(device="cpu")
        assert batch["prev_logprobs"].device.type == "cpu"
        assert batch["prefix_kv"].device.type == "cpu"
        assert batch["prefix_kv"].shape == (3, 2, 4, 2, 8, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_to_dict_moves_cuda_stack_to_cpu(self):
        """Regression: device='cpu' must D2H even when lists were CUDA."""
        buffer = EmbodiedRolloutBatch(rollout_epochs=1)
        buffer.append(
            ChunkStepResult(
                prev_logprobs=torch.randn(2, 5, 7, device="cuda"),
                forward_inputs={
                    "prefix_kv": torch.randn(2, 2, 2, 4, 8, device="cuda"),
                },
            )
        )
        batch = buffer.to_dict(device="cpu")
        assert batch["prev_logprobs"].device.type == "cpu"
        assert batch["prefix_kv"].device.type == "cpu"


class TestPrefixKvOffloadHelpers:
    def test_schedule_and_apply_pageable(self):
        from joyrlinf.workers.rollout_worker import (
            _apply_prefix_kv_offload,
            _schedule_d2h_prefix_kv_fields,
        )

        kv = torch.randn(2, 3, 2, 4, 8)
        pad = torch.ones(2, 4, dtype=torch.bool)
        tokens = torch.arange(2)
        fi = {"prefix_kv": kv, "prefix_pad_masks": pad, "tokens": tokens}
        kv_cpu = _schedule_d2h_prefix_kv_fields(fi, pin=False)
        assert set(kv_cpu.keys()) == {"prefix_kv", "prefix_pad_masks"}
        merged = _apply_prefix_kv_offload(fi, kv_cpu)
        assert merged["tokens"] is tokens
        assert merged["prefix_kv"] is not kv
        assert torch.allclose(merged["prefix_kv"], kv)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_schedule_and_apply_pinned(self):
        from joyrlinf.workers.rollout_worker import (
            _apply_prefix_kv_offload,
            _schedule_d2h_prefix_kv_fields,
        )

        kv = torch.randn(2, 3, 2, 4, 8, device="cuda")
        tokens = torch.arange(2, device="cuda")
        fi = {"prefix_kv": kv, "tokens": tokens}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kv_cpu = _schedule_d2h_prefix_kv_fields(fi, pin=True)
        stream.synchronize()
        merged = _apply_prefix_kv_offload(fi, kv_cpu)
        assert merged["prefix_kv"].device.type == "cpu"
        assert merged["prefix_kv"].is_pinned()
        assert merged["tokens"].is_cuda
        assert torch.allclose(merged["prefix_kv"], kv.cpu())


class TestPrefixKvHostPool:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_ensure_once_then_copy_rows(self):
        from joyrlinf.core.types import PrefixKvHostPool
        from joyrlinf.workers.rollout_worker import (
            _apply_prefix_kv_offload,
            ensure_kv_host_pool,
            finalize_host_kv_fields,
        )

        pool = None
        stream = torch.cuda.Stream()
        capacity = 3
        for t in range(2):
            kv = torch.randn(2, 4, 2, 8, 16, device="cuda")
            pad = torch.ones(2, 8, dtype=torch.bool, device="cuda")
            fi = {"prefix_kv": kv, "prefix_pad_masks": pad, "tokens": torch.arange(2)}
            pool = ensure_kv_host_pool(pool, fi, capacity=capacity, pin=True)
            # Side-stream D2H must wait for default-stream producers of kv/pad.
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                pending = pool.schedule_row(t, fi)
            stream.synchronize()
            assert pool.capacity >= capacity
            assert pool.prefix_kv is not None and pool.prefix_kv.is_pinned()
            merged = _apply_prefix_kv_offload(fi, pending)
            assert merged["prefix_kv"].data_ptr() == pool.prefix_kv[t].data_ptr()
            assert torch.allclose(merged["prefix_kv"], kv.cpu())

        ptr = pool.prefix_kv.data_ptr()
        pool.ensure(
            capacity,
            kv=torch.randn(2, 4, 2, 8, 16, device="cuda"),
            pad=torch.ones(2, 8, dtype=torch.bool, device="cuda"),
            pin=True,
        )
        assert pool.prefix_kv.data_ptr() == ptr

        buffer = EmbodiedRolloutBatch(rollout_epochs=1)
        for t in range(2):
            buffer.append(
                ChunkStepResult(
                    prev_logprobs=torch.randn(2, 5, 7),
                    forward_inputs={
                        "prefix_kv": pool.prefix_kv[t],
                        "tokens": torch.randint(0, 10, (2, 8)),
                    },
                )
            )
        finalize_host_kv_fields(pool, buffer)
        batch = buffer.to_dict(device="cpu")
        assert batch["prefix_kv"].is_pinned()
        assert batch["prefix_kv"].shape == (2, 2, 4, 2, 8, 16)
        assert batch["prefix_kv"].data_ptr() == pool.prefix_kv.data_ptr()

    def test_host_pool_cpu_path_no_cuda(self):
        """CPU-only ensure + schedule (no GPU required)."""
        from joyrlinf.workers.rollout_worker import (
            _apply_prefix_kv_offload,
            ensure_kv_host_pool,
            finalize_host_kv_fields,
        )

        pool = None
        capacity = 2
        for t in range(2):
            kv = torch.randn(2, 4, 2, 8, 16)
            pad = torch.ones(2, 8, dtype=torch.bool)
            fi = {"prefix_kv": kv, "prefix_pad_masks": pad}
            pool = ensure_kv_host_pool(pool, fi, capacity=capacity, pin=False)
            pending = pool.schedule_row(t, fi)
            merged = _apply_prefix_kv_offload(fi, pending)
            assert torch.allclose(merged["prefix_kv"], kv)

        buffer = EmbodiedRolloutBatch(rollout_epochs=1)
        for t in range(2):
            buffer.append(
                ChunkStepResult(
                    prev_logprobs=torch.randn(2, 5, 7),
                    forward_inputs={"prefix_kv": pool.prefix_kv[t]},
                )
            )
        finalize_host_kv_fields(pool, buffer)
        batch = buffer.to_dict(device="cpu")
        assert batch["prefix_kv"].shape == (2, 2, 4, 2, 8, 16)


class TestAppendBootstrapPredict:
    def test_keeps_prev_values_drops_forward_inputs(self):
        """Bootstrap predict must only contribute the GAE value; its
        forward_inputs/KV are never consumed (flatten_for_training drops
        the [T+1] row) so nothing may land in the buffer or host pool."""
        from joyrlinf.workers.rollout_worker import CombinedEnvRolloutWorker

        buffer = EmbodiedRolloutBatch(rollout_epochs=1)
        final_result = {
            "prev_values": torch.ones(2, 1),
            "forward_inputs": {"prefix_kv": torch.zeros(2, 4)},
        }
        CombinedEnvRolloutWorker._append_bootstrap_predict(
            None, buffer, final_result,
        )
        assert len(buffer.prev_values) == 1
        assert buffer.forward_inputs == []

    def test_no_prev_values_is_noop(self):
        from joyrlinf.workers.rollout_worker import CombinedEnvRolloutWorker

        buffer = EmbodiedRolloutBatch(rollout_epochs=1)
        CombinedEnvRolloutWorker._append_bootstrap_predict(
            None, buffer, {"forward_inputs": {"prefix_kv": torch.zeros(1)}},
        )
        assert buffer.prev_values == []
        assert buffer.forward_inputs == []
