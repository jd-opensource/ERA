"""Unit tests for SharedHostKvChannel (no distributed / GPU required)."""

from __future__ import annotations

import uuid

import pytest
import torch

from joyrlinf.core.shared_host_kv import (
    SharedHostKvChannel,
    resolve_prefix_kv_ipc,
)
from joyrlinf.core.types import PrefixKvHostPool


def _unique_prefix() -> str:
    return f"joyrl_test_{uuid.uuid4().hex[:12]}"


class TestResolvePrefixKvIpc:
    def test_default_off(self):
        assert resolve_prefix_kv_ipc({}) == "off"

    def test_host_shm(self):
        cfg = {
            "actor": {"model": {"openpi": {"prefix_kv_ipc": "host_shm"}}},
        }
        assert resolve_prefix_kv_ipc(cfg) == "host_shm"

    def test_rejects_bad(self):
        with pytest.raises(ValueError, match="prefix_kv_ipc"):
            resolve_prefix_kv_ipc(
                {"actor": {"model": {"openpi": {"prefix_kv_ipc": "cuda"}}}}
            )


class TestSharedHostKvChannel:
    def test_owner_actor_roundtrip(self):
        prefix = _unique_prefix()
        owner = SharedHostKvChannel(prefix, n_slots=2, is_owner=True)
        actor = SharedHostKvChannel(prefix, n_slots=2, is_owner=False)
        try:
            owner.begin_epoch_write()
            sample_kv = torch.randn(4, 3, 2, 8, 16, dtype=torch.float32)
            sample_pad = torch.ones(4, 8, dtype=torch.bool)
            owner.ensure(capacity=3, kv=sample_kv, pad=sample_pad)
            pool = PrefixKvHostPool()
            pool.set_shm_channel(owner)
            owner.bind_pool(pool)

            for row in range(3):
                fi = {
                    "prefix_kv": torch.full(
                        (4, 3, 2, 8, 16), float(row), dtype=torch.float32,
                    ),
                    "prefix_pad_masks": torch.ones(4, 8, dtype=torch.bool),
                }
                pool.schedule_row(row, fi)
                pool.commit_staged_row(row)

            meta = owner.publish(n_rows=3)
            views = actor.view(meta)
            assert views["prefix_kv"].shape == (3, 4, 3, 2, 8, 16)
            assert torch.allclose(
                views["prefix_kv"][1],
                torch.full((4, 3, 2, 8, 16), 1.0),
            )
            assert views["prefix_pad_masks"].shape == (3, 4, 8)

            # Slot 0 still published; begin slot 1 should proceed.
            owner.begin_epoch_write()
            owner.bind_pool(pool)
            fi = {
                "prefix_kv": torch.zeros(4, 3, 2, 8, 16),
                "prefix_pad_masks": torch.zeros(4, 8, dtype=torch.bool),
            }
            pool.schedule_row(0, fi)
            pool.commit_staged_row(0)
            meta1 = owner.publish(n_rows=1)
            assert meta1.slot == 1

            # Reusing slot 0 must wait until ACK.
            actor.send_ack(meta.slot)
            owner.begin_epoch_write()
            assert owner._write_idx == 0
        finally:
            owner.close()
            actor.close()

    def test_pool_ensure_via_channel(self):
        prefix = _unique_prefix()
        owner = SharedHostKvChannel(prefix, n_slots=2, is_owner=True)
        try:
            owner.begin_epoch_write()
            pool = PrefixKvHostPool()
            pool.set_shm_channel(owner)
            sample = torch.randn(2, 2, 2, 4, 8, dtype=torch.bfloat16)
            pad = torch.ones(2, 4, dtype=torch.bool)
            pool.ensure(capacity=2, kv=sample, pad=pad, pin=False)
            assert pool.prefix_kv is not None
            assert pool.prefix_kv.shape[0] == 2
            assert pool._staging_kv is None
            assert owner.ready
        finally:
            owner.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pin_staging_d2h_then_h2h_shm(self):
        """Scheme-2: CUDA → one-row pinned staging → pageable shm."""
        prefix = _unique_prefix()
        owner = SharedHostKvChannel(prefix, n_slots=2, is_owner=True)
        try:
            owner.begin_epoch_write()
            pool = PrefixKvHostPool()
            pool.set_shm_channel(owner)
            sample = torch.randn(2, 2, 2, 4, 8, device="cuda")
            pad = torch.ones(2, 4, dtype=torch.bool, device="cuda")
            pool.ensure(capacity=2, kv=sample, pad=pad, pin=True)
            assert pool._staging_kv is not None and pool._staging_kv.is_pinned()
            assert pool._staging_pad is not None and pool._staging_pad.is_pinned()
            assert pool.prefix_kv is not None
            assert not pool.prefix_kv.is_pinned()

            stream = torch.cuda.Stream()
            for row in range(2):
                kv = torch.full(
                    (2, 2, 2, 4, 8), float(row + 1), device="cuda",
                )
                fi = {"prefix_kv": kv, "prefix_pad_masks": pad}
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    pending = pool.schedule_row(row, fi)
                # Before sync+commit, staging holds the pending view.
                assert pending["prefix_kv"].data_ptr() == pool._staging_kv.data_ptr()
                stream.synchronize()
                committed = pool.commit_staged_row(row)
                assert committed["prefix_kv"].data_ptr() == pool.prefix_kv[row].data_ptr()
                assert torch.allclose(committed["prefix_kv"], kv.cpu())
                # Staging is one chunk; shape matches a single row, not capacity.
                assert pool._staging_kv.shape == kv.shape
        finally:
            owner.close()
