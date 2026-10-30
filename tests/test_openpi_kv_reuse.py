"""CPU-only tests for OpenPI prefix-KV reuse / expert-only actor (S1).

Heavy GPU equivalence tests are deferred until a free GPU is available.

Run inside the openpi container::

    cd /ufs_500T/jh/code/joyrlinf-dev/colocated_async/joyrlinf
    PYTHONPATH=. /opt/venv/openpi/bin/python -m pytest \
        tests/test_openpi_kv_reuse.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("openpi")

from joyrlinf.models.openpi_action_model import (  # noqa: E402
    OpenPi0ForRLActionPrediction,
    _detach_kv_tree,
    _pack_prefix_kv,
    _unpack_prefix_kv,
)


class TestDetachKvTree:
    def test_detaches_tensor_tree(self):
        k = torch.randn(2, 1, 4, 8, requires_grad=True) * 2
        v = torch.randn(2, 1, 4, 8, requires_grad=True) * 2
        out = _detach_kv_tree([(k, v)])
        assert isinstance(out, list) and isinstance(out[0], tuple)
        assert not out[0][0].requires_grad
        assert not out[0][1].requires_grad
        assert torch.equal(out[0][0], k.detach())

    def test_handles_dynamic_cache(self):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        k = (torch.randn(2, 1, 4, 8) * 2).requires_grad_(True)
        v = (torch.randn(2, 1, 4, 8) * 2).requires_grad_(True)
        cache.update(k * 1.0, v * 1.0, 0)
        assert cache.key_cache[0].requires_grad
        out = _detach_kv_tree(cache)
        assert not out.key_cache[0].requires_grad
        assert not out.value_cache[0].requires_grad


class TestPackUnpackPrefixKv:
    def _make_cache(self, b=2, layers=3, s=16, d=8):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for idx in range(layers):
            cache.update(
                torch.randn(b, 1, s, d, dtype=torch.bfloat16),
                torch.randn(b, 1, s, d, dtype=torch.bfloat16),
                idx,
            )
        return cache

    def test_roundtrip_bitwise(self):
        cache = self._make_cache()
        packed = _pack_prefix_kv(cache)
        assert list(packed.shape) == [2, 3, 2, 16, 8]
        assert packed.dtype == torch.bfloat16
        rebuilt = _unpack_prefix_kv(packed)
        for idx in range(3):
            assert torch.equal(rebuilt.key_cache[idx], cache.key_cache[idx])
            assert torch.equal(
                rebuilt.value_cache[idx], cache.value_cache[idx]
            )

    def test_pack_accepts_tuple_layout(self):
        layers = [
            (torch.randn(2, 1, 8, 4), torch.randn(2, 1, 8, 4)),
            (torch.randn(2, 1, 8, 4), torch.randn(2, 1, 8, 4)),
        ]
        packed = _pack_prefix_kv(layers)
        rebuilt = _unpack_prefix_kv(packed)
        assert len(rebuilt.key_cache) == 2

    def test_pack_rejects_non_mqa(self):
        layers = [(torch.randn(2, 2, 8, 4), torch.randn(2, 2, 8, 4))]
        with pytest.raises(ValueError, match="MQA"):
            _pack_prefix_kv(layers)

    def test_unpack_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="packed prefix KV"):
            _unpack_prefix_kv(torch.randn(2, 3, 16, 8))


class TestExpertOnlyActorStrip:
    def _stub(self, *, reuse=True, paligemma="sentinel"):
        return SimpleNamespace(
            config=SimpleNamespace(reuse_rollout_prefix_kv=reuse),
            paligemma_with_expert=SimpleNamespace(paligemma=paligemma),
        )

    def test_strip_requires_reuse_flag(self):
        stub = self._stub(reuse=False)
        with pytest.raises(ValueError, match="reuse_rollout_prefix_kv"):
            OpenPi0ForRLActionPrediction.strip_vlm_for_expert_only_actor(stub)

    def test_strip_sets_none_and_guards_raise(self):
        stub = self._stub(reuse=True)
        OpenPi0ForRLActionPrediction.strip_vlm_for_expert_only_actor(stub)
        assert stub.paligemma_with_expert.paligemma is None
        assert OpenPi0ForRLActionPrediction.vlm_stripped.fget(stub) is True
        with pytest.raises(RuntimeError, match="expert_only_actor"):
            OpenPi0ForRLActionPrediction._ensure_vlm_present(
                stub, "sample_actions"
            )

    def test_guard_passes_with_vlm_present(self):
        stub = self._stub(reuse=True)
        OpenPi0ForRLActionPrediction._ensure_vlm_present(stub, "x")


class TestInjectionGuards:
    def _stub(self, *, train_expert_only=True, use_vlm_value=False):
        return SimpleNamespace(
            config=SimpleNamespace(train_expert_only=train_expert_only),
            use_vlm_value=use_vlm_value,
        )

    def _call(self, stub, *, kv, pad):
        state = torch.zeros(2, 4)
        return OpenPi0ForRLActionPrediction.get_log_prob_value(
            stub,
            images=None,
            img_masks=None,
            lang_tokens=None,
            lang_masks=None,
            state=state,
            chains=None,
            denoise_inds=None,
            past_key_values=kv,
            prefix_pad_masks=pad,
        )

    def test_requires_pad_masks(self):
        with pytest.raises(ValueError, match="prefix_pad_masks"):
            self._call(self._stub(), kv=[(torch.zeros(1),) * 2], pad=None)

    def test_rejects_value_after_vlm(self):
        with pytest.raises(ValueError, match="value_after_vlm"):
            self._call(
                self._stub(use_vlm_value=True),
                kv=[(torch.zeros(1),) * 2],
                pad=torch.ones(2, 4, dtype=torch.bool),
            )

    def test_requires_train_expert_only(self):
        with pytest.raises(ValueError, match="train_expert_only"):
            self._call(
                self._stub(train_expert_only=False),
                kv=[(torch.zeros(1),) * 2],
                pad=torch.ones(2, 4, dtype=torch.bool),
            )
