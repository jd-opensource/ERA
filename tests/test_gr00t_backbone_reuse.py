"""CPU-only guards for GR00T rollout-backbone reuse."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("gr00t")

from joyrlinf.models.gr00t.action_model import (  # noqa: E402
    GR00T_N1_5_ForRLActionPrediction,
)
from joyrlinf.workers.rollout_worker import _resolve_rollout_buffer_d2h  # noqa: E402


class TestExpertOnlyActorStrip:
    @staticmethod
    def _stub(*, reuse: bool, trainable: bool = False):
        backbone = torch.nn.Linear(4, 4)
        for parameter in backbone.parameters():
            parameter.requires_grad = trainable
        return SimpleNamespace(
            reuse_rollout_backbone_features=reuse,
            backbone=backbone,
        )

    def test_strip_requires_cached_backbone(self):
        stub = self._stub(reuse=False)
        with pytest.raises(ValueError, match="reuse_rollout_backbone_features"):
            GR00T_N1_5_ForRLActionPrediction.strip_backbone_for_expert_only_actor(
                stub
            )

    def test_strip_rejects_trainable_backbone(self):
        stub = self._stub(reuse=True, trainable=True)
        with pytest.raises(ValueError, match="trainable GR00T backbone"):
            GR00T_N1_5_ForRLActionPrediction.strip_backbone_for_expert_only_actor(
                stub
            )

    def test_strip_frozen_backbone(self):
        stub = self._stub(reuse=True)
        GR00T_N1_5_ForRLActionPrediction.strip_backbone_for_expert_only_actor(stub)
        assert stub.backbone is None
        assert (
            GR00T_N1_5_ForRLActionPrediction.backbone_stripped.fget(stub) is True
        )


def test_gr00t_per_chunk_d2h_config():
    cfg = {
        "actor": {
            "model": {
                "model_type": "gr00t",
                "gr00t": {"rollout_buffer_d2h": "per_chunk"},
            }
        }
    }
    assert _resolve_rollout_buffer_d2h(cfg) == "per_chunk"
