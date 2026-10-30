"""VLA model factory (OpenPI / GR00T)."""

from __future__ import annotations

import torch
from omegaconf import DictConfig


def build_vla_model(model_cfg: DictConfig, device: torch.device, mode: str):
    """Build a VLA wrapper for rollout (inference) or actor (training)."""
    model_type = str(model_cfg.get("model_type", "openpi")).lower()
    if model_type == "openpi":
        from joyrlinf.models.openpi_model import OpenPiModel

        return OpenPiModel(model_cfg, device, mode=mode)
    if model_type == "gr00t":
        from joyrlinf.models.gr00t.model import Gr00tModel

        return Gr00tModel(model_cfg, device, mode=mode)
    if model_type in ("openvla_oft", "openvla-oft"):
        from joyrlinf.models.openvla_oft_model import OpenVlaOftModel

        return OpenVlaOftModel(model_cfg, device, mode=mode)
    raise ValueError(
        f"Unsupported model_type={model_type!r}. Supported: openpi, gr00t, openvla_oft"
    )
