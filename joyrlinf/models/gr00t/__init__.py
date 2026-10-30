"""GR00T N1.5 model loading helpers for joyrlinf."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


_PATCHED = False


def apply_embodiment_tag_patches() -> None:
    """Patch gr00t package tags so joyrlinf embodiments validate and map correctly."""
    global _PATCHED
    if _PATCHED:
        return

    import gr00t.data.embodiment_tags as gr00t_tags
    import gr00t.data.schema as gr00t_schema

    from joyrlinf.models.gr00t import embodiment_tags as joy_tags  # noqa: PLC0415

    gr00t_tags.EmbodimentTag = joy_tags.EmbodimentTag
    gr00t_tags.EMBODIMENT_TAG_MAPPING = joy_tags.EMBODIMENT_TAG_MAPPING
    gr00t_schema.EmbodimentTag = joy_tags.EmbodimentTag
    gr00t_schema.DatasetMetadata.model_rebuild(force=True)

    try:
        import gr00t.model.transforms as gr00t_transforms

        gr00t_transforms.EMBODIMENT_TAG_MAPPING = joy_tags.EMBODIMENT_TAG_MAPPING
    except ImportError:
        pass

    _PATCHED = True


def _to_rl_head_config(cfg: Any) -> SimpleNamespace:
    if isinstance(cfg, SimpleNamespace):
        return cfg
    if hasattr(cfg, "items"):
        return SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))
    return cfg


def build_gr00t_core_model(model_cfg: DictConfig, torch_dtype=torch.bfloat16):
    """Build ``GR00T_N1_5_ForRLActionPrediction`` from actor.model config."""
    from gr00t.experiment.data_config import load_data_config

    from joyrlinf.models.gr00t.action_model import GR00T_N1_5_ForRLActionPrediction
    from joyrlinf.models.gr00t.utils import replace_dropout_with_identity

    apply_embodiment_tag_patches()

    gr00t_cfg = model_cfg.gr00t if hasattr(model_cfg, "gr00t") else model_cfg
    embodiment_tag = gr00t_cfg.get("embodiment_tag", "maniskill_widowx")
    if embodiment_tag in ("libero_franka", "isaaclab_franka"):
        data_config = load_data_config(
            "joyrlinf.models.gr00t.configs:LiberoFrankaDataConfig"
        )
    elif embodiment_tag == "maniskill_widowx":
        data_config = load_data_config(
            "joyrlinf.models.gr00t.configs:ManiskillWidowXDataConfig"
        )
    elif embodiment_tag == "maniskill_panda":
        data_config = load_data_config(
            "joyrlinf.models.gr00t.configs:ManiskillPandaDataConfig"
        )
    else:
        raise ValueError(f"Invalid embodiment_tag: {embodiment_tag}")

    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    model_path = Path(model_cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"GR00T model_path does not exist: {model_path}")

    rl_head_cfg = _to_rl_head_config(
        gr00t_cfg.get("rl_head_config", model_cfg.get("rl_head_config", {}))
    )

    rl_action_dim = int(model_cfg.get("action_dim", 7))
    model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=int(gr00t_cfg.get("denoising_steps", model_cfg.num_steps)),
        output_action_chunks=int(model_cfg.num_action_chunks),
        rl_action_dim=rl_action_dim,
        obs_converter_type=gr00t_cfg.get("obs_converter_type", "maniskill"),
        tune_visual=gr00t_cfg.get("tune_visual", False),
        tune_llm=gr00t_cfg.get("tune_llm", False),
        rl_head_config=rl_head_cfg,
    )
    model.rl_action_dim = rl_action_dim
    model.action_head.rl_action_dim = rl_action_dim
    model.reuse_rollout_backbone_features = bool(
        gr00t_cfg.get("reuse_rollout_backbone_features", False)
    )
    model.to(torch_dtype)

    if getattr(rl_head_cfg, "add_value_head", False):
        model.action_head.value_head._init_weights()

    if getattr(rl_head_cfg, "disable_dropout", True):
        replace_dropout_with_identity(model)

    return model


from joyrlinf.models.gr00t.model import Gr00tModel

__all__ = ["Gr00tModel", "build_gr00t_core_model", "apply_embodiment_tag_patches"]
