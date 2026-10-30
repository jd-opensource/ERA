"""GR00T N1.5 model wrapper for inference and training (OpenPiModel-aligned API)."""

from __future__ import annotations

import gc
import logging
from typing import Any, Literal

import torch
from omegaconf import DictConfig

from joyrlinf.models.predict_batching import (
    predict_with_micro_batch,
    read_predict_micro_batch_size,
)

logger = logging.getLogger(__name__)


class Gr00tModel:
    """Wraps GR00T N1.5 for rollout inference and actor training."""

    def __init__(self, model_cfg: DictConfig, device: torch.device, mode: str = "inference"):
        self.cfg = model_cfg
        self.device = device
        self.mode = mode
        self.predict_micro_batch_size = read_predict_micro_batch_size(model_cfg)
        self.model = self._build_model()

    def _build_model(self):
        from joyrlinf.models.gr00t import build_gr00t_core_model

        model_path = self.cfg.model_path
        logger.info(
            "Gr00tModel build start: mode=%s model_path=%s embodiment=%s",
            self.mode,
            model_path,
            self.cfg.get("gr00t", {}).get("embodiment_tag", "maniskill_widowx"),
        )

        precision = self.cfg.get("gr00t", {}).get("precision", "bf16")
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(precision), torch.bfloat16)

        model = build_gr00t_core_model(self.cfg, torch_dtype=torch_dtype)
        gr00t_cfg = self.cfg.get("gr00t", {}) or {}
        expert_only_actor = bool(gr00t_cfg.get("expert_only_actor", False))
        if expert_only_actor and not bool(
            gr00t_cfg.get("reuse_rollout_backbone_features", False)
        ):
            raise ValueError(
                "gr00t.expert_only_actor requires "
                "gr00t.reuse_rollout_backbone_features=true"
            )
        total_before_strip = sum(p.numel() for p in model.parameters())
        if self.mode == "training" and expert_only_actor:
            model.strip_backbone_for_expert_only_actor()
            gc.collect()
            total_after_strip = sum(p.numel() for p in model.parameters())
            logger.info(
                "GR00T expert-only actor: backbone stripped, %.1fM -> %.1fM "
                "params (BF16 GPU weights reduced by ~%.1f GiB)",
                total_before_strip / 1e6,
                total_after_strip / 1e6,
                (total_before_strip - total_after_strip) * 2 / 1024**3,
            )
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "GR00T loaded: total_params=%.1fM trainable=%.1fM chunks=%d "
            "steps=%d reuse_backbone=%s expert_only_actor=%s",
            total_params / 1e6,
            trainable / 1e6,
            self.cfg.num_action_chunks,
            self.cfg.get("gr00t", {}).get(
                "denoising_steps", self.cfg.get("num_steps", 4)
            ),
            bool(gr00t_cfg.get("reuse_rollout_backbone_features", False)),
            expert_only_actor,
        )

        if self.mode == "inference":
            model.eval()
        return model

    @torch.no_grad()
    def predict(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        return predict_with_micro_batch(
            lambda obs: self.model.predict_action_batch(
                obs, mode=mode, compute_values=compute_values,
            ),
            env_obs,
            self.predict_micro_batch_size,
            log_label="gr00t",
        )

    def forward(
        self,
        data: dict[str, torch.Tensor],
        compute_values: bool = False,
        compute_entropy: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        return self.model(
            forward_type="default_forward",
            data=data,
            compute_values=compute_values,
            **kwargs,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        self.model.load_state_dict(state_dict, strict=strict)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def offload_to_cpu(self) -> None:
        """Move model weights to CPU to free GPU for sync actor phase."""
        self.model = self.model.cpu()
        self.device = torch.device("cpu")

    def onload_to_device(self, device: torch.device) -> None:
        """Restore model to GPU for rollout inference (sync colocated reuse)."""
        self.model = self.model.to(device)
        self.device = device
        if self.mode == "inference":
            self.model.eval()

    def set_global_step(self, global_step: int) -> None:
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
