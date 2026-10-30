"""OpenPI (Pi0/Pi0.5) model wrapper for inference and training.

Thin wrapper that uses rlinf's model initialization flow:
  1. get_openpi_config() -> TrainConfig
  2. OpenPi0Config from model config + user overrides
  3. OpenPi0ForRLActionPrediction(config)
  4. Load safetensors weights
  5. Setup input/output transforms (normalization, etc.)
"""

from __future__ import annotations

import gc
import glob
import logging
import os
from typing import Any, Literal, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from joyrlinf.models.predict_batching import (
    predict_with_micro_batch,
    read_predict_micro_batch_size,
)

logger = logging.getLogger(__name__)


class OpenPiModel:
    """Wraps OpenPI model for both rollout inference and actor training."""

    def __init__(self, model_cfg: DictConfig, device: torch.device, mode: str = "inference"):
        self.cfg = model_cfg
        self.device = device
        self.mode = mode
        self.predict_micro_batch_size = read_predict_micro_batch_size(model_cfg)
        self.model = self._build_model()

    def _build_model(self):
        import safetensors.torch
        import openpi.shared.download as download
        import openpi.transforms as transforms
        from openpi.training import checkpoints as _checkpoints

        from joyrlinf.models.openpi_configs import get_openpi_config
        from joyrlinf.models.openpi_action_model import (
            OpenPi0Config,
            OpenPi0ForRLActionPrediction,
        )

        openpi_cfg = self.cfg.openpi
        config_name = openpi_cfg.config_name
        model_path = self.cfg.model_path
        logger.info(
            "OpenPiModel build start: mode=%s config=%s model_path=%s",
            self.mode, config_name, model_path,
        )
        if not os.path.exists(model_path):
            parent = os.path.dirname(model_path.rstrip("/"))
            parent_msg = f"parent dir missing: {parent}"
            if os.path.isdir(parent):
                siblings = sorted(os.listdir(parent))
                preview = ", ".join(siblings[:8])
                parent_msg = f"parent dir exists: {parent}; first entries: {preview}"
            raise FileNotFoundError(
                f"Configured model_path does not exist: {model_path}. {parent_msg}"
            )

        actor_train_config = get_openpi_config(config_name, model_path=model_path)
        actor_model_config = actor_train_config.model
        actor_model_config = OpenPi0Config(**actor_model_config.__dict__)

        for key, val in openpi_cfg.items():
            if hasattr(actor_model_config, key):
                actor_model_config.__dict__[key] = val

        override_map = {
            "num_action_chunks": "action_chunk",
            "action_dim": "action_env_dim",
            "num_steps": "num_steps",
            "add_value_head": "add_value_head",
        }
        for cfg_key, model_key in override_map.items():
            if self.cfg.get(cfg_key) is not None:
                actor_model_config.__dict__[model_key] = self.cfg[cfg_key]

        if getattr(actor_model_config, "reuse_rollout_prefix_kv", False):
            if not actor_model_config.train_expert_only:
                raise ValueError(
                    "openpi.reuse_rollout_prefix_kv requires "
                    "openpi.train_expert_only=true (frozen VLM); otherwise "
                    "rollout KV goes stale during training."
                )
            if getattr(actor_model_config, "value_after_vlm", False) and getattr(
                actor_model_config, "add_value_head", False
            ):
                raise ValueError(
                    "openpi.reuse_rollout_prefix_kv is incompatible with "
                    "value_after_vlm (needs prefix_output which is not cached)."
                )
            kv_xfer = str(
                getattr(actor_model_config, "prefix_kv_transfer", "microbatch")
            ).strip().lower()
            if kv_xfer not in ("bulk_gpu", "microbatch"):
                raise ValueError(
                    "openpi.prefix_kv_transfer must be 'bulk_gpu' or "
                    f"'microbatch', got {kv_xfer!r}."
                )
            # OpenPi0Config is frozen; mutate via __dict__ like other overrides.
            actor_model_config.__dict__["prefix_kv_transfer"] = kv_xfer
        kv_ipc = str(
            getattr(actor_model_config, "prefix_kv_ipc", "off")
        ).strip().lower()
        if kv_ipc in ("", "none", "false", "0"):
            kv_ipc = "off"
        if kv_ipc not in ("off", "host_shm"):
            raise ValueError(
                "openpi.prefix_kv_ipc must be 'off' or 'host_shm', "
                f"got {kv_ipc!r}."
            )
        if kv_ipc == "host_shm":
            if not getattr(actor_model_config, "reuse_rollout_prefix_kv", False):
                raise ValueError(
                    "openpi.prefix_kv_ipc=host_shm requires "
                    "reuse_rollout_prefix_kv=true"
                )
            d2h = str(
                getattr(actor_model_config, "rollout_buffer_d2h", "end")
            ).strip().lower()
            if d2h != "per_chunk":
                raise ValueError(
                    "openpi.prefix_kv_ipc=host_shm requires "
                    "rollout_buffer_d2h=per_chunk"
                )
        actor_model_config.__dict__["prefix_kv_ipc"] = kv_ipc
        d2h_mode = str(
            getattr(actor_model_config, "rollout_buffer_d2h", "end")
        ).strip().lower()
        if d2h_mode not in ("end", "per_chunk"):
            raise ValueError(
                "openpi.rollout_buffer_d2h must be 'end' or 'per_chunk', "
                f"got {d2h_mode!r}."
            )
        actor_model_config.__dict__["rollout_buffer_d2h"] = d2h_mode
        if getattr(actor_model_config, "expert_only_actor", False) and not getattr(
            actor_model_config, "reuse_rollout_prefix_kv", False
        ):
            raise ValueError(
                "openpi.expert_only_actor requires "
                "openpi.reuse_rollout_prefix_kv=true: without injected KV the "
                "actor must recompute the VLM prefix."
            )
        strip_vlm = (
            self.mode == "training"
            and getattr(actor_model_config, "expert_only_actor", False)
        )

        logger.debug("Resolving checkpoint directory from model_path")
        checkpoint_dir = download.maybe_download(str(model_path))
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        logger.debug(
            "Checkpoint resolved: dir=%s safetensors_count=%d",
            checkpoint_dir, len(weight_paths),
        )

        model = OpenPi0ForRLActionPrediction(actor_model_config)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "Model created: train_expert_only=%s total_params=%.1fM",
            actor_model_config.train_expert_only,
            total_params / 1e6,
        )
        if actor_model_config.train_expert_only:
            model.freeze_vlm()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = total_params - trainable
            logger.info(
                "VLM frozen: trainable=%.1fM  frozen=%.1fM  (saves ~%.1f GB optimizer mem)",
                trainable / 1e6, frozen / 1e6, frozen * 4 * 2 / 1e9,
            )

        if strip_vlm:
            # Expert-only actor: drop the VLM before the weight load so its
            # 2.9B params are never read from disk nor moved to GPU.
            model.strip_vlm_for_expert_only_actor()
            gc.collect()
            kept_params = sum(p.numel() for p in model.parameters())
            logger.info(
                "Expert-only actor: VLM stripped, %.1fM -> %.1fM params "
                "(GPU weight footprint ~%.1f GB smaller)",
                total_params / 1e6,
                kept_params / 1e6,
                (total_params - kept_params) * 2 / 1e9,
            )

        for weight_path in weight_paths:
            safetensors.torch.load_model(model, weight_path, strict=False)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

        data_config = actor_train_config.data.create(
            actor_train_config.assets_dirs, actor_model_config
        )
        norm_stats = None
        if data_config.asset_id is not None:
            norm_stats = _checkpoints.load_norm_stats(
                checkpoint_dir, data_config.asset_id
            )

        repack_transforms = transforms.Group()
        default_prompt = None
        model.setup_wrappers(
            transforms=[
                *repack_transforms.inputs,
                transforms.InjectDefaultPrompt(default_prompt),
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.model_transforms.inputs,
            ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.data_transforms.outputs,
                *repack_transforms.outputs,
            ],
        )

        model = model.to(self.device)
        if self.mode == "inference":
            model.eval()

        logger.info(
            "OpenPI model loaded: config=%s, chunks=%d, steps=%d, "
            "reuse_rollout_prefix_kv=%s expert_only_actor=%s vlm_stripped=%s",
            actor_model_config.config_name,
            actor_model_config.action_chunk,
            actor_model_config.num_steps,
            getattr(actor_model_config, "reuse_rollout_prefix_kv", False),
            getattr(actor_model_config, "expert_only_actor", False),
            getattr(model, "vlm_stripped", False),
        )
        return model

    def preprocess_obs(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        return self.model.obs_processor(env_obs)

    @torch.no_grad()
    def predict(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return predict_with_micro_batch(
            lambda obs: self.model.predict_action_batch(
                obs, mode=mode, compute_values=compute_values,
            ),
            env_obs,
            self.predict_micro_batch_size,
            log_label="openpi",
        )

    def forward(
        self,
        data: dict[str, torch.Tensor],
        compute_values: bool = False,
        compute_entropy: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        return self.model.default_forward(
            data, compute_values=compute_values, **kwargs
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
        """Move model weights to CPU to free GPU for actor phase."""
        self.model = self.model.cpu()
        self.device = torch.device("cpu")

    def onload_to_device(self, device: torch.device) -> None:
        """Restore model to GPU for rollout inference."""
        self.model = self.model.to(device)
        self.device = device
        if self.mode == "inference":
            self.model.eval()

    def set_global_step(self, global_step: int) -> None:
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
