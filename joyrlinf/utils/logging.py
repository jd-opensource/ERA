"""Logging utilities: TensorBoard, WandB, console."""
from __future__ import annotations
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)


def coerce_metric_scalar(value) -> float:
    """Normalize metric values for logging without rounding."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item):
        return float(item())
    raise TypeError(f"Unsupported metric scalar type: {type(value)!r}")


def format_metric_value(value: float, *, key: str = "") -> str:
    """Format one scalar with enough digits for early-training losses."""
    if not math.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0"
    if key.endswith("_pct"):
        return f"{value:.4f}"
    abs_v = abs(value)
    if abs_v >= 1000:
        return f"{value:.2f}"
    if abs_v >= 1:
        return f"{value:.6f}"
    if abs_v >= 1e-8:
        return f"{value:.8f}"
    return f"{value:.3e}"


def format_metrics_compact(
    metrics: dict,
    keys: list[str] | None = None,
) -> str:
    """Format a metrics dict as ``k=v`` pairs for step summaries."""
    if keys is None:
        keys = sorted(metrics.keys())
    parts: list[str] = []
    for k in keys:
        if k not in metrics:
            continue
        v = metrics[k]
        if isinstance(v, float):
            parts.append(f"{k}={format_metric_value(v, key=k)}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts) if parts else "(empty)"


class TrainingLogger:
    def __init__(
        self,
        cfg,
        rank: int = 0,
        output_dir: str | None = None,
        *,
        experiment_cfg: Any | None = None,
        config_path: str | None = None,
    ):
        self.rank = rank
        self.log_path = cfg.get("log_path", "./results")
        self.project = cfg.get("project_name", "joyrlinf")
        self.experiment = cfg.get("experiment_name", "default")
        self.output_dir = output_dir or os.path.join(self.log_path, self.experiment)
        self.backends = cfg.get("logger_backends", ["tensorboard"])
        self.histogram_interval = int(cfg.get("histogram_interval", 0))
        self.tb_writer = None
        self.wandb_run = None
        self._experiment_cfg = experiment_cfg
        self._config_path = config_path
        if rank == 0:
            self._init_backends()

    def _init_backends(self):
        if "tensorboard" in self.backends:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(self.output_dir, "tb")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_dir)
            logger.info("TensorBoard logging to %s", tb_dir)
        if "wandb" in self.backends:
            try:
                import wandb
                from omegaconf import OmegaConf

                init_kwargs: dict[str, Any] = {
                    "project": self.project,
                    "name": self.experiment,
                    "dir": self.output_dir,
                }
                if self._config_path and os.path.isfile(self._config_path):
                    init_kwargs["config"] = OmegaConf.load(self._config_path)
                elif self._experiment_cfg is not None:
                    init_kwargs["config"] = OmegaConf.to_container(
                        self._experiment_cfg, resolve=True,
                    )
                self.wandb_run = wandb.init(**init_kwargs)
                for prefix in (
                    "train", "adv", "rollout", "eval", "time", "memory", "data",
                ):
                    wandb.define_metric(f"{prefix}/*", step_metric="global_step")
            except ImportError:
                logger.warning("wandb not installed")

    def log(self, metrics: dict, step: int):
        if self.rank != 0:
            return
        normalized = {}
        for k, v in metrics.items():
            if isinstance(v, (bool, int, float)) or hasattr(v, "item"):
                try:
                    normalized[k] = coerce_metric_scalar(v)
                except TypeError:
                    continue
            else:
                continue
        if self.tb_writer:
            for k, v in normalized.items():
                self.tb_writer.add_scalar(k, v, step)
        if self.wandb_run:
            import wandb
            payload = dict(normalized)
            payload["global_step"] = step
            wandb.log(payload, step=step)

    def log_histograms(self, histograms: dict[str, list[float]], step: int) -> None:
        if self.rank != 0 or not histograms:
            return
        for name, values in histograms.items():
            if not values:
                continue
            if self.tb_writer:
                self.tb_writer.add_histogram(name, values, step)
            if self.wandb_run:
                import wandb
                wandb.log({name: wandb.Histogram(values)}, step=step)

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()
