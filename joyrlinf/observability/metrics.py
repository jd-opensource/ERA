"""Structured per-step metrics (P0 observability contract)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from joyrlinf.utils.runner_utils import prefix_metrics


# Canonical TensorBoard prefixes.
BUCKETS = ("train", "adv", "rollout", "eval", "time", "memory", "data")

# GRPO / rollout data-quality diagnostics (P1 → ``data/`` bucket).
DATA_METRIC_NAMES = frozenset({
    "reward_nonzero_pct",
    "score_nonzero_pct",
    "groups_with_var_pct",
    "score_max",
    "loss_mask_frac",
    "reward_filter_groups_kept_pct",
    "reward_filter_groups_dropped_pct",
    "reward_filter_episodes_kept_pct",
})


@dataclass
class StepMetrics:
    """One global training step worth of scalar metrics."""

    step: int
    train: dict[str, float] = field(default_factory=dict)
    adv: dict[str, float] = field(default_factory=dict)
    rollout: dict[str, float] = field(default_factory=dict)
    eval: dict[str, float] = field(default_factory=dict)
    time: dict[str, float] = field(default_factory=dict)
    memory: dict[str, float] = field(default_factory=dict)
    data: dict[str, float] = field(default_factory=dict)

    def flatten(self) -> dict[str, float]:
        """Flatten to TensorBoard keys ``{bucket}/{name}``."""
        out: dict[str, float] = {}
        for bucket in BUCKETS:
            bucket_map = getattr(self, bucket)
            out.update(prefix_metrics(bucket_map, bucket))
        return out

    def merge_flat(self, flat: dict[str, float]) -> None:
        """Merge prefixed or legacy flat keys into buckets."""
        for key, val in flat.items():
            self._assign(key, float(val))

    def _assign(self, key: str, val: float) -> None:
        key = _normalize_metric_key(key)
        if "/" in key:
            bucket, name = key.split("/", 1)
            if bucket in BUCKETS:
                getattr(self, bucket)[name] = val
                return
        # Unprefixed keys default to rollout (success_rate) or data.
        if key in ("success_rate", "success_once_count"):
            self.rollout[key] = val
        elif key.startswith("reward_filter_"):
            self.data[key] = val
        else:
            self.rollout[key] = val


def _normalize_metric_key(key: str) -> str:
    """Map legacy names to canonical bucket prefixes."""
    if key == "train/success_rate":
        return "rollout/success_rate"
    if key.startswith("eval/"):
        return key
    if key.startswith("time/"):
        return key
    if key.startswith("memory/"):
        return key
    return key


def strip_runner_metrics(batch: dict) -> dict[str, float]:
    """Remove ``_runner_metric_*`` tensors from *batch*; return scalar dict."""
    if not batch:
        return {}
    out: dict[str, float] = {}
    for key in list(batch.keys()):
        if key.startswith("_runner_metric_"):
            name = key[len("_runner_metric_") :].replace("__", "/")
            out[_normalize_metric_key(name)] = float(
                batch.pop(key).reshape(-1)[0].item()
            )
    legacy = batch.pop("_runner_metrics", None)
    if isinstance(legacy, dict):
        for k, v in legacy.items():
            out[_normalize_metric_key(str(k))] = float(v)
    return out


def split_adv_data_metrics(metrics: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """Split advantage vs data-quality diagnostics (P1)."""
    adv: dict[str, float] = {}
    data: dict[str, float] = {}
    for key, val in metrics.items():
        name = key.split("/", 1)[-1] if "/" in key else key
        if name in DATA_METRIC_NAMES or name.startswith("reward_filter_"):
            data[name] = float(val)
        else:
            adv[name] = float(val)
    return adv, data


def partition_stripped_metrics(stripped: dict[str, float]) -> dict[str, dict[str, float]]:
    """Split stripped rollout-batch metrics into bucket dicts."""
    buckets: dict[str, dict[str, float]] = {b: {} for b in BUCKETS}
    for key, val in stripped.items():
        key = _normalize_metric_key(key)
        if "/" in key:
            bucket, name = key.split("/", 1)
            if bucket in buckets:
                buckets[bucket][name] = val
                continue
        buckets["rollout"][key] = val
    return buckets


def memory_snapshot_to_metrics(snap: Any, *, guard: Any | None = None) -> dict[str, float]:
    """Convert ``MemorySnapshot`` (+ optional guard counters) to memory/* scalars."""
    out = {
        "free_mib": float(snap.free_mib),
        "pt_allocated_mib": float(snap.pt_allocated_mib),
        "pt_reserved_mib": float(snap.pt_reserved_mib),
        "pt_fragmented_mib": float(snap.pt_fragmented_mib),
        "non_pt_used_mib": float(snap.non_pt_used_mib),
        "total_mib": float(snap.total_mib),
    }
    if guard is not None:
        out["critical_events"] = float(getattr(guard, "critical_event_count", 0))
        out["safety_warnings"] = float(getattr(guard, "safety_warning_count", 0))
    return out


def coerce_step_metrics(step: int, **buckets: dict[str, float]) -> StepMetrics:
    m = StepMetrics(step=step)
    for bucket in BUCKETS:
        if bucket in buckets and buckets[bucket]:
            getattr(m, bucket).update(buckets[bucket])
    return m
