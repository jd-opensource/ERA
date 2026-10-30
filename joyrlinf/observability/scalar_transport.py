"""Gloo/CPU point-to-point transport for rollout scalar metrics (P1).

Replaces packing scalars as ``_runner_metric_*`` tensors on rollout batches.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Fixed schema: (bucket, name) → flattened as ``bucket/name`` when unpacking.
ROLLOUT_SCALAR_SCHEMA: tuple[tuple[str, str], ...] = (
    ("rollout", "success_rate"),
    ("eval", "success_rate"),
    ("eval", "mean_return"),
    ("time", "rollout_weight_load"),
    ("time", "rollout_core"),
    ("time", "rollout_eval"),
    ("time", "rollout_total"),
    ("time", "rollout_send"),
    # Per-process PyTorch peaks on the EnvRollout MPS client (same physical GPU
    # as the paired Actor). Device free/total are card-wide.
    ("memory", "rollout_pt_peak_alloc_mib"),
    ("memory", "rollout_pt_peak_reserved_mib"),
    ("memory", "rollout_pt_alloc_mib"),
    ("memory", "rollout_pt_reserved_mib"),
    ("memory", "rollout_free_mib"),
    ("memory", "rollout_total_mib"),
)

METRICS_SCALAR_TAG = 77123

# Legacy alias accepted when packing.
_PACK_ALIASES: dict[str, tuple[str, ...]] = {
    "rollout/success_rate": ("train/success_rate", "success_rate"),
}


def _canonical_key(bucket: str, name: str) -> str:
    return f"{bucket}/{name}"


def _metric_values(metrics: dict[str, float]) -> list[float]:
    flat: dict[str, float] = {str(k): float(v) for k, v in metrics.items()}
    if "rollout/success_rate" not in flat and "success_rate" in flat:
        flat["rollout/success_rate"] = flat["success_rate"]

    values: list[float] = []
    for bucket, name in ROLLOUT_SCALAR_SCHEMA:
        key = _canonical_key(bucket, name)
        val = flat.get(key)
        if val is None:
            for alt in _PACK_ALIASES.get(key, ()):
                if alt in flat:
                    val = flat[alt]
                    break
        values.append(float(val) if val is not None else 0.0)
    return values


def pack_rollout_scalars(metrics: dict[str, float]):
    """Pack metrics dict into a CPU float32 vector following ``ROLLOUT_SCALAR_SCHEMA``."""
    values = _metric_values(metrics)
    try:
        import torch
    except ImportError:
        return values
    return torch.tensor(values, dtype=torch.float32, device="cpu")


def unpack_rollout_scalars(payload) -> dict[str, float]:
    """Unpack CPU float32 vector to prefixed metric dict."""
    if hasattr(payload, "detach"):
        vals = payload.detach().cpu().tolist()
    else:
        vals = list(payload)
    out: dict[str, float] = {}
    for i, (bucket, name) in enumerate(ROLLOUT_SCALAR_SCHEMA):
        if i < len(vals):
            out[_canonical_key(bucket, name)] = float(vals[i])
    return out


def send_rollout_scalars(
    dst_rank: int,
    metrics: dict[str, float],
    *,
    group=None,
    tag: int = METRICS_SCALAR_TAG,
) -> None:
    """Send packed rollout scalars from the current rank to *dst_rank*.

    *group* must be a **gloo** process group (e.g. ``shutdown``); the default
  WORLD group is often NCCL and cannot move CPU tensors via point-to-point ops.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return
    if group is None:
        raise ValueError(
            "send_rollout_scalars requires a gloo process group; "
            "pass groups['shutdown'] from the runner."
        )
    payload = pack_rollout_scalars(metrics)
    dist.send(payload, dst=dst_rank, tag=tag, group=group)


def recv_rollout_scalars(
    src_rank: int,
    *,
    group=None,
    tag: int = METRICS_SCALAR_TAG,
) -> dict[str, float]:
    """Receive packed rollout scalars from *src_rank* on the current rank."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return {}
    if group is None:
        raise ValueError(
            "recv_rollout_scalars requires a gloo process group; "
            "pass groups['shutdown'] from the runner."
        )
    payload = torch.zeros(len(ROLLOUT_SCALAR_SCHEMA), dtype=torch.float32, device="cpu")
    dist.recv(payload, src=src_rank, tag=tag, group=group)
    return unpack_rollout_scalars(payload)
