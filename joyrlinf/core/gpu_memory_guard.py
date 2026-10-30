"""GPU memory guard: startup budget + runtime monitoring.

Design principles:
  - Prevention over reaction: set hard limits at init, never call empty_cache()
    during hot loops (it nukes cuBLAS workspace → autotune cold-start penalty).
  - Two-zone budget: PyTorch caching allocator gets a capped fraction;
    the remainder is left for PhysX/SAPIEN GPU simulation buffers + CUDA driver.
  - Runtime monitoring via torch.cuda.mem_get_info() is cheap (~1 μs, no sync)
    and never triggers allocator side-effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from joyrlinf.utils.logging_setup import log_at_global_leader

logger = logging.getLogger(__name__)

_MiB = 1 << 20


@dataclass
class MemoryBudget:
    """Describes how to split GPU VRAM between PyTorch and PhysX.

    All sizes in MiB.  Set via config yaml::

        memory:
          pytorch_fraction: 0.45       # hard cap for PyTorch allocator
          env_reserved_mib: 4096       # expected PhysX peak (advisory)
          safety_margin_mib: 512       # free VRAM below this → warning
          critical_margin_mib: 256     # free VRAM below this → abort
          gc_threshold: 0.8            # PYTORCH_CUDA_ALLOC_CONF gc threshold
    """
    pytorch_fraction: float = 0.0
    env_reserved_mib: float = 0.0
    safety_margin_mib: float = 512.0
    critical_margin_mib: float = 256.0
    gc_threshold: float = 0.8

    def __post_init__(self):
        if self.pytorch_fraction < 0 or self.pytorch_fraction > 1:
            raise ValueError(f"pytorch_fraction must be in [0, 1], got {self.pytorch_fraction}")


@dataclass
class MemorySnapshot:
    """Point-in-time GPU memory state (all in MiB)."""
    total_mib: float
    pt_allocated_mib: float
    pt_reserved_mib: float
    non_pt_used_mib: float
    free_mib: float

    @property
    def pt_fragmented_mib(self) -> float:
        return self.pt_reserved_mib - self.pt_allocated_mib

    def __str__(self) -> str:
        return (
            f"GPU mem: total={self.total_mib:.0f}M "
            f"pt_alloc={self.pt_allocated_mib:.0f}M "
            f"pt_rsv={self.pt_reserved_mib:.0f}M "
            f"frag={self.pt_fragmented_mib:.0f}M "
            f"non_pt={self.non_pt_used_mib:.0f}M "
            f"free={self.free_mib:.0f}M"
        )

    def format_share(self) -> str:
        """MiB + % of device total for the main VRAM buckets.

        Buckets are approximately disjoint over the full device::
            pt_alloc + frag + non_pt + free ≈ total
        where ``frag = pt_rsv - pt_alloc`` (caching-allocator idle reserve).
        Under MPS, ``non_pt`` also includes the peer client's CUDA usage.
        """
        total = max(float(self.total_mib), 1e-6)

        def _pct(mib: float) -> float:
            return 100.0 * float(mib) / total

        return (
            f"VRAM share: "
            f"pt_alloc={self.pt_allocated_mib:.0f}MiB({_pct(self.pt_allocated_mib):.1f}%) "
            f"frag={self.pt_fragmented_mib:.0f}MiB({_pct(self.pt_fragmented_mib):.1f}%) "
            f"non_pt={self.non_pt_used_mib:.0f}MiB({_pct(self.non_pt_used_mib):.1f}%) "
            f"free={self.free_mib:.0f}MiB({_pct(self.free_mib):.1f}%) "
            f"| pt_rsv={self.pt_reserved_mib:.0f}MiB({_pct(self.pt_reserved_mib):.1f}%) "
            f"total={self.total_mib:.0f}MiB"
        )


def take_memory_snapshot(device: int = 0) -> MemorySnapshot:
    """Process-local PyTorch stats + device-wide free (MPS-aware non_pt)."""
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    pt_alloc = torch.cuda.memory_allocated(device)
    pt_rsv = torch.cuda.memory_reserved(device)
    non_pt = total_bytes - free_bytes - pt_rsv
    return MemorySnapshot(
        total_mib=total_bytes / _MiB,
        pt_allocated_mib=pt_alloc / _MiB,
        pt_reserved_mib=pt_rsv / _MiB,
        non_pt_used_mib=max(non_pt, 0) / _MiB,
        free_mib=free_bytes / _MiB,
    )


class GPUMemoryGuard:
    """GPU memory budget enforcer and monitor.

    Usage::

        guard = GPUMemoryGuard(budget, device=0)
        guard.apply_pytorch_limit()   # once at startup

        # in rollout loop — cheap, no side-effects
        snap = guard.snapshot()
        guard.check(snap)             # logs warning/error if thresholds crossed
    """

    def __init__(self, budget: MemoryBudget, device: int = 0):
        self.budget = budget
        self.device = device
        self._warned = False
        self._critical_count = 0
        self._step_idx = 0
        self.critical_event_count = 0
        self.safety_warning_count = 0

        props = torch.cuda.get_device_properties(device)
        self.total_mib = props.total_memory / _MiB
        log_at_global_leader(
            logger, logging.INFO,
            "GPUMemoryGuard: device=%d total=%.0fMiB budget=%s",
            device, self.total_mib, budget,
        )

    def apply_pytorch_limit(self) -> None:
        """Set PyTorch allocator hard cap. Call once before any allocation."""
        frac = self.budget.pytorch_fraction
        if frac <= 0 or frac >= 1.0:
            log_at_global_leader(
                logger, logging.INFO,
                "GPUMemoryGuard: pytorch_fraction=%.2f, skipping limit", frac,
            )
            return

        cap_mib = self.total_mib * frac
        torch.cuda.set_per_process_memory_fraction(frac, self.device)
        log_at_global_leader(
            logger, logging.INFO,
            "GPUMemoryGuard: PyTorch allocator capped at %.1f%% = %.0f MiB "
            "(leaves ~%.0f MiB for PhysX + driver)",
            frac * 100, cap_mib, self.total_mib - cap_mib,
        )

    def snapshot(self) -> MemorySnapshot:
        """Capture current memory state.  ~1 μs, no cudaDeviceSynchronize."""
        return take_memory_snapshot(self.device)

    def log_share(self, label: str = "profile", *, level: int = logging.INFO) -> MemorySnapshot:
        """Snapshot and log MiB + %-of-total breakdown for each VRAM bucket."""
        snap = self.snapshot()
        logger.log(level, "[%s] %s", label, snap.format_share())
        return snap

    def check(self, snap: Optional[MemorySnapshot] = None, label: str = "") -> bool:
        """Check memory against thresholds. Returns True if safe.

        Never calls empty_cache() or any allocator-mutating API.
        """
        if snap is None:
            snap = self.snapshot()

        self._step_idx += 1
        prefix = f"[step {self._step_idx}]" if not label else f"[{label}]"

        if snap.free_mib < self.budget.critical_margin_mib:
            self._critical_count += 1
            self.critical_event_count += 1
            logger.error(
                "%s CRITICAL memory pressure! %s "
                "(critical_margin=%.0fM, consecutive=%d)",
                prefix, snap, self.budget.critical_margin_mib,
                self._critical_count,
            )
            return False

        if snap.free_mib < self.budget.safety_margin_mib:
            if not self._warned:
                self.safety_warning_count += 1
                logger.warning(
                    "%s Memory pressure warning. %s "
                    "(safety_margin=%.0fM)",
                    prefix, snap, self.budget.safety_margin_mib,
                )
                self._warned = True
            self._critical_count = 0
            return True

        self._warned = False
        self._critical_count = 0
        return True

    def log_profile(
        self, label: str = "profile", *, level: int = logging.DEBUG
    ) -> MemorySnapshot:
        """Snapshot + log at ``level`` (default DEBUG). Use INFO for startup checks."""
        snap = self.snapshot()
        logger.log(level, "[%s] %s", label, snap)
        return snap


def build_alloc_conf(
    gc_threshold: float = 0.8,
    max_split_size_mb: int = 128,
    expandable_segments: bool = True,
) -> str:
    """Build PYTORCH_CUDA_ALLOC_CONF value.

    garbage_collection_threshold triggers the allocator's internal GC
    (coalesce free blocks, release stale segments) when reserved memory
    reaches this fraction of the cap — WITHOUT freeing cuBLAS/cuDNN
    workspace blocks that are still plan-cached.

    This is fundamentally different from empty_cache():
      - empty_cache(): cudaFree ALL unreferenced blocks → workspace gone
        → next forward re-autotunes from scratch (micro_step 1-2 very slow)
      - GC threshold: merge adjacent free blocks, release segments that
        are entirely free, keep live workspace → no autotune penalty
    """
    parts = []
    if expandable_segments:
        # Never combine with max_split_size_mb: oversize (>max_split) blocks
        # inside expandable segments are not reused, so reserved memory grows
        # unboundedly across identical forward passes.
        parts.append("expandable_segments:True")
    else:
        parts.append(f"max_split_size_mb:{max_split_size_mb}")
    if 0 < gc_threshold < 1:
        parts.append(f"garbage_collection_threshold:{gc_threshold}")
    return ",".join(parts)


def apply_memory_budget_from_cfg(cfg, device: int = 0) -> Optional[GPUMemoryGuard]:
    """Create and apply GPUMemoryGuard from OmegaConf config.

    Expected config path: cfg.memory (optional).  If absent, returns None
    and no limits are applied (backward compatible).
    """
    mem_cfg = cfg.get("memory")
    if mem_cfg is None:
        logger.debug("No memory config found, GPUMemoryGuard disabled")
        return None

    budget = MemoryBudget(
        pytorch_fraction=mem_cfg.get("pytorch_fraction", 0.0),
        env_reserved_mib=mem_cfg.get("env_reserved_mib", 0.0),
        safety_margin_mib=mem_cfg.get("safety_margin_mib", 512.0),
        critical_margin_mib=mem_cfg.get("critical_margin_mib", 256.0),
        gc_threshold=mem_cfg.get("gc_threshold", 0.8),
    )

    guard = GPUMemoryGuard(budget, device=device)
    guard.apply_pytorch_limit()
    return guard


def _resolve_cuda_device_index(
    device: int | torch.device | None = None,
) -> int | None:
    if not torch.cuda.is_available():
        return None
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return None
        return device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.current_device()


def reset_cuda_peak_stats(device: int | torch.device | None = None) -> None:
    """Reset PyTorch per-process peak counters for the next measurement window."""
    idx = _resolve_cuda_device_index(device)
    if idx is None:
        return
    torch.cuda.reset_peak_memory_stats(idx)


def read_cuda_peak_metrics(
    device: int | torch.device | None = None,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Read per-process CUDA peaks + instantaneous alloc, and device-wide free.

    Under NVIDIA MPS each process has its own allocator stats, so peaks are
    **process-local**. Device ``free_mib`` / ``total_mib`` are card-wide.

    Keys (with optional ``prefix``, e.g. ``\"rollout_\"`` / ``\"actor_\"``)::

        {prefix}pt_peak_alloc_mib
        {prefix}pt_peak_reserved_mib
        {prefix}pt_alloc_mib
        {prefix}pt_reserved_mib
        free_mib          # only when prefix is empty
        total_mib         # only when prefix is empty
        {prefix}free_mib  # when prefix is set (same device-wide free)
        {prefix}total_mib
    """
    idx = _resolve_cuda_device_index(device)
    if idx is None:
        return {}

    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
    peak_alloc = torch.cuda.max_memory_allocated(idx)
    peak_rsv = torch.cuda.max_memory_reserved(idx)
    alloc = torch.cuda.memory_allocated(idx)
    rsv = torch.cuda.memory_reserved(idx)
    free_mib = free_bytes / _MiB
    total_mib = total_bytes / _MiB

    out = {
        f"{prefix}pt_peak_alloc_mib": peak_alloc / _MiB,
        f"{prefix}pt_peak_reserved_mib": peak_rsv / _MiB,
        f"{prefix}pt_alloc_mib": alloc / _MiB,
        f"{prefix}pt_reserved_mib": rsv / _MiB,
    }
    if prefix:
        out[f"{prefix}free_mib"] = free_mib
        out[f"{prefix}total_mib"] = total_mib
    else:
        out["free_mib"] = free_mib
        out["total_mib"] = total_mib
    return out
