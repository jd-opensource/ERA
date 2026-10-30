"""Core data structures for embodied RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch.profiler import record_function


def _move_dict_to_device(d: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    out = {}
    target = torch.device(device)
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            # Already on the target device (incl. pinned CPU): keep reference so
            # we do not drop is_pinned() via a redundant .to("cpu").
            if v.device.type == target.type and (
                target.type == "cpu" or v.device.index == target.index
            ):
                out[k] = v
            else:
                out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = _move_dict_to_device(v, device)
        else:
            out[k] = v
    return out


def _d2h_tensor(t: torch.Tensor, *, pin: bool = True) -> torch.Tensor:
    """Copy one tensor to CPU (non_blocking from CUDA into pinned host if pin)."""
    t = t.detach()
    if t.is_cuda:
        if pin:
            # Split alloc vs copy: cudaHostAlloc often device-syncs; copy_ can overlap.
            with record_function("d2h/pin_alloc"):
                host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            with record_function("d2h/pin_copy"):
                host.copy_(t, non_blocking=True)
            return host
        with record_function("d2h/pageable_to_cpu"):
            return t.to("cpu", non_blocking=True)
    if pin and t.device.type == "cpu" and t.numel() > 0 and not t.is_pinned():
        try:
            with record_function("d2h/pin_memory"):
                return t.pin_memory()
        except RuntimeError:
            return t
    return t


class PrefixKvHostPool:
    """Host buffers for per-chunk prefix KV D2H.

    Allocate once for a ``(capacity, *chunk_shape)`` layout and reuse across
    rollout steps; each chunk only ``copy_`` into ``buf[t]`` so the hot path
    avoids ``cudaHostAlloc`` device syncs.

    SharedHost (POSIX shm) slots are pageable, so CUDA ``non_blocking`` D2H
    would otherwise sync on the host. When ``pin=True`` with an external shm
    binding, a **one-row pinned staging** buffer is kept: async D2H lands in
    staging, then ``commit_staged_row`` does a host ``copy_`` into ``shm[row]``.
    """

    def __init__(self) -> None:
        self.capacity: int = 0
        self.prefix_kv: torch.Tensor | None = None
        self.prefix_pad_masks: torch.Tensor | None = None
        self._kv_meta: tuple[tuple[int, ...], torch.dtype] | None = None
        self._pad_meta: tuple[tuple[int, ...], torch.dtype] | None = None
        # When set, ``ensure`` allocates/binds via SharedHostKvChannel instead
        # of private pinned storage (cross-process KV without Gloo clone).
        self._shm_channel: Any | None = None
        self._external: bool = False
        # One-chunk pinned staging for async D2H → pageable shm (scheme 2).
        self._staging_kv: torch.Tensor | None = None
        self._staging_pad: torch.Tensor | None = None
        self._staged_row: int | None = None
        self._staged_keys: tuple[str, ...] = ()
        self._last_schedule_keys: tuple[str, ...] = ()

    def set_shm_channel(self, channel: Any | None) -> None:
        self._shm_channel = channel
        if channel is None:
            self._clear_pin_staging()

    def bind_external(
        self,
        capacity: int,
        *,
        prefix_kv: torch.Tensor,
        prefix_pad_masks: torch.Tensor | None = None,
        kv_chunk_shape: tuple[int, ...] | None = None,
        kv_dtype: torch.dtype | None = None,
        pad_chunk_shape: tuple[int, ...] | None = None,
        pad_dtype: torch.dtype | None = None,
    ) -> None:
        """Use caller-owned CPU storage (e.g. SharedHost write slot)."""
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if prefix_kv.shape[0] < capacity:
            raise ValueError(
                f"prefix_kv rows {prefix_kv.shape[0]} < capacity {capacity}"
            )
        self.capacity = capacity
        self.prefix_kv = prefix_kv
        self.prefix_pad_masks = prefix_pad_masks
        chunk = kv_chunk_shape or tuple(prefix_kv.shape[1:])
        self._kv_meta = (chunk, kv_dtype or prefix_kv.dtype)
        if prefix_pad_masks is not None:
            pchunk = pad_chunk_shape or tuple(prefix_pad_masks.shape[1:])
            self._pad_meta = (pchunk, pad_dtype or prefix_pad_masks.dtype)
        else:
            self._pad_meta = None
        self._external = True

    @staticmethod
    def _alloc_rows(
        capacity: int,
        chunk_shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        pin: bool,
    ) -> torch.Tensor:
        with record_function("d2h/pin_alloc"):
            return torch.empty(
                (capacity,) + chunk_shape,
                dtype=dtype,
                device="cpu",
                pin_memory=pin,
            )

    def _clear_pin_staging(self) -> None:
        self._staging_kv = None
        self._staging_pad = None
        self._staged_row = None
        self._staged_keys = ()
        self._last_schedule_keys = ()

    @staticmethod
    def _need_staging_realloc(
        buf: torch.Tensor | None, sample: torch.Tensor,
    ) -> bool:
        if buf is None:
            return True
        if tuple(buf.shape) != tuple(sample.shape) or buf.dtype != sample.dtype:
            return True
        return not buf.is_pinned()

    def _ensure_pin_staging(
        self,
        *,
        kv: torch.Tensor | None,
        pad: torch.Tensor | None,
    ) -> None:
        """Allocate / resize one-chunk pinned staging for SharedHost D2H."""
        if kv is not None and self._need_staging_realloc(self._staging_kv, kv):
            with record_function("d2h/pin_alloc"):
                self._staging_kv = torch.empty(
                    tuple(kv.shape), dtype=kv.dtype, device="cpu", pin_memory=True,
                )
        if pad is not None and self._need_staging_realloc(self._staging_pad, pad):
            with record_function("d2h/pin_alloc"):
                self._staging_pad = torch.empty(
                    tuple(pad.shape), dtype=pad.dtype, device="cpu", pin_memory=True,
                )

    def _row_views(
        self, row: int, keys: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        want = keys if keys is not None else ("prefix_kv", "prefix_pad_masks")
        out: dict[str, torch.Tensor] = {}
        if "prefix_kv" in want and self.prefix_kv is not None:
            out["prefix_kv"] = self.prefix_kv[row]
        if "prefix_pad_masks" in want and self.prefix_pad_masks is not None:
            out["prefix_pad_masks"] = self.prefix_pad_masks[row]
        return out

    def ensure(
        self,
        capacity: int,
        *,
        kv: torch.Tensor | None = None,
        pad: torch.Tensor | None = None,
        pin: bool = True,
    ) -> None:
        """Grow / (re)allocate host rows if capacity or chunk shape changed.

        With SharedHost binding, pool rows stay pageable shm. ``pin=True``
        additionally keeps a one-row pinned staging buffer for async CUDA D2H.
        """
        if capacity < 1:
            raise ValueError(f"PrefixKvHostPool capacity must be >= 1, got {capacity}")

        if self._shm_channel is not None:
            # Allocate/reuse POSIX shm double-buffer; bind current write slot.
            self._shm_channel.ensure(capacity, kv=kv, pad=pad)
            if self._shm_channel.ready:
                self._shm_channel.bind_pool(self)
            if pin:
                self._ensure_pin_staging(kv=kv, pad=pad)
            else:
                self._clear_pin_staging()
            return

        self._clear_pin_staging()

        def _need_realloc(
            buf: torch.Tensor | None,
            meta: tuple[tuple[int, ...], torch.dtype] | None,
            sample: torch.Tensor,
        ) -> bool:
            if buf is None or meta is None:
                return True
            if self.capacity < capacity:
                return True
            if meta != (tuple(sample.shape), sample.dtype):
                return True
            if pin and not buf.is_pinned():
                return True
            return False

        new_cap = max(capacity, self.capacity)
        if kv is not None and _need_realloc(self.prefix_kv, self._kv_meta, kv):
            self.prefix_kv = self._alloc_rows(
                new_cap, tuple(kv.shape), kv.dtype, pin=pin,
            )
            self._kv_meta = (tuple(kv.shape), kv.dtype)
            self.capacity = new_cap
            self._external = False
        if pad is not None and _need_realloc(
            self.prefix_pad_masks, self._pad_meta, pad,
        ):
            self.prefix_pad_masks = self._alloc_rows(
                new_cap, tuple(pad.shape), pad.dtype, pin=pin,
            )
            self._pad_meta = (tuple(pad.shape), pad.dtype)
            self.capacity = new_cap
            self._external = False

    def schedule_row(
        self, row: int, forward_inputs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Enqueue D2H/H2H into staging or ``buf[row]``.

        SharedHost + pinned staging + CUDA src: async D2H into staging only.
        Call ``commit_staged_row(row)`` after the copy stream sync to flush
        staging → ``shm[row]`` and obtain durable row views for offload.

        Private pinned pool / CPU src: writes ``buf[row]`` directly (commit is
        a no-op that re-returns those views).
        """
        if row < 0 or row >= self.capacity:
            raise IndexError(
                f"PrefixKvHostPool row {row} out of range for capacity {self.capacity}"
            )
        self._staged_row = None
        self._staged_keys = ()
        out: dict[str, torch.Tensor] = {}
        scheduled_keys: list[str] = []
        staged_keys: list[str] = []

        kv = forward_inputs.get("prefix_kv")
        if kv is not None:
            if self.prefix_kv is None:
                raise RuntimeError("PrefixKvHostPool.ensure() required before schedule_row")
            src = kv.detach()
            use_staging = (
                self._external
                and self._staging_kv is not None
                and src.is_cuda
            )
            if use_staging:
                with record_function("d2h/pin_copy"):
                    self._staging_kv.copy_(src, non_blocking=True)
                out["prefix_kv"] = self._staging_kv
                staged_keys.append("prefix_kv")
            else:
                dst_pinned = self.prefix_kv.is_pinned()
                with record_function("d2h/pin_copy"):
                    self.prefix_kv[row].copy_(
                        src, non_blocking=src.is_cuda and dst_pinned,
                    )
                out["prefix_kv"] = self.prefix_kv[row]
            scheduled_keys.append("prefix_kv")

        pad = forward_inputs.get("prefix_pad_masks")
        if pad is not None:
            if self.prefix_pad_masks is None:
                raise RuntimeError("PrefixKvHostPool.ensure() required before schedule_row")
            src = pad.detach()
            use_staging = (
                self._external
                and self._staging_pad is not None
                and src.is_cuda
            )
            if use_staging:
                with record_function("d2h/pin_copy"):
                    self._staging_pad.copy_(src, non_blocking=True)
                out["prefix_pad_masks"] = self._staging_pad
                staged_keys.append("prefix_pad_masks")
            else:
                dst_pinned = self.prefix_pad_masks.is_pinned()
                with record_function("d2h/pin_copy"):
                    self.prefix_pad_masks[row].copy_(
                        src, non_blocking=src.is_cuda and dst_pinned,
                    )
                out["prefix_pad_masks"] = self.prefix_pad_masks[row]
            scheduled_keys.append("prefix_pad_masks")

        self._last_schedule_keys = tuple(scheduled_keys)
        if staged_keys:
            self._staged_row = row
            self._staged_keys = tuple(staged_keys)
        return out

    def commit_staged_row(self, row: int) -> dict[str, torch.Tensor]:
        """After copy-stream sync: flush pinned staging into ``shm[row]`` (H2H).

        Safe to call when staging was not used; returns durable ``buf[row]``
        views either way. Must run before buffer offload when staging was used.
        """
        if row < 0 or row >= self.capacity:
            raise IndexError(
                f"PrefixKvHostPool row {row} out of range for capacity {self.capacity}"
            )
        keys = self._last_schedule_keys
        if self._staged_row is None:
            return self._row_views(row, keys)
        if self._staged_row != row:
            raise RuntimeError(
                f"commit_staged_row({row}) but staged row is {self._staged_row}"
            )
        with record_function("d2h/h2h_shm"):
            if "prefix_kv" in self._staged_keys:
                if self.prefix_kv is None or self._staging_kv is None:
                    raise RuntimeError("staging commit missing prefix_kv buffers")
                self.prefix_kv[row].copy_(self._staging_kv)
            if "prefix_pad_masks" in self._staged_keys:
                if self.prefix_pad_masks is None or self._staging_pad is None:
                    raise RuntimeError("staging commit missing prefix_pad_masks buffers")
                self.prefix_pad_masks[row].copy_(self._staging_pad)
        self._staged_row = None
        self._staged_keys = ()
        return self._row_views(row, keys)

    def stacked(self, n_rows: int) -> dict[str, torch.Tensor]:
        """Contiguous ``[:n_rows]`` views for ``EmbodiedRolloutBatch.to_dict``."""
        if n_rows < 0 or n_rows > self.capacity:
            raise IndexError(
                f"PrefixKvHostPool stacked n_rows={n_rows} capacity={self.capacity}"
            )
        out: dict[str, torch.Tensor] = {}
        if self.prefix_kv is not None and n_rows > 0:
            out["prefix_kv"] = self.prefix_kv[:n_rows]
        if self.prefix_pad_masks is not None and n_rows > 0:
            out["prefix_pad_masks"] = self.prefix_pad_masks[:n_rows]
        return out


def async_d2h_tensor_tree(obj: Any, *, pin: bool = True) -> Any:
    """Recursively D2H a nested Tensor/dict/list/tuple tree to (pinned) CPU.

    CUDA tensors use non_blocking copies; call under a dedicated CUDA stream
    and synchronize before reading the returned tensors. Already-CPU tensors
    are detached (and optionally pinned).
    """
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return _d2h_tensor(obj, pin=pin)
    if isinstance(obj, dict):
        return {k: async_d2h_tensor_tree(v, pin=pin) for k, v in obj.items()}
    if isinstance(obj, list):
        return [async_d2h_tensor_tree(v, pin=pin) for v in obj]
    if isinstance(obj, tuple):
        return tuple(async_d2h_tensor_tree(v, pin=pin) for v in obj)
    return obj


def _normalize_batch_tensor(t: torch.Tensor) -> torch.Tensor:
    """Use float32 for floats so NCCL comm and actor training share one dtype."""
    if t.is_floating_point() and t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return t.float()
    return t


def _stack_tensors_preserve_pin(
    vals: list[torch.Tensor], dim: int = 0,
) -> torch.Tensor:
    """Stack tensors into a contiguous buffer (pageable on CPU)."""
    norms = [_normalize_batch_tensor(v) for v in vals]
    return torch.stack(norms, dim=dim)


def _stack_dict_list(dicts: list[dict[str, Any]], dim: int = 0) -> dict[str, Any]:
    """Stack a list of dicts with tensor values into a single dict."""
    if not dicts:
        return {}
    keys = dicts[0].keys()
    result = {}
    for k in keys:
        vals = [d[k] for d in dicts if d.get(k) is not None]
        if not vals:
            result[k] = None
        elif isinstance(vals[0], torch.Tensor):
            result[k] = _stack_tensors_preserve_pin(vals, dim=dim)
        elif isinstance(vals[0], dict):
            result[k] = _stack_dict_list(vals, dim=dim)
        else:
            result[k] = vals
    return result


@dataclass
class ChunkStepResult:
    """Result from a single chunk step (model prediction + env step)."""
    prev_logprobs: torch.Tensor  # [B, action_chunk, action_dim]
    prev_values: Optional[torch.Tensor] = None  # [B, num_action_chunks]
    dones: Optional[torch.Tensor] = None  # [B] or [B, num_action_chunks]
    terminations: Optional[torch.Tensor] = None  # [B] or [B, num_action_chunks]
    truncations: Optional[torch.Tensor] = None  # [B] or [B, num_action_chunks]
    rewards: Optional[torch.Tensor] = None  # [B] or [B, num_action_chunks]
    # Weight version used for this prediction (async staleness correction, PPO).
    versions: Optional[torch.Tensor] = None  # [B, num_action_chunks]
    forward_inputs: Optional[dict[str, Any]] = None


@dataclass
class EmbodiedRolloutBatch:
    """Accumulated rollout data for one or more epochs, ready for actor training.

    Rewards and prev_logprobs are indexed per chunk step [T, ...]. Dones use
    RLinf layout [T+1, ...]: leading all-False row, then post-chunk done rows
    (truncation marked on the last sub-step of each chunk). GAE may append
    prev_values on the final bootstrap predict without an extra dones row.
    """
    rollout_epochs: int = 1
    prev_logprobs: list[torch.Tensor] = field(default_factory=list)
    prev_values: list[torch.Tensor] = field(default_factory=list)
    dones: list[torch.Tensor] = field(default_factory=list)
    terminations: list[torch.Tensor] = field(default_factory=list)
    truncations: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    versions: list[torch.Tensor] = field(default_factory=list)
    forward_inputs: list[dict[str, Any]] = field(default_factory=list)
    # Optional contiguous host KV written by PrefixKvHostPool (per_chunk path).
    # When set, ``to_dict`` uses these instead of stacking per-chunk views.
    host_kv_fields: dict[str, torch.Tensor] = field(default_factory=dict)

    def append(self, result: ChunkStepResult) -> None:
        if result.prev_logprobs is not None:
            self.prev_logprobs.append(result.prev_logprobs)
        if result.prev_values is not None:
            self.prev_values.append(result.prev_values)
        if result.dones is not None:
            self.dones.append(result.dones)
        if result.terminations is not None:
            self.terminations.append(result.terminations)
        if result.truncations is not None:
            self.truncations.append(result.truncations)
        if result.rewards is not None:
            self.rewards.append(result.rewards)
        if result.versions is not None:
            self.versions.append(result.versions)
        if result.forward_inputs is not None:
            self.forward_inputs.append(result.forward_inputs)

    def to_dict(self, device: str | torch.device = "cpu") -> dict[str, Any]:
        """Flatten into a single dict of stacked tensors for actor consumption."""
        result: dict[str, Any] = {}

        if self.prev_logprobs:
            result["prev_logprobs"] = torch.stack(
                [_normalize_batch_tensor(t) for t in self.prev_logprobs], dim=0
            )
        if self.prev_values:
            result["prev_values"] = torch.stack(
                [_normalize_batch_tensor(t) for t in self.prev_values], dim=0
            )
        if self.dones:
            result["dones"] = torch.stack(self.dones, dim=0)
        if self.terminations:
            result["terminations"] = torch.stack(self.terminations, dim=0)
        if self.truncations:
            result["truncations"] = torch.stack(self.truncations, dim=0)
        if self.rewards:
            result["rewards"] = torch.stack(
                [_normalize_batch_tensor(t) for t in self.rewards], dim=0
            )
        if self.versions:
            result["versions"] = torch.stack(
                [_normalize_batch_tensor(t) for t in self.versions], dim=0
            )
        if self.forward_inputs:
            skip = set(self.host_kv_fields)
            if skip:
                fi_for_stack = [
                    {k: v for k, v in fi.items() if k not in skip}
                    for fi in self.forward_inputs
                ]
                merged = _stack_dict_list(fi_for_stack, dim=0)
                n = len(self.forward_inputs)
                for k, v in self.host_kv_fields.items():
                    if v.shape[0] < n:
                        raise ValueError(
                            f"host_kv_fields[{k}] rows={v.shape[0]} < forward_inputs={n}"
                        )
                    merged[k] = v[:n]
            else:
                merged = _stack_dict_list(self.forward_inputs, dim=0)
            for k, v in merged.items():
                if k not in result:
                    result[k] = v

        # Always move: stack keeps the device of list elements. With GPU-resident
        # chunks and device="cpu", we must D2H; with CPU-resident chunks this is
        # a no-op. Previously skipped when device=="cpu", leaving CUDA stacks.
        result = _move_dict_to_device(result, device)
        return result

    def clear(self) -> None:
        self.prev_logprobs.clear()
        self.prev_values.clear()
        self.dones.clear()
        self.terminations.clear()
        self.truncations.clear()
        self.rewards.clear()
        self.versions.clear()
        self.forward_inputs.clear()
        self.host_kv_fields.clear()
