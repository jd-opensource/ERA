"""Shared-host double buffer for rollout→actor prefix_kv (same-node MPS).

Avoids Gloo cloning the multi‑GiB ``prefix_kv`` tensor. Rollout D2Hs into the
current write slot (POSIX shm); Actor maps the published slot by name and
reads in place. A second slot lets the next epoch fill while the actor still
trains on the previous publish.

Handshake:

* Rollout ``begin_epoch_write`` → wait until control-shm marks slot FREE
* Rollout ``publish`` → meta piggybacked with the small-tensor Gloo send
* Actor ``view`` → zero-copy CPU tensors from data shm
* Actor ``send_ack`` → set control-shm slot FREE (no Gloo; PG not thread-safe)

Shm data pages are ordinary (pageable) host memory. Per-chunk CUDA D2H uses a
one-row pinned staging buffer (``PrefixKvHostPool``) so ``non_blocking`` copy
can overlap ``env.chunk_step``; after stream sync, staging is H2H-copied into
the current shm write slot.
"""

from __future__ import annotations

import atexit
import logging
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

KV_IPC_KEYS = ("prefix_kv", "prefix_pad_masks")

_SLOT_FREE = 0
_SLOT_WRITING = 1
_SLOT_PUBLISHED = 2


def _dtype_to_code(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _code_to_dtype(code: str) -> torch.dtype:
    if not hasattr(torch, code):
        raise ValueError(f"Unknown dtype code {code!r}")
    dt = getattr(torch, code)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"Not a torch.dtype: {code!r}")
    return dt


def _nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * dtype.itemsize


@dataclass
class SharedHostKvMeta:
    """Serializable publish record (sent via Gloo object list)."""

    slot: int
    n_rows: int
    capacity: int
    kv_name: str
    kv_shape: tuple[int, ...]
    kv_dtype: str
    pad_name: str | None
    pad_shape: tuple[int, ...] | None
    pad_dtype: str | None
    ctrl_name: str
    version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "n_rows": self.n_rows,
            "capacity": self.capacity,
            "kv_name": self.kv_name,
            "kv_shape": list(self.kv_shape),
            "kv_dtype": self.kv_dtype,
            "pad_name": self.pad_name,
            "pad_shape": list(self.pad_shape) if self.pad_shape is not None else None,
            "pad_dtype": self.pad_dtype,
            "ctrl_name": self.ctrl_name,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SharedHostKvMeta":
        pad_shape = d.get("pad_shape")
        return cls(
            slot=int(d["slot"]),
            n_rows=int(d["n_rows"]),
            capacity=int(d["capacity"]),
            kv_name=str(d["kv_name"]),
            kv_shape=tuple(int(x) for x in d["kv_shape"]),
            kv_dtype=str(d["kv_dtype"]),
            pad_name=(str(d["pad_name"]) if d.get("pad_name") is not None else None),
            pad_shape=(
                tuple(int(x) for x in pad_shape) if pad_shape is not None else None
            ),
            pad_dtype=(str(d["pad_dtype"]) if d.get("pad_dtype") is not None else None),
            ctrl_name=str(d["ctrl_name"]),
            version=int(d["version"]),
        )


class _ShmTensor:
    """Own or attach a byte shm region and expose a reshaped torch view."""

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        create: bool,
    ):
        self.name = name
        self.shape = tuple(int(x) for x in shape)
        self.dtype = dtype
        size = _nbytes(self.shape, dtype)
        if create:
            try:
                self._shm = shared_memory.SharedMemory(
                    name=name, create=True, size=size,
                )
            except FileExistsError:
                stale = shared_memory.SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
                self._shm = shared_memory.SharedMemory(
                    name=name, create=True, size=size,
                )
            self._owner = True
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            self._owner = False
            if self._shm.size < size:
                raise RuntimeError(
                    f"shm {name!r} size {self._shm.size} < required {size}"
                )
        flat = torch.frombuffer(
            self._shm.buf, dtype=dtype, count=size // dtype.itemsize,
        )
        self.tensor = flat.reshape(self.shape)

    def close(self) -> None:
        self.tensor = None  # type: ignore[assignment]
        if self._shm is not None:
            self._shm.close()
            if self._owner:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
            self._shm = None


class SharedHostKvChannel:
    """Double-buffered shared-host KV between one EnvRollout↔Actor pair."""

    def __init__(
        self,
        name_prefix: str,
        *,
        n_slots: int = 2,
        is_owner: bool,
        src_rank: int | None = None,
        dst_rank: int | None = None,
        group=None,
    ):
        if n_slots < 2:
            raise ValueError("SharedHostKvChannel requires n_slots >= 2")
        self.name_prefix = name_prefix
        self.n_slots = n_slots
        self.is_owner = is_owner
        # Kept for API symmetry with the runner; ACK no longer uses Gloo.
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.group = group

        self.capacity = 0
        self._kv_chunk_shape: tuple[int, ...] | None = None
        self._kv_dtype: torch.dtype | None = None
        self._pad_chunk_shape: tuple[int, ...] | None = None
        self._pad_dtype: torch.dtype | None = None

        self._kv: list[_ShmTensor | None] = [None] * n_slots
        self._pad: list[_ShmTensor | None] = [None] * n_slots
        self._ctrl_shm: shared_memory.SharedMemory | None = None
        self._ctrl: np.ndarray | None = None  # int32[n_slots]
        self._ctrl_name = f"{name_prefix}_ctrl"
        self._write_idx = 0
        self._version = 0
        self._closed = False

        if is_owner:
            self._create_ctrl()
            atexit.register(self.close)

    def _create_ctrl(self) -> None:
        size = self.n_slots * 4
        try:
            self._ctrl_shm = shared_memory.SharedMemory(
                name=self._ctrl_name, create=True, size=size,
            )
        except FileExistsError:
            stale = shared_memory.SharedMemory(name=self._ctrl_name, create=False)
            stale.close()
            stale.unlink()
            self._ctrl_shm = shared_memory.SharedMemory(
                name=self._ctrl_name, create=True, size=size,
            )
        self._ctrl = np.ndarray(
            (self.n_slots,), dtype=np.int32, buffer=self._ctrl_shm.buf,
        )
        self._ctrl[:] = _SLOT_FREE

    def _attach_ctrl(self, name: str) -> None:
        if self._ctrl is not None and self._ctrl_name == name:
            return
        if self._ctrl_shm is not None:
            self._ctrl_shm.close()
        self._ctrl_name = name
        self._ctrl_shm = shared_memory.SharedMemory(name=name, create=False)
        self._ctrl = np.ndarray(
            (self.n_slots,), dtype=np.int32, buffer=self._ctrl_shm.buf,
        )

    @property
    def ready(self) -> bool:
        return self._kv[0] is not None

    def ensure(
        self,
        capacity: int,
        *,
        kv: torch.Tensor | None = None,
        pad: torch.Tensor | None = None,
    ) -> None:
        """Owner: create shm slots sized for ``(capacity, *chunk_shape)``."""
        if not self.is_owner:
            raise RuntimeError("Actor channel attaches via open_from_meta, not ensure()")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if kv is None and pad is None:
            return

        need = False
        if kv is not None:
            shape, dtype = tuple(kv.shape), kv.dtype
            if (
                self._kv[0] is None
                or self.capacity < capacity
                or self._kv_chunk_shape != shape
                or self._kv_dtype != dtype
            ):
                need = True
        if pad is not None:
            shape, dtype = tuple(pad.shape), pad.dtype
            if (
                self._pad[0] is None
                or self.capacity < capacity
                or self._pad_chunk_shape != shape
                or self._pad_dtype != dtype
            ):
                need = True
        if not need:
            return

        self._release_data_slots()
        new_cap = max(capacity, self.capacity)
        if kv is not None:
            self._kv_chunk_shape = tuple(kv.shape)
            self._kv_dtype = kv.dtype
        if pad is not None:
            self._pad_chunk_shape = tuple(pad.shape)
            self._pad_dtype = pad.dtype
        if self._kv_chunk_shape is None or self._kv_dtype is None:
            raise RuntimeError("SharedHostKvChannel.ensure requires a kv sample")
        self.capacity = new_cap
        kv_full = (new_cap,) + self._kv_chunk_shape
        for i in range(self.n_slots):
            self._kv[i] = _ShmTensor(
                f"{self.name_prefix}_kv{i}",
                kv_full,
                self._kv_dtype,
                create=True,
            )
            if self._pad_chunk_shape is not None and self._pad_dtype is not None:
                pad_full = (new_cap,) + self._pad_chunk_shape
                self._pad[i] = _ShmTensor(
                    f"{self.name_prefix}_pad{i}",
                    pad_full,
                    self._pad_dtype,
                    create=True,
                )
        assert self._ctrl is not None
        self._ctrl[:] = _SLOT_FREE
        self._ctrl[self._write_idx] = _SLOT_WRITING
        logger.info(
            "SharedHostKvChannel allocated: prefix=%s capacity=%d "
            "kv_shape=%s dtype=%s slots=%d bytes/slot=%.1fMiB",
            self.name_prefix,
            new_cap,
            (new_cap,) + self._kv_chunk_shape,
            self._kv_dtype,
            self.n_slots,
            _nbytes(kv_full, self._kv_dtype) / (1024**2),
        )

    def begin_epoch_write(self) -> int:
        """Wait until the next write slot is FREE; mark it WRITING."""
        if not self.is_owner:
            raise RuntimeError("begin_epoch_write is owner-only")
        idx = self._write_idx
        if self._ctrl is None:
            return idx
        # Poll control page — Actor ACK is a store to this page (no Gloo).
        while int(self._ctrl[idx]) not in (_SLOT_FREE,):
            time.sleep(0.001)
        self._ctrl[idx] = _SLOT_WRITING
        return idx

    def write_buffers(self) -> dict[str, torch.Tensor]:
        if not self.ready:
            raise RuntimeError("SharedHostKvChannel not allocated yet")
        idx = self._write_idx
        assert self._kv[idx] is not None
        out: dict[str, torch.Tensor] = {"prefix_kv": self._kv[idx].tensor}
        if self._pad[idx] is not None:
            out["prefix_pad_masks"] = self._pad[idx].tensor
        return out

    def bind_pool(self, pool: Any) -> None:
        bufs = self.write_buffers()
        pool.bind_external(
            self.capacity,
            prefix_kv=bufs["prefix_kv"],
            prefix_pad_masks=bufs.get("prefix_pad_masks"),
            kv_chunk_shape=self._kv_chunk_shape,
            kv_dtype=self._kv_dtype,
            pad_chunk_shape=self._pad_chunk_shape,
            pad_dtype=self._pad_dtype,
        )

    def publish(self, n_rows: int) -> SharedHostKvMeta:
        if not self.is_owner or not self.ready:
            raise RuntimeError("publish requires an allocated owner channel")
        if n_rows < 0 or n_rows > self.capacity:
            raise ValueError(f"n_rows={n_rows} capacity={self.capacity}")
        idx = self._write_idx
        assert self._kv[idx] is not None and self._ctrl is not None
        self._ctrl[idx] = _SLOT_PUBLISHED
        self._version += 1
        meta = SharedHostKvMeta(
            slot=idx,
            n_rows=n_rows,
            capacity=self.capacity,
            kv_name=self._kv[idx].name,
            kv_shape=tuple(self._kv[idx].shape),
            kv_dtype=_dtype_to_code(self._kv[idx].dtype),
            pad_name=(self._pad[idx].name if self._pad[idx] is not None else None),
            pad_shape=(
                tuple(self._pad[idx].shape) if self._pad[idx] is not None else None
            ),
            pad_dtype=(
                _dtype_to_code(self._pad[idx].dtype)
                if self._pad[idx] is not None
                else None
            ),
            ctrl_name=self._ctrl_name,
            version=self._version,
        )
        self._write_idx = (idx + 1) % self.n_slots
        return meta

    def open_from_meta(self, meta: SharedHostKvMeta) -> None:
        self._attach_ctrl(meta.ctrl_name)
        need = (
            self._kv[meta.slot] is None
            or self._kv[meta.slot].name != meta.kv_name
            or tuple(self._kv[meta.slot].shape) != meta.kv_shape
        )
        if not need and meta.pad_name is not None:
            need = (
                self._pad[meta.slot] is None
                or self._pad[meta.slot].name != meta.pad_name
            )
        if not need:
            return
        if self._kv[meta.slot] is not None:
            self._kv[meta.slot].close()
            self._kv[meta.slot] = None
        if self._pad[meta.slot] is not None:
            self._pad[meta.slot].close()
            self._pad[meta.slot] = None
        self.capacity = meta.capacity
        self._kv[meta.slot] = _ShmTensor(
            meta.kv_name,
            meta.kv_shape,
            _code_to_dtype(meta.kv_dtype),
            create=False,
        )
        if meta.pad_name is not None and meta.pad_shape is not None:
            self._pad[meta.slot] = _ShmTensor(
                meta.pad_name,
                meta.pad_shape,
                _code_to_dtype(meta.pad_dtype or "bool"),
                create=False,
            )

    def view(self, meta: SharedHostKvMeta) -> dict[str, torch.Tensor]:
        self.open_from_meta(meta)
        assert self._kv[meta.slot] is not None
        n = meta.n_rows
        out: dict[str, torch.Tensor] = {
            "prefix_kv": self._kv[meta.slot].tensor[:n],
        }
        if self._pad[meta.slot] is not None:
            out["prefix_pad_masks"] = self._pad[meta.slot].tensor[:n]
        return out

    def send_ack(self, slot: int) -> None:
        """Actor: mark slot FREE so rollout may overwrite it."""
        if self._ctrl is None:
            raise RuntimeError("send_ack before open_from_meta")
        if not (0 <= slot < self.n_slots):
            raise IndexError(f"ack slot {slot} out of range")
        self._ctrl[slot] = _SLOT_FREE

    def _release_data_slots(self) -> None:
        for i in range(self.n_slots):
            if self._kv[i] is not None:
                self._kv[i].close()
                self._kv[i] = None
            if self._pad[i] is not None:
                self._pad[i].close()
                self._pad[i] = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release_data_slots()
        if self._ctrl_shm is not None:
            self._ctrl_shm.close()
            if self.is_owner:
                try:
                    self._ctrl_shm.unlink()
                except FileNotFoundError:
                    pass
            self._ctrl_shm = None
            self._ctrl = None


def resolve_prefix_kv_ipc(cfg) -> str:
    """Parse ``actor.model.openpi.prefix_kv_ipc`` → ``off`` | ``host_shm``."""
    openpi = None
    try:
        actor = cfg["actor"] if isinstance(cfg, dict) else cfg.actor
        model = actor["model"] if isinstance(actor, dict) else actor.model
        openpi = model["openpi"] if isinstance(model, dict) else model.openpi
    except Exception:
        openpi = None
    if openpi is None:
        return "off"
    if isinstance(openpi, dict):
        mode = str(openpi.get("prefix_kv_ipc", "off")).strip().lower()
    else:
        mode = str(getattr(openpi, "prefix_kv_ipc", "off")).strip().lower()
    if mode in ("", "none", "false", "0"):
        return "off"
    if mode not in ("off", "host_shm"):
        raise ValueError(
            f"openpi.prefix_kv_ipc must be 'off' or 'host_shm', got {mode!r}"
        )
    return mode
