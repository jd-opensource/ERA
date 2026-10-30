"""DirectPipe: placement-aware zero-overhead communication.

Three transport modes:
1. InProcessPipe  — co-located env+rollout in same process, zero-copy Python refs
2. NCCLPipe       — cross-GPU tensor transfer via NCCL point-to-point
3. (TODO) SharedGPUPipe — CUDA IPC for different processes on same GPU

For the MVP with co-located sync mode:
- env↔rollout: InProcessPipe (trivially, just function calls — no pipe needed)
- rollout→actor: NCCLPipe (NCCL send/recv across GPUs)
- actor→rollout: weight broadcast via NCCLPipe
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class DirectPipe(ABC):
    """Abstract base for placement-aware data transfer."""

    @abstractmethod
    def send_dict(self, data: dict[str, torch.Tensor], tag: int = 0) -> None:
        """Send a dict of tensors."""

    @abstractmethod
    def recv_dict(self, template: dict[str, torch.Tensor], tag: int = 0) -> dict[str, torch.Tensor]:
        """Receive a dict of tensors, using template for shapes/dtypes."""

    def send_tensor(self, tensor: torch.Tensor, tag: int = 0) -> None:
        self.send_dict({"_t": tensor}, tag)

    def recv_tensor(self, template: torch.Tensor, tag: int = 0) -> torch.Tensor:
        return self.recv_dict({"_t": template}, tag)["_t"]


class InProcessPipe(DirectPipe):
    """Zero-copy in-process pipe. Data is passed by reference.

    Used when env and rollout are in the same process. In sync mode this
    is not even instantiated — just call the function directly.
    """

    def __init__(self):
        self._buffer: Optional[dict[str, Any]] = None

    def send_dict(self, data: dict[str, torch.Tensor], tag: int = 0) -> None:
        self._buffer = data

    def recv_dict(self, template: dict[str, torch.Tensor], tag: int = 0) -> dict[str, torch.Tensor]:
        assert self._buffer is not None, "No data in InProcessPipe buffer"
        data = self._buffer
        self._buffer = None
        return data


class NCCLPipe(DirectPipe):
    """Cross-GPU tensor transfer via NCCL point-to-point send/recv.

    Handles dict-of-tensors by sending metadata (key count, shapes, dtypes)
    via a control tensor, then each value tensor individually.
    """

    DTYPE_MAP = {
        torch.float32: 0, torch.float16: 1, torch.bfloat16: 2,
        torch.int64: 3, torch.int32: 4, torch.bool: 5, torch.uint8: 6,
    }
    DTYPE_RMAP = {v: k for k, v in DTYPE_MAP.items()}

    def __init__(
        self,
        src_rank: int,
        dst_rank: int,
        local_rank: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.group = group

    def send_dict(self, data: dict[str, torch.Tensor], tag: int = 0) -> None:
        import json
        keys = sorted(k for k, v in data.items() if isinstance(v, torch.Tensor))
        n = len(keys)

        meta = [n]
        for k in keys:
            t = data[k]
            meta.append(len(t.shape))
            meta.extend(t.shape)
            meta.append(self.DTYPE_MAP.get(t.dtype, 0))
        meta_tensor = torch.tensor(meta, dtype=torch.int64, device=self.device)

        keys_bytes = json.dumps(keys).encode("utf-8")
        keys_tensor = torch.tensor(
            list(keys_bytes), dtype=torch.uint8, device=self.device)

        header = torch.tensor(
            [len(meta), len(keys_bytes)], dtype=torch.int64, device=self.device)
        dist.send(header, dst=self.dst_rank, group=self.group, tag=tag)
        dist.send(meta_tensor, dst=self.dst_rank, group=self.group, tag=tag + 1)
        dist.send(keys_tensor, dst=self.dst_rank, group=self.group, tag=tag + 2)

        for i, k in enumerate(keys):
            dist.send(data[k].contiguous().to(self.device), dst=self.dst_rank,
                      group=self.group, tag=tag + 3 + i)

    def recv_dict(self, template: dict[str, torch.Tensor] | None = None,
                  tag: int = 0) -> dict[str, torch.Tensor]:
        import json
        header = torch.zeros(2, dtype=torch.int64, device=self.device)
        dist.recv(header, src=self.src_rank, group=self.group, tag=tag)
        meta_len, keys_len = int(header[0].item()), int(header[1].item())

        meta_tensor = torch.zeros(meta_len, dtype=torch.int64, device=self.device)
        dist.recv(meta_tensor, src=self.src_rank, group=self.group, tag=tag + 1)
        meta = meta_tensor.tolist()

        keys_tensor = torch.zeros(keys_len, dtype=torch.uint8, device=self.device)
        dist.recv(keys_tensor, src=self.src_rank, group=self.group, tag=tag + 2)
        keys = json.loads(bytes(keys_tensor.cpu().tolist()).decode("utf-8"))

        n_keys = int(meta[0])
        idx = 1
        shapes_dtypes = []
        for _ in range(n_keys):
            ndim = int(meta[idx]); idx += 1
            shape = tuple(int(meta[idx + j]) for j in range(ndim)); idx += ndim
            dtype_code = int(meta[idx]); idx += 1
            shapes_dtypes.append((shape, self.DTYPE_RMAP.get(dtype_code, torch.float32)))

        result = {}
        for i, (k, (shape, dtype)) in enumerate(zip(keys, shapes_dtypes)):
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            dist.recv(buf, src=self.src_rank, group=self.group, tag=tag + 3 + i)
            result[k] = buf

        return result


# ------------------------------------------------------------------ #
#  commv2: all-to-all batch exchange (n rollout : m actor)
# ------------------------------------------------------------------ #

DTYPE_MAP_A2A = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.bool: 5,
    torch.uint8: 6,
    torch.float64: 7,
}
DTYPE_RMAP_A2A = {v: k for k, v in DTYPE_MAP_A2A.items()}


def _comm_cast_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Cast tensor for NCCL all_to_all (float64 was missing from dtype map)."""
    t = t.contiguous().to(device)
    if t.dtype == torch.float64:
        return t.float()
    if t.dtype not in DTYPE_MAP_A2A:
        if t.is_floating_point():
            return t.float()
        raise TypeError(f"all_to_all_batch: unsupported dtype {t.dtype}")
    return t


def all_to_all_batch(
    batch: dict[str, torch.Tensor] | None,
    n_rollouts: int,
    n_actors: int,
    my_group_rank: int,
    group: dist.ProcessGroup,
    device: torch.device,
    split_dim: int = 1,
) -> dict[str, torch.Tensor] | None:
    """Exchange rollout data via a single NCCL ``all_to_all_single`` per key.

    Group layout (fixed): ranks ``[0, n_rollouts)`` are rollouts,
    ranks ``[n_rollouts, n_rollouts + n_actors)`` are actors.

    * **Rollout** callers pass ``batch`` (dict of tensors).  Each tensor
      is split along ``split_dim`` into ``n_actors`` equal chunks and one
      chunk is sent to each actor.  Returns ``None``.
    * **Actor** callers pass ``batch=None``.  They receive one chunk from
      every rollout, concatenated along ``split_dim``.

    ``split_dim`` defaults to 1 because rollout batches have shape
    ``[T, B, ...]`` where dim-1 (B) is the env/batch axis that should be
    distributed across actors while keeping full trajectories intact.
    """
    import json

    world = n_rollouts + n_actors
    is_rollout = my_group_rank < n_rollouts

    # -- Step 1: broadcast metadata from rollout-0 so actors know the schema --
    if is_rollout:
        keys = sorted(k for k, v in batch.items() if isinstance(v, torch.Tensor))
        meta_entries = []
        for k in keys:
            t = batch[k]
            if t.dim() <= split_dim:
                raise ValueError(
                    f"all_to_all_batch: tensor '{k}' has {t.dim()} dims, "
                    f"cannot split along dim={split_dim}"
                )
            if t.shape[split_dim] % n_actors != 0:
                raise ValueError(
                    f"all_to_all_batch: tensor '{k}' shape[{split_dim}]="
                    f"{t.shape[split_dim]} not divisible by n_actors={n_actors}"
                )
            chunk_size = t.shape[split_dim] // n_actors
            full_shape = list(t.shape)
            meta_entries.append({
                "key": k,
                "chunk_dim_size": chunk_size,
                "full_shape": full_shape,
                "dtype": DTYPE_MAP_A2A.get(t.dtype, 0),
            })
        meta_bytes = json.dumps(meta_entries).encode("utf-8")
    else:
        meta_bytes = None

    meta_obj = [meta_bytes]
    src_global = dist.get_process_group_ranks(group)[0]
    dist.broadcast_object_list(meta_obj, src=src_global, group=group)

    raw = meta_obj[0]
    meta_entries = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)

    # -- Step 2: per-key all_to_all_single --
    result = {}
    for entry in meta_entries:
        k = entry["key"]
        chunk_dim_size = entry["chunk_dim_size"]
        full_shape = entry["full_shape"]
        dtype = DTYPE_RMAP_A2A.get(entry["dtype"], torch.float32)

        # Total elements in one chunk (the slice along split_dim for one actor)
        chunk_shape = list(full_shape)
        chunk_shape[split_dim] = chunk_dim_size
        elems_per_chunk = 1
        for s in chunk_shape:
            elems_per_chunk *= s

        if is_rollout:
            t = _comm_cast_tensor(batch[k], device)
            # Reorder dims: move split_dim to front, flatten for all_to_all
            # After moving split_dim to front: [split_dim_size, *rest]
            # Split along dim-0 into n_actors chunks, each of size chunk_dim_size*rest_elems
            perm = list(range(t.dim()))
            perm[0], perm[split_dim] = perm[split_dim], perm[0]
            t_perm = t.permute(perm).contiguous()
            flat_in = t_perm.reshape(-1)

            input_splits = [0] * n_rollouts + [elems_per_chunk] * n_actors
            output_splits = [0] * world
            flat_out = torch.empty(0, dtype=dtype, device=device)
        else:
            input_splits = [0] * world
            output_splits = [elems_per_chunk] * n_rollouts + [0] * n_actors
            total_out = sum(output_splits)
            flat_in = torch.empty(0, dtype=dtype, device=device)
            flat_out = torch.empty(total_out, dtype=dtype, device=device)

        dist.all_to_all_single(
            flat_out, flat_in,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )

        if not is_rollout:
            # Reconstruct: flat_out has n_rollouts chunks each of elems_per_chunk
            # Reshape to [n_rollouts * chunk_dim_size, *rest_dims_with_split_dim_first]
            recv_shape = list(full_shape)
            recv_shape[split_dim] = chunk_dim_size * n_rollouts
            # The data arrived with split_dim transposed to dim-0.
            # First reshape to [n_rollouts * chunk_dim_size, *shape_without_split_dim]
            shape_no_split = [full_shape[i] for i in range(len(full_shape)) if i != split_dim]
            t_recv = flat_out.reshape(chunk_dim_size * n_rollouts, *shape_no_split)
            # Permute back: move dim-0 back to split_dim position
            if split_dim != 0:
                perm = list(range(t_recv.dim()))
                perm.insert(split_dim, perm.pop(0))
                t_recv = t_recv.permute(perm).contiguous()
            result[k] = t_recv

    return None if is_rollout else result


def warmup_all2all(
    n_rollouts: int,
    n_actors: int,
    my_group_rank: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> None:
    """Pre-warm the all2all NCCL communicator with a tiny dummy exchange."""
    is_rollout = my_group_rank < n_rollouts
    dummy_batch = None
    if is_rollout:
        dummy_batch = {"_w": torch.zeros(1, n_actors, device=device)}
    all_to_all_batch(dummy_batch, n_rollouts, n_actors, my_group_rank, group, device)
    logger.info(
        "all2all warmup done: group_rank=%d role=%s",
        my_group_rank, "rollout" if is_rollout else "actor",
    )


def broadcast_state_dict(
    state_dict: dict[str, torch.Tensor],
    src_rank: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Broadcast model state_dict from src_rank to all members of group."""
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name].to(device)
        dist.broadcast(tensor, src=src_rank, group=group)
        state_dict[name] = tensor
    return state_dict
