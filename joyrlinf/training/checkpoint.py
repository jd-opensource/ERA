"""Lightweight FSDP actor checkpoint save/load (no_shard friendly)."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

logger = logging.getLogger(__name__)


def _is_src_rank(group: dist.ProcessGroup | None) -> bool:
    if group is None:
        return dist.get_rank() == 0
    return dist.get_rank(group) == 0


def save_actor_checkpoint(
    fsdp_model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    save_dir: str,
    global_step: int,
    trainable_param_names: set[str] | None,
    process_group: dist.ProcessGroup | None = None,
) -> None:
    """Save actor training state under *save_dir* (rank 0 of *process_group* writes)."""
    os.makedirs(save_dir, exist_ok=True)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        full_sd = fsdp_model.state_dict()

    if trainable_param_names is not None:
        filtered = {}
        for k, v in full_sd.items():
            norm = k
            for prefix in ("_fsdp_wrapped_module.", "module.", "_orig_mod."):
                if norm.startswith(prefix):
                    norm = norm[len(prefix):]
            if norm.startswith("model."):
                norm = norm[len("model."):]
            if norm in trainable_param_names or k in trainable_param_names:
                filtered[k] = v
        full_sd = filtered if filtered else full_sd

    if _is_src_rank(process_group):
        torch.save(full_sd, os.path.join(save_dir, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
        meta = {"global_step": global_step}
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        logger.info("Saved checkpoint to %s (step=%d)", save_dir, global_step)
    # Do NOT barrier here: non-src actor ranks must return to NCCL all2all recv
    # immediately.  Rank-0 disk I/O can overlap with the next rollout transfer.


def load_actor_checkpoint(
    fsdp_model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    load_dir: str,
    process_group: dist.ProcessGroup | None = None,
) -> int:
    """Load actor checkpoint from *load_dir*. Returns global_step from meta.json."""
    model_path = os.path.join(load_dir, "model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint model not found: {model_path}")

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    model_sd = torch.load(model_path, map_location="cpu", weights_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        fsdp_model.load_state_dict(model_sd, strict=False)

    opt_path = os.path.join(load_dir, "optimizer.pt")
    if os.path.isfile(opt_path):
        optimizer.load_state_dict(
            torch.load(opt_path, map_location="cpu", weights_only=True)
        )

    sched_path = os.path.join(load_dir, "scheduler.pt")
    if os.path.isfile(sched_path):
        scheduler.load_state_dict(
            torch.load(sched_path, map_location="cpu", weights_only=True)
        )

    meta_path = os.path.join(load_dir, "meta.json")
    global_step = 0
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            global_step = int(json.load(f).get("global_step", 0))

    if process_group is not None:
        dist.barrier(group=process_group)
    else:
        dist.barrier()

    logger.info("Loaded checkpoint from %s (step=%d)", load_dir, global_step)
    return global_step


def load_actor_model_state_dict_cpu(load_dir: str) -> dict[str, torch.Tensor]:
    """Load ``model.pt`` from an actor checkpoint directory (CPU tensors).

    Used by colocated sync mode to seed inference weights without constructing
    a full FSDP actor on every rank at startup.
    """
    model_path = os.path.join(load_dir, "model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint model not found: {model_path}")
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise TypeError(f"Expected state dict in {model_path}, got {type(sd)}")
    return sd
