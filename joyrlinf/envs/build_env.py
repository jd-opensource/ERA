"""Factory for embodied environment wrappers."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from joyrlinf.core.launcher import Placement
from joyrlinf.envs import get_env_cls
from joyrlinf.envs.libero.libero_env import LiberoEnv
from joyrlinf.envs.maniskill_env import ManiSkillEnv


def _env_slot_index(rank: int, placement: Placement, colocated: bool) -> int:
    if colocated:
        return placement.rollout_ranks.index(rank)
    if rank in placement.env_ranks:
        return placement.env_ranks.index(rank)
    return placement.rollout_ranks.index(rank)


def _env_slot_count(placement: Placement, colocated: bool) -> int:
    return len(placement.rollout_ranks if colocated else placement.env_ranks)


def build_env(
    cfg_section: DictConfig,
    num_envs: int,
    rank: int,
    placement: Placement,
    device: torch.device,
    *,
    colocated: bool | None = None,
):
    env_type = str(cfg_section.get("env_type", "maniskill")).lower()
    if colocated is None:
        colocated = placement.colocated

    if env_type == "maniskill":
        return ManiSkillEnv(
            cfg=cfg_section,
            num_envs=num_envs,
            seed_offset=rank * 1000,
            reset_partition=str(cfg_section.get("reset_partition", "per_rank")),
            slot_index=_env_slot_index(rank, placement, colocated),
            slot_count=_env_slot_count(placement, colocated),
        )

    if env_type == "libero":
        slot = _env_slot_index(rank, placement, colocated)
        total = _env_slot_count(placement, colocated)
        return LiberoEnv(
            cfg=cfg_section,
            num_envs=num_envs,
            seed_offset=slot,
            total_num_processes=total,
            device=device,
        )

    if env_type == "wan_wm":
        env_cls = get_env_cls(env_type)
        return env_cls(
            cfg=cfg_section,
            num_envs=num_envs,
            seed_offset=rank * 1000,
        )

    raise ValueError(
        f"Unknown env_type={env_type!r}; supported: maniskill, libero, wan_wm"
    )
