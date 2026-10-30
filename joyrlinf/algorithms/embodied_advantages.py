"""Embodied GRPO advantage preprocessing (aligned with RLinf algorithms/utils.py)."""

from __future__ import annotations

from typing import Optional

import torch

from joyrlinf.algorithms.grpo import calculate_episode_scores


def compute_loss_mask(dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-sub-step loss masks from action-level dones (RLinf layout).

    Args:
        dones: [n_chunk_step + 1, bsz, num_action_chunks] with a leading all-False
            row before the first action and post-chunk done rows thereafter.

    Returns:
        loss_mask: [n_chunk_step, bsz, num_action_chunks]
        loss_mask_sum: same shape, per-env valid step counts for loss aggregation
    """
    _, actual_bsz, num_action_chunks = dones.shape
    n_chunk_step = dones.shape[0] - 1
    flattened_dones = dones.transpose(1, 2).reshape(-1, actual_bsz)
    flattened_dones = flattened_dones[-(n_chunk_step * num_action_chunks + 1) :]
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[:-1]

    loss_mask = flattened_loss_mask.reshape(
        n_chunk_step, num_action_chunks, actual_bsz,
    ).transpose(1, 2)
    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True).expand_as(loss_mask)
    return loss_mask, loss_mask_sum


def pack_chunk_step_signals(
    chunk_rewards: torch.Tensor,
    chunk_terms: torch.Tensor,
    chunk_truncs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack env chunk outputs as [B, num_action_chunks] for rollout batches."""
    rewards = chunk_rewards
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(-1)
    terminations = chunk_terms
    truncations = chunk_truncs
    if terminations.dim() == 1:
        terminations = terminations.unsqueeze(-1)
    if truncations.dim() == 1:
        truncations = truncations.unsqueeze(-1)
    dones = torch.logical_or(terminations, truncations)
    return rewards, dones, terminations, truncations


def preprocess_embodied_advantages_inputs(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    reward_type: str,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
) -> dict:
    """Flatten [T, B, C] chunk tensors to [T*C, B] for score accumulation."""
    if reward_type == "chunk_level":
        rewards = rewards.sum(dim=-1, keepdim=True)
        dones = dones.max(dim=-1, keepdim=True)[0]
        if loss_mask is not None:
            loss_mask = loss_mask.max(dim=-1, keepdim=True)[0]
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.max(dim=-1, keepdim=True)[0]

    if rewards.dim() == 2:
        rewards = rewards.unsqueeze(-1)
    if dones.dim() == 2:
        dones = dones.unsqueeze(-1)

    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size

    rewards = rewards.transpose(1, 2).reshape(n_steps, bsz)
    if loss_mask is not None:
        loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz)

    flattened_dones_full = dones.transpose(1, 2).reshape(
        (num_chunk + 1) * chunk_size, bsz
    )
    dones = flattened_dones_full[-(n_steps + 1) :]

    return {
        "rewards": rewards,
        "dones": dones,
        "loss_mask": loss_mask,
        "loss_mask_sum": loss_mask_sum,
        "num_chunk": num_chunk,
        "batch_size": bsz,
        "chunk_size": chunk_size,
        "n_steps": n_steps,
    }


def postprocess_embodied_advantages_outputs(
    advantages: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    reward_type: str = "action_level",
) -> torch.Tensor:
    """Map [n_steps, B] advantages back to rollout time axis for actor training.

    chunk_level (chunk_size=1): [T_chunk, B] — one scalar per (chunk_step, env).
    action_level: [T_chunk, B, C] — one value per action chunk.
    """
    if reward_type == "chunk_level" and chunk_size == 1:
        return advantages.reshape(num_chunk, -1)
    return advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)


def compute_embodied_grpo_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    group_size: int,
    reward_type: str = "action_level",
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute GRPO advantages with RLinf action_level / chunk_level time axes.

    Returns:
        chunk_level: [num_chunk_steps, batch_size]
        action_level: [num_chunk_steps, batch_size, chunk_size]
    """
    prep = preprocess_embodied_advantages_inputs(
        rewards,
        dones,
        reward_type=reward_type,
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask_sum,
    )
    grouped_scores = calculate_episode_scores(
        prep["rewards"],
        prep["dones"],
        prep["n_steps"],
        prep["batch_size"],
        group_size,
    )
    grouped_reward_mean = grouped_scores.mean(dim=-1, keepdim=True).expand_as(
        grouped_scores
    )
    grouped_reward_std = grouped_scores.std(dim=-1, keepdim=True).expand_as(
        grouped_scores
    )
    advantages_grouped = (grouped_scores - grouped_reward_mean) / (
        grouped_reward_std + 1e-6
    )

    n_steps = prep["n_steps"]
    bsz = prep["batch_size"]
    loss_mask = prep["loss_mask"]
    if loss_mask is None:
        loss_mask = torch.ones(
            n_steps, bsz, device=rewards.device, dtype=torch.float32,
        )
    advantages_2d = (
        torch.zeros_like(loss_mask) + advantages_grouped.reshape(-1).view(1, -1)
    ) * loss_mask

    return postprocess_embodied_advantages_outputs(
        advantages_2d,
        prep["num_chunk"],
        prep["chunk_size"],
        reward_type=reward_type,
    )
