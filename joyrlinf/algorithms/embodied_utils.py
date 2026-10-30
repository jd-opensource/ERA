"""Embodied rollout post-processing (loss masks, reward filters)."""

from __future__ import annotations

from typing import Optional

import torch

from joyrlinf.algorithms.embodied_advantages import compute_loss_mask


def apply_reward_filter(
    rewards: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    group_size: int,
    rewards_lower_bound: float,
    rewards_upper_bound: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Filter GRPO groups by mean episode reward (RLinf EmbodiedFSDPActor)."""
    if loss_mask is not None:
        rewards = rewards * loss_mask

    n_chunk_step, batch_size, num_action_chunks = rewards.shape
    if batch_size % group_size != 0:
        raise ValueError(
            f"batch {batch_size} not divisible by group_size {group_size}"
        )
    n_prompts = batch_size // group_size

    rew = rewards.transpose(0, 1).reshape(batch_size, -1)
    reward_matrix = rew.reshape(n_prompts, group_size, -1).sum(dim=-1)
    mean_reward_in_group = reward_matrix.mean(dim=1)

    reward_filter_mask = (
        (mean_reward_in_group >= rewards_lower_bound)
        & (mean_reward_in_group <= rewards_upper_bound)
    )
    reward_filter_mask = reward_filter_mask.repeat_interleave(group_size)
    reward_filter_mask = (
        reward_filter_mask.unsqueeze(0)
        .expand(n_chunk_step, -1)
        .unsqueeze(-1)
    )

    if loss_mask is not None:
        out_mask = reward_filter_mask & loss_mask
    else:
        out_mask = reward_filter_mask

    kept = (
        (mean_reward_in_group >= rewards_lower_bound)
        & (mean_reward_in_group <= rewards_upper_bound)
    )
    stats = {
        "reward_filter_groups_kept_pct": 100.0 * kept.float().mean().item(),
        "reward_filter_groups_total": float(n_prompts),
    }
    return out_mask, stats


def process_received_rollout_batch(
    rollout_batch: dict[str, torch.Tensor],
    *,
    auto_reset: bool,
    ignore_terminations: bool,
    reward_type: str,
    filter_rewards: bool,
    rewards_lower_bound: float,
    rewards_upper_bound: float,
    group_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Build loss masks and optional reward filters before advantage compute."""
    metrics: dict[str, float] = {}

    if not auto_reset and not ignore_terminations:
        dones = rollout_batch["dones"]
        loss_mask, loss_mask_sum = compute_loss_mask(dones)
        if reward_type == "chunk_level":
            loss_mask = loss_mask.any(dim=-1, keepdim=True)
            loss_mask_sum = loss_mask_sum[..., -1:]
        rollout_batch["loss_mask"] = loss_mask
        rollout_batch["loss_mask_sum"] = loss_mask_sum

    if filter_rewards:
        rewards = rollout_batch["rewards"]
        loss_mask, filter_stats = apply_reward_filter(
            rewards,
            rollout_batch.get("loss_mask"),
            group_size=group_size,
            rewards_lower_bound=rewards_lower_bound,
            rewards_upper_bound=rewards_upper_bound,
        )
        rollout_batch["loss_mask"] = loss_mask
        if rollout_batch.get("loss_mask_sum") is not None:
            rollout_batch["loss_mask_sum"] = loss_mask.sum(
                dim=(0, 2), keepdim=True,
            ).expand_as(loss_mask)
        metrics.update(filter_stats)

    if isinstance(rollout_batch.get("loss_mask"), torch.Tensor):
        metrics["loss_mask_frac"] = float(
            rollout_batch["loss_mask"].float().mean().item()
        )

    return rollout_batch, metrics
