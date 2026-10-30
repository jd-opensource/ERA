"""GRPO (Group Relative Policy Optimization) advantage computation.

For embodied RL: episode scores are computed per env, then grouped by
initial state (group_size envs share same init state), and advantages
are normalized within each group.
"""

from __future__ import annotations

import torch


def calculate_episode_scores(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    n_steps: int,
    batch_size: int,
    group_size: int,
    **kwargs,
) -> torch.Tensor:
    """Compute per-episode cumulative scores from step-level rewards.

    Args:
        rewards: [n_steps, batch_size] step rewards
        dones:   [n_steps+1, batch_size] done flags (extra for bootstrap)
        n_steps: number of chunk steps
        batch_size: number of envs on this rank
        group_size: GRPO group size

    Returns:
        scores: [num_groups, group_size] episode scores
    """
    scores = torch.zeros(batch_size, device=rewards.device)
    for step in reversed(range(n_steps)):
        scores = scores * ~dones[step + 1]
        scores += rewards[step]
    return scores.reshape(-1, group_size)


def compute_grpo_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    n_steps: int,
    batch_size: int,
    group_size: int,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Compute GRPO advantages.

    1. Calculate episode scores per env
    2. Group by group_size (envs sharing same initial state)
    3. Normalize within each group (mean/std)

    Returns:
        advantages: [batch_size] per-env advantage
        returns: None (GRPO doesn't use returns)
    """
    # [num_groups, group_size]
    grouped_scores = calculate_episode_scores(
        rewards, dones, n_steps, batch_size, group_size
    )

    mean = grouped_scores.mean(dim=-1, keepdim=True)
    std = grouped_scores.std(dim=-1, keepdim=True)
    advantages = (grouped_scores - mean) / (std + 1e-6)

    return advantages.reshape(-1), None


def compute_ppo_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns for PPO. (TODO: full implementation)"""
    n_steps = rewards.shape[0]
    batch_size = rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(n_steps)):
        next_non_terminal = ~dones[t + 1].float()
        next_values = values[t + 1] if t + 1 < values.shape[0] else torch.zeros_like(values[0])
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:n_steps]
    return advantages.reshape(-1), returns.reshape(-1)
