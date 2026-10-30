"""Embodied PPO GAE / returns (RLinf-aligned time layout).

Time layout (matches ``EmbodiedRolloutBatch``):
  rewards      [T, B, C]      per chunk step, per action sub-step
  dones        [T+1, B, C]    leading all-False row + post-chunk done rows
  prev_values  [T+1, B, C]    action-level critic values, or [T+1, B, 1]
                              for chunk-level critic values
                              (bootstrap row from the final extra predict;
                              only its first sub-step value is used)

Action-level GAE runs on the flattened sub-step axis [T*C, B], then maps
back to [T, B, C]. Chunk-level GAE first reduces rewards/dones to one scalar
per chunk step and returns [T, B], matching OpenPI/Pi0.5 critic outputs.
"""

from __future__ import annotations

from typing import Optional

import torch


def masked_mean_std(
    tensor: torch.Tensor, mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return tensor.mean(), tensor.std()
    m = mask.float()
    count = m.sum().clamp(min=1.0)
    mean = (tensor * m).sum() / count
    var = ((tensor - mean) ** 2 * m).sum() / count
    return mean, var.clamp(min=0.0).sqrt()


def masked_whiten(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize to zero mean / unit std over masked entries; re-apply mask."""
    mean, std = masked_mean_std(tensor, mask)
    out = (tensor - mean) / (std + eps)
    if mask is not None:
        out = out * mask.float()
    return out


def compute_embodied_gae_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    prev_values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    loss_mask: Optional[torch.Tensor] = None,
    reward_type: str = "action_level",
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE for embodied rollout batches.

    Args:
        rewards: [T, B, C] float
        dones: [T+1, B, C] bool (leading zero row + post-chunk rows)
        prev_values: [T+1, B, C] action-level or [T+1, B, 1] chunk-level values
        loss_mask: optional [T, B, C] or [T, B, 1]; applied to outputs
        reward_type: "action_level" preserves the original OpenVLA PPO path;
            "chunk_level" supports OpenPI/Pi0.5 scalar chunk critics.

    Returns:
        action_level: (advantages [T, B, C], returns [T, B, C])
        chunk_level: (advantages [T, B], returns [T, B])
    """
    if rewards.dim() != 3:
        raise ValueError(f"rewards must be [T,B,C], got {tuple(rewards.shape)}")
    reward_type = str(reward_type)
    if reward_type not in ("action_level", "chunk_level"):
        raise ValueError(
            f"reward_type must be 'action_level' or 'chunk_level', got {reward_type!r}"
        )

    if reward_type == "chunk_level":
        rewards = rewards.sum(dim=-1, keepdim=True)
        if dones.dim() != 3:
            raise ValueError(f"dones must be [T+1,B,C], got {tuple(dones.shape)}")
        dones = dones.max(dim=-1, keepdim=True)[0]
        if prev_values.dim() == 2:
            prev_values = prev_values.unsqueeze(-1)
        if prev_values.dim() != 3 or prev_values.shape[-1] != 1:
            raise ValueError(
                "chunk_level GAE expects prev_values [T+1,B,1] "
                f"(or [T+1,B]), got {tuple(prev_values.shape)}"
            )
        if loss_mask is not None:
            if loss_mask.dim() == 2:
                loss_mask = loss_mask.unsqueeze(-1)
            elif loss_mask.dim() == 3:
                loss_mask = loss_mask.max(dim=-1, keepdim=True)[0]
            else:
                raise ValueError(
                    f"chunk_level loss_mask must be [T,B] or [T,B,C], got {tuple(loss_mask.shape)}"
                )

    t_chunk, bsz, chunk_size = rewards.shape
    if tuple(dones.shape) != (t_chunk + 1, bsz, chunk_size):
        raise ValueError(
            f"dones must be [T+1,B,C]={t_chunk + 1, bsz, chunk_size}, "
            f"got {tuple(dones.shape)}"
        )
    if tuple(prev_values.shape) != (t_chunk + 1, bsz, chunk_size):
        raise ValueError(
            f"prev_values must be [T+1,B,C]={t_chunk + 1, bsz, chunk_size} "
            f"(T chunk rows + 1 bootstrap row), got {tuple(prev_values.shape)}"
        )
    if loss_mask is not None and tuple(loss_mask.shape) != (t_chunk, bsz, chunk_size):
        raise ValueError(
            f"loss_mask must be [T,B,C]={t_chunk, bsz, chunk_size}, "
            f"got {tuple(loss_mask.shape)}"
        )

    n_steps = t_chunk * chunk_size
    rewards_flat = rewards.float().transpose(1, 2).reshape(n_steps, bsz)

    # Same layout trick as preprocess_embodied_advantages_inputs: keep the row
    # preceding each sub-step so dones_flat[t+1] is the post-step done of step t.
    dones_flat = dones.transpose(1, 2).reshape((t_chunk + 1) * chunk_size, bsz)
    dones_flat = dones_flat[-(n_steps + 1):]

    values_chunk = prev_values[:t_chunk].float().transpose(1, 2).reshape(n_steps, bsz)
    bootstrap = prev_values[t_chunk, :, 0].float().unsqueeze(0)  # [1, B]
    values_flat = torch.cat([values_chunk, bootstrap], dim=0)    # [T*C+1, B]

    advantages = torch.zeros(n_steps, bsz, dtype=torch.float32, device=rewards.device)
    last_gae = torch.zeros(bsz, dtype=torch.float32, device=rewards.device)
    for t in reversed(range(n_steps)):
        next_nonterminal = (~dones_flat[t + 1].bool()).float()
        delta = (
            rewards_flat[t]
            + gamma * values_flat[t + 1] * next_nonterminal
            - values_flat[t]
        )
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values_flat[:n_steps]

    # [T*C, B] -> [T, C, B] -> [T, B, C]
    advantages = advantages.reshape(t_chunk, chunk_size, bsz).transpose(1, 2)
    returns = returns.reshape(t_chunk, chunk_size, bsz).transpose(1, 2)

    if loss_mask is not None:
        advantages = advantages * loss_mask.float()
        returns = returns * loss_mask.float()

    if reward_type == "chunk_level":
        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)
    return advantages, returns
