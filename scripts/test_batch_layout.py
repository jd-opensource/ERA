#!/usr/bin/env python3
"""CPU smoke tests for embodied batch layout (chunk_level / action_level)."""
from __future__ import annotations

import torch

from joyrlinf.algorithms.embodied_advantages import (
    compute_embodied_grpo_advantages,
    compute_loss_mask,
)
from joyrlinf.algorithms.embodied_batch_layout import (
    flatten_for_training,
    layout_from_batch,
)
from joyrlinf.algorithms.embodied_utils import process_received_rollout_batch
from joyrlinf.algorithms.loss import policy_loss


def _make_rollout_batch(t: int = 4, b: int = 8, c: int = 5, d: int = 7):
    dones = torch.zeros(t + 1, b, c, dtype=torch.bool)
    return {
        "rewards": torch.randn(t, b, c),
        "dones": dones,
        "prev_logprobs": torch.randn(t, b, c * d),
        "forward_inputs": {"obs": torch.randn(t, b, 3)},
    }


def test_chunk_level_flatten_and_loss():
    t, b, c, d = 4, 8, 5, 7
    batch = _make_rollout_batch(t, b, c, d)
    batch, _ = process_received_rollout_batch(
        batch,
        auto_reset=False,
        ignore_terminations=False,
        reward_type="chunk_level",
        filter_rewards=False,
        group_size=2,
    )
    layout = layout_from_batch(
        batch, reward_type="chunk_level", logprob_type="chunk_level", action_dim=d,
    )
    adv = compute_embodied_grpo_advantages(
        batch["rewards"],
        batch["dones"],
        group_size=2,
        reward_type="chunk_level",
        loss_mask=batch.get("loss_mask"),
        loss_mask_sum=batch.get("loss_mask_sum"),
    )
    assert adv.shape == (t, b), f"expected [T,B], got {adv.shape}"
    batch["advantages"] = adv

    flat = flatten_for_training(
        batch, layout, compute_loss_mask_fn=compute_loss_mask,
    )
    n = t * b
    assert flat["prev_logprobs"].shape == (n, c * d)
    assert flat["advantages"].shape == (n,)
    assert flat["loss_mask"].shape == (n,)
    assert flat["loss_mask_sum"].shape == (n,)

    mb = 4
    loss, _ = policy_loss(
        logprobs=flat["prev_logprobs"][:mb],
        old_logprobs=flat["prev_logprobs"][:mb],
        advantages=flat["advantages"][:mb],
        loss_mask=flat["loss_mask"][:mb],
        loss_mask_sum=flat["loss_mask_sum"][:mb],
        max_episode_steps=80,
        logprob_type="chunk_level",
        reward_type="chunk_level",
        action_dim=d,
        layout=layout,
    )
    assert torch.isfinite(loss)


def test_action_level_flatten_shapes():
    t, b, c, d = 4, 8, 5, 7
    batch = _make_rollout_batch(t, b, c, d)
    layout = layout_from_batch(
        batch, reward_type="action_level", logprob_type="token_level", action_dim=d,
    )
    adv = compute_embodied_grpo_advantages(
        batch["rewards"], batch["dones"], group_size=2, reward_type="action_level",
    )
    assert adv.shape == (t, b, c)
    batch["advantages"] = adv
    flat = flatten_for_training(
        batch, layout, compute_loss_mask_fn=compute_loss_mask,
    )
    n = t * b
    assert flat["advantages"].shape == (n, c)
    assert flat["loss_mask"].shape == (n, c)


def test_process_received_filter_chunk_level():
    t, b, c = 2, 4, 1
    dones = torch.zeros(t + 1, b, c, dtype=torch.bool)
    rewards = torch.zeros(t, b, c)
    rewards[0, 2, 0] = 0.5
    batch = {"rewards": rewards, "dones": dones, "prev_logprobs": torch.randn(t, b, 7)}
    out, metrics = process_received_rollout_batch(
        batch,
        auto_reset=True,
        ignore_terminations=False,
        reward_type="chunk_level",
        filter_rewards=True,
        rewards_lower_bound=0.0,
        rewards_upper_bound=1.0,
        group_size=2,
    )
    assert "loss_mask" in out
    assert "reward_filter_groups_kept_pct" in metrics


if __name__ == "__main__":
    test_chunk_level_flatten_and_loss()
    test_action_level_flatten_shapes()
    test_process_received_filter_chunk_level()
    print("test_batch_layout: OK")
