#!/usr/bin/env python3
"""CPU unit tests for embodied PPO: GAE, actor_critic loss, stale correction."""
from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from joyrlinf.algorithms.embodied_advantages import compute_loss_mask
from joyrlinf.algorithms.embodied_batch_layout import (
    flatten_for_training,
    layout_from_batch,
)
from joyrlinf.algorithms.loss import policy_loss
from joyrlinf.algorithms.ppo import compute_embodied_gae_advantages, masked_whiten


def _make_ppo_rollout_batch(t: int = 4, b: int = 8, c: int = 5, d: int = 7):
    dones = torch.zeros(t + 1, b, c, dtype=torch.bool)
    return {
        "rewards": torch.randn(t, b, c),
        "dones": dones,
        "prev_logprobs": torch.randn(t, b, c * d),
        "prev_values": torch.randn(t + 1, b, c),
        "versions": torch.zeros(t, b, c),
        "forward_inputs": {"obs": torch.randn(t, b, 3)},
    }


def test_gae_shapes():
    t, b, c = 4, 8, 5
    batch = _make_ppo_rollout_batch(t, b, c)
    adv, ret = compute_embodied_gae_advantages(
        rewards=batch["rewards"],
        dones=batch["dones"],
        prev_values=batch["prev_values"],
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert adv.shape == (t, b, c), f"advantages shape {adv.shape}"
    assert ret.shape == (t, b, c), f"returns shape {ret.shape}"
    assert torch.isfinite(adv).all() and torch.isfinite(ret).all()

    # shape errors must raise, not broadcast
    try:
        compute_embodied_gae_advantages(
            rewards=batch["rewards"],
            dones=batch["dones"][:t],          # wrong: [T,B,C]
            prev_values=batch["prev_values"],
        )
        raise AssertionError("expected ValueError for bad dones shape")
    except ValueError:
        pass
    try:
        compute_embodied_gae_advantages(
            rewards=batch["rewards"],
            dones=batch["dones"],
            prev_values=batch["prev_values"][:t],  # wrong: missing bootstrap
        )
        raise AssertionError("expected ValueError for bad prev_values shape")
    except ValueError:
        pass


def test_gae_values_hand_checked():
    """T=1, B=1, C=2, gamma=lam=1: closed-form telescoping check."""
    t, b, c = 1, 1, 2
    rewards = torch.tensor([[[1.0, 0.5]]])              # [1,1,2]
    dones = torch.zeros(t + 1, b, c, dtype=torch.bool)
    prev_values = torch.tensor([[[0.2, 0.4]], [[0.6, 999.0]]])  # bootstrap row: only [...,0] used
    adv, ret = compute_embodied_gae_advantages(
        rewards, dones, prev_values, gamma=1.0, gae_lambda=1.0,
    )
    # flattened: r=[1.0, 0.5], v=[0.2, 0.4], bootstrap=0.6
    # delta1 = 0.5 + 0.6 - 0.4 = 0.7 ; adv1 = 0.7
    # delta0 = 1.0 + 0.4 - 0.2 = 1.2 ; adv0 = 1.2 + 0.7 = 1.9
    assert torch.allclose(adv[0, 0], torch.tensor([1.9, 0.7]), atol=1e-6), adv
    assert torch.allclose(ret[0, 0], torch.tensor([2.1, 1.1]), atol=1e-6), ret

    # terminal after last sub-step: bootstrap must be cut off
    dones2 = dones.clone()
    dones2[1, 0, 1] = True
    adv2, _ = compute_embodied_gae_advantages(
        rewards, dones2, prev_values, gamma=1.0, gae_lambda=1.0,
    )
    # delta1 = 0.5 - 0.4 = 0.1 ; adv0 = 1.2 + 0.1 = 1.3
    assert torch.allclose(adv2[0, 0], torch.tensor([1.3, 0.1]), atol=1e-6), adv2


def test_masked_whiten():
    x = torch.randn(4, 8, 5)
    mask = (torch.rand(4, 8, 5) > 0.3).float()
    out = masked_whiten(x, mask)
    m = mask.bool()
    assert abs(out[m].mean().item()) < 1e-4
    assert (out[~m] == 0).all()


def _flat_ppo_batch(t=4, b=8, c=5, d=7):
    batch = _make_ppo_rollout_batch(t, b, c, d)
    layout = layout_from_batch(
        batch, reward_type="action_level", logprob_type="token_level", action_dim=d,
    )
    adv, ret = compute_embodied_gae_advantages(
        rewards=batch["rewards"], dones=batch["dones"],
        prev_values=batch["prev_values"],
    )
    batch["advantages"] = adv
    batch["returns"] = ret
    flat = flatten_for_training(batch, layout, compute_loss_mask_fn=compute_loss_mask)
    n = t * b
    assert flat["advantages"].shape == (n, c)
    assert flat["returns"].shape == (n, c)
    assert flat["prev_values"].shape == (n, c)  # bootstrap row sliced off
    assert flat["versions"].shape == (n, c)
    return flat, layout


def test_chunk_level_gae_and_actor_critic_loss():
    """Pi0.5 PPO path: chunk-level rewards/logprobs with scalar critic values."""
    t, b, c, d = 4, 8, 5, 7
    dones = torch.zeros(t + 1, b, c, dtype=torch.bool)
    dones[2, 0, 3:] = True
    batch = {
        "rewards": torch.randn(t, b, c),
        "dones": dones,
        "prev_logprobs": torch.randn(t, b, c * d),
        "prev_values": torch.randn(t + 1, b, 1),
        "versions": torch.zeros(t, b, 1),
        "forward_inputs": {"obs": torch.randn(t, b, 3)},
    }
    adv, ret = compute_embodied_gae_advantages(
        rewards=batch["rewards"],
        dones=batch["dones"],
        prev_values=batch["prev_values"],
        gamma=0.99,
        gae_lambda=0.95,
        reward_type="chunk_level",
    )
    assert adv.shape == (t, b), f"chunk advantages shape {adv.shape}"
    assert ret.shape == (t, b), f"chunk returns shape {ret.shape}"
    assert torch.isfinite(adv).all() and torch.isfinite(ret).all()

    batch["advantages"] = masked_whiten(adv)
    batch["returns"] = ret
    layout = layout_from_batch(
        batch, reward_type="chunk_level", logprob_type="chunk_level", action_dim=d,
    )
    flat = flatten_for_training(batch, layout, compute_loss_mask_fn=compute_loss_mask)
    n = t * b
    assert flat["advantages"].shape == (n,)
    assert flat["returns"].shape == (n,)
    assert flat["prev_values"].shape == (n,)
    assert flat["versions"].shape == (n,)

    lin = torch.nn.Linear(4, flat["prev_logprobs"].shape[1])
    logprobs = lin(torch.randn(n, 4))
    vlin = torch.nn.Linear(4, 1)
    values = vlin(torch.randn(n, 4)).squeeze(-1)
    loss, metrics = policy_loss(
        logprobs=logprobs,
        old_logprobs=flat["prev_logprobs"],
        advantages=flat["advantages"],
        loss_mask=flat["loss_mask"],
        logprob_type="chunk_level",
        reward_type="chunk_level",
        action_dim=d,
        layout=layout,
        loss_type="decoupled_actor_critic",
        values=values,
        returns=flat["returns"],
        prev_values=flat["prev_values"],
        versions=flat["versions"],
        current_version=1,
        behave_weight_threshold=2.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()
    assert vlin.weight.grad is not None and torch.isfinite(vlin.weight.grad).all()
    for key in ("policy_loss", "value_loss", "proximal_approx_kl", "returns_mean"):
        assert key in metrics, f"missing metric {key}"
        assert metrics[key] == metrics[key], f"NaN metric {key}"


def test_actor_critic_loss_forward_backward():
    """Sync PPO (actor_critic): actor+critic, strictly on-policy, no proximal."""
    flat, layout = _flat_ppo_batch()
    d = layout.action_dim
    n = flat["prev_logprobs"].shape[0]

    lin = torch.nn.Linear(4, flat["prev_logprobs"].shape[1])
    logprobs = lin(torch.randn(n, 4))
    vlin = torch.nn.Linear(4, flat["advantages"].shape[1])
    values = vlin(torch.randn(n, 4))

    loss, metrics = policy_loss(
        logprobs=logprobs,
        old_logprobs=flat["prev_logprobs"],
        advantages=flat["advantages"],
        loss_mask=flat["loss_mask"],
        logprob_type="token_level",
        reward_type="action_level",
        action_dim=d,
        layout=layout,
        loss_type="actor_critic",
        values=values,
        returns=flat["returns"],
        prev_values=flat["prev_values"],
        value_clip=0.2,
        huber_delta=10.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()
    assert vlin.weight.grad is not None and torch.isfinite(vlin.weight.grad).all()

    for key in (
        "policy_loss", "value_loss", "clip_fraction", "ratio_mean", "ratio_abs",
        "approx_kl", "advantages_mean", "values_mean", "returns_mean",
    ):
        assert key in metrics, f"missing metric {key}"
        assert metrics[key] == metrics[key], f"NaN metric {key}"  # not NaN
    # sync path never runs proximal stale correction
    assert "proximal_approx_kl" not in metrics
    assert "behav_clip_fraction" not in metrics


def test_actor_critic_ignores_versions():
    """Sync PPO is strictly on-policy: versions must NOT alter the loss."""
    flat, layout = _flat_ppo_batch()
    d = layout.action_dim
    torch.manual_seed(0)
    logprobs = flat["prev_logprobs"] + 0.5 * torch.randn_like(flat["prev_logprobs"])
    values = flat["prev_values"] + 0.1
    kwargs = dict(
        logprobs=logprobs,
        old_logprobs=flat["prev_logprobs"],
        advantages=flat["advantages"],
        loss_mask=flat["loss_mask"],
        logprob_type="token_level",
        reward_type="action_level",
        action_dim=d,
        layout=layout,
        loss_type="actor_critic",
        values=values,
        returns=flat["returns"],
        prev_values=flat["prev_values"],
    )
    loss_no_ver, m_no_ver = policy_loss(**kwargs)
    assert "proximal_approx_kl" not in m_no_ver
    loss_v1, m_v1 = policy_loss(
        **kwargs, versions=torch.zeros_like(flat["versions"]), current_version=1,
        behave_weight_threshold=2.0,
    )
    assert torch.allclose(loss_no_ver, loss_v1, atol=1e-6)
    assert "proximal_approx_kl" not in m_v1


def test_decoupled_actor_critic_stale_correction():
    """Async PPO (decoupled_actor_critic): versions drive proximal correction."""
    flat, layout = _flat_ppo_batch()
    d = layout.action_dim
    torch.manual_seed(0)
    # Large perturbation so PPO clipping triggers: unclipped samples satisfy
    # behavior_weight * proximal_ratio == plain ratio (decoupled-PPO identity),
    # so only clipped/filtered samples distinguish stale correction.
    logprobs = flat["prev_logprobs"] + 0.5 * torch.randn_like(flat["prev_logprobs"])
    values = flat["prev_values"] + 0.1

    kwargs = dict(
        logprobs=logprobs,
        old_logprobs=flat["prev_logprobs"],
        advantages=flat["advantages"],
        loss_mask=flat["loss_mask"],
        logprob_type="token_level",
        reward_type="action_level",
        action_dim=d,
        layout=layout,
        loss_type="decoupled_actor_critic",
        values=values,
        returns=flat["returns"],
        prev_values=flat["prev_values"],
    )
    loss_no_ver, m_no_ver = policy_loss(**kwargs)
    assert "proximal_approx_kl" not in m_no_ver
    assert "behav_clip_fraction" not in m_no_ver
    assert "value_loss" in m_no_ver
    assert torch.isfinite(loss_no_ver)

    # staleness=0 everywhere -> proximal anchor == old_logprobs == plain PPO
    loss_v0, m_v0 = policy_loss(
        **kwargs, versions=torch.zeros_like(flat["versions"]), current_version=0,
        behave_weight_threshold=2.0,
    )
    assert torch.allclose(loss_no_ver, loss_v0, atol=1e-6)
    assert m_v0["behav_clip_fraction"] == 0.0

    # staleness=1 -> correction active, loss differs but stays finite
    loss_v1, m_v1 = policy_loss(
        **kwargs, versions=torch.zeros_like(flat["versions"]), current_version=1,
        behave_weight_threshold=2.0,
    )
    assert torch.isfinite(loss_v1)
    assert "proximal_approx_kl" in m_v1
    assert not torch.allclose(loss_no_ver, loss_v1)


def test_actor_loss_unaffected():
    """GRPO actor-only path: no critic metrics, identical default behavior."""
    flat, layout = _flat_ppo_batch()
    d = layout.action_dim
    loss, metrics = policy_loss(
        logprobs=flat["prev_logprobs"],
        old_logprobs=flat["prev_logprobs"],
        advantages=flat["advantages"],
        loss_mask=flat["loss_mask"],
        logprob_type="token_level",
        reward_type="action_level",
        action_dim=d,
        layout=layout,
    )
    assert torch.isfinite(loss)
    assert "value_loss" not in metrics
    assert "values_mean" not in metrics
    assert "proximal_approx_kl" not in metrics


if __name__ == "__main__":
    test_gae_shapes()
    test_gae_values_hand_checked()
    test_masked_whiten()
    test_chunk_level_gae_and_actor_critic_loss()
    test_actor_critic_loss_forward_backward()
    test_actor_critic_ignores_versions()
    test_decoupled_actor_critic_stale_correction()
    test_actor_loss_unaffected()
    print("test_ppo_algorithms: OK")
