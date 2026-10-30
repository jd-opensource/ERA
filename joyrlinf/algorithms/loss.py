"""Policy loss functions for embodied RL (PPO-clip style, aligned with RLinf)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from joyrlinf.algorithms.embodied_batch_layout import (
    ActorBatchLayout,
    preprocess_loss_inputs,
)


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean()
    mask_f = mask.float() if mask.dtype == torch.bool else mask
    return (tensor * mask_f).sum() / mask_f.sum().clamp(min=1)


def masked_mean_ratio(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
    loss_mask_ratio: torch.Tensor,
) -> torch.Tensor:
    """Embodied loss agg (RLinf): scale by inverse episode-length ratio."""
    if mask is None:
        return (values / loss_mask_ratio).mean()
    mask_f = mask.float() if mask.dtype == torch.bool else mask
    return (values / loss_mask_ratio * mask_f).mean()


def policy_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio_high: float = 0.2,
    clip_ratio_low: float = 0.2,
    clip_ratio_c: Optional[float] = None,
    clip_log_ratio_min: Optional[float] = None,
    clip_log_ratio_max: Optional[float] = None,
    nan_to_num: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    max_episode_steps: Optional[int] = None,
    entropy: Optional[torch.Tensor] = None,
    entropy_bonus: float = 0.0,
    logprob_type: str = "chunk_level",
    reward_type: Optional[str] = None,
    action_dim: int = 7,
    layout: Optional[ActorBatchLayout] = None,
    loss_type: str = "actor",
    values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    prev_values: Optional[torch.Tensor] = None,
    value_clip: float = 0.2,
    huber_delta: float = 10.0,
    versions: Optional[torch.Tensor] = None,
    current_version: Optional[int] = None,
    behave_weight_threshold: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """PPO-clip policy loss with layout-driven shape alignment.

    ``loss_type="actor"``: clip loss only (synchronous GRPO path, unchanged).
    ``loss_type="decoupled_actor"``: asynchronous GRPO — clip actor loss with a
    proximal-style stale correction but NO critic (GRPO has no value head).
    ``loss_type="actor_critic"``: standard synchronous PPO — clip actor loss
    plus a clipped huber critic loss, strictly on-policy (no stale correction).
    ``loss_type="decoupled_actor_critic"``: asynchronous PPO — same actor+critic
    plus a proximal-style stale correction (decoupled-PPO idea).

    Proximal correction (``decoupled_actor``/``decoupled_actor_critic``): when
    ``versions``/``current_version`` are provided, the clip anchor is
    interpolated from the behavior policy toward the current policy as
    staleness grows, and an importance ``behavior_weight`` reweights/filters
    stale samples. Without versions it degrades to plain PPO/GRPO.
    """
    critic_enabled = loss_type in ("actor_critic", "decoupled_actor_critic")
    decoupled = loss_type in ("decoupled_actor", "decoupled_actor_critic")
    metrics: dict = {}

    if layout is None:
        layout = ActorBatchLayout(
            t_chunk=1,
            batch_size=logprobs.shape[0],
            num_action_chunks=max(1, logprobs.shape[-1] // max(action_dim, 1)),
            action_dim=action_dim,
            reward_type=reward_type or "action_level",
            logprob_type=logprob_type,
        )

    prepped = preprocess_loss_inputs(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        layout=layout,
        loss_mask=loss_mask,
        values=values,
        prev_values=prev_values,
        returns=returns,
        versions=versions,
    )
    logprobs_r = prepped["logprobs"].float()
    old_logprobs_r = prepped["old_logprobs"].float()
    adv = prepped["advantages"].detach().float()
    lm = prepped["loss_mask"]
    lm_sum = prepped["loss_mask_sum"]
    versions_r = prepped["versions"]

    loss_mask_ratio = None
    if (
        max_episode_steps is not None
        and lm_sum is not None
        and lm is not None
    ):
        loss_mask_ratio = lm_sum.float() / float(max_episode_steps)

    def _agg(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if loss_mask_ratio is not None:
            return masked_mean_ratio(values, mask, loss_mask_ratio)
        return masked_mean(values, mask)

    if lm is not None:
        lm_bool = lm.bool()
    else:
        lm_bool = None

    # ---- proximal stale correction (async PPO: decoupled_actor_critic) ----
    proximal_logprobs = None
    behavior_weight = None
    if decoupled and versions_r is not None and current_version is not None:
        staleness = (float(current_version) - versions_r.float()).clamp(min=0.0)
        # staleness=0 -> alpha=0 -> anchor=old_logprobs (plain PPO);
        # staleness=s -> anchor moves s/(s+1) toward the current policy.
        alpha = staleness / (staleness + 1.0)
        proximal_logprobs = (
            old_logprobs_r + alpha * (logprobs_r.detach() - old_logprobs_r)
        )
        behavior_weight = torch.exp(
            (proximal_logprobs - old_logprobs_r).clamp(max=20.0)
        ).detach()
        if behave_weight_threshold is not None and behave_weight_threshold > 0:
            behav_keep = behavior_weight <= float(behave_weight_threshold)
            # Reduce token dims that the loss mask broadcasts over (size-1)
            # so combining masks never changes aggregation denominators.
            if lm_bool is not None and behav_keep.shape != lm_bool.shape:
                for dim in range(behav_keep.dim()):
                    if lm_bool.shape[dim] == 1 and behav_keep.shape[dim] != 1:
                        behav_keep = behav_keep.all(dim=dim, keepdim=True)
            valid_before = (
                lm_bool if lm_bool is not None
                else torch.ones_like(behav_keep)
            )
            n_valid = valid_before.float().sum().clamp(min=1.0)
            metrics["behav_clip_fraction"] = float(
                ((~behav_keep) & valid_before).float().sum().item() / n_valid.item()
            )
            lm_bool = behav_keep if lm_bool is None else (lm_bool & behav_keep)
        else:
            metrics["behav_clip_fraction"] = 0.0

    anchor_logprobs = (
        proximal_logprobs if proximal_logprobs is not None else old_logprobs_r
    )

    log_ratio = logprobs_r - anchor_logprobs
    if clip_log_ratio_min is not None:
        log_ratio = torch.clamp(log_ratio, min=clip_log_ratio_min)
    if clip_log_ratio_max is not None:
        log_ratio = torch.clamp(log_ratio, max=clip_log_ratio_max)
    if lm_bool is not None:
        ratio = torch.where(
            lm_bool,
            torch.exp(log_ratio),
            torch.zeros_like(logprobs_r),
        )
    else:
        ratio = torch.exp(log_ratio)

    def _finite_frac(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
        finite = torch.isfinite(tensor)
        if mask is not None:
            if mask.shape != finite.shape:
                mask = mask.expand_as(finite)
            finite = finite[mask]
        if finite.numel() == 0:
            return 1.0
        return finite.float().mean().item()

    metrics["logprobs_finite_frac"] = _finite_frac(logprobs_r, lm_bool)
    metrics["old_logprobs_finite_frac"] = _finite_frac(old_logprobs_r, lm_bool)
    metrics["ratio_finite_frac"] = _finite_frac(ratio, lm_bool)

    if nan_to_num:
        ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    surr1 = -adv * ratio
    surr2 = -adv * clipped_ratio
    ploss_raw = torch.max(surr1, surr2)

    if clip_ratio_c is not None and clip_ratio_c > 1.0:
        surr3 = torch.sign(adv) * clip_ratio_c * adv
        ploss_raw = torch.min(ploss_raw, surr3)

    if behavior_weight is not None:
        ploss_raw = ploss_raw * behavior_weight

    clip_mask = surr1.detach() < surr2.detach()
    lm_for_agg = lm_bool if lm_bool is not None else None
    ploss = _agg(ploss_raw, lm_for_agg)
    ploss_abs = _agg(ploss_raw.abs(), lm_for_agg)

    with torch.no_grad():
        lr_behav = (logprobs_r - old_logprobs_r).clamp(-20.0, 20.0)
        if lm_bool is not None:
            lr_behav = torch.where(lm_bool, lr_behav, torch.zeros_like(lr_behav))
        metrics["approx_kl"] = masked_mean(
            lr_behav.exp() - 1.0 - lr_behav, lm_for_agg,
        ).item()
        if proximal_logprobs is not None:
            lr_prox = (logprobs_r - proximal_logprobs).clamp(-20.0, 20.0)
            if lm_bool is not None:
                lr_prox = torch.where(lm_bool, lr_prox, torch.zeros_like(lr_prox))
            metrics["proximal_approx_kl"] = masked_mean(
                lr_prox.exp() - 1.0 - lr_prox, lm_for_agg,
            ).item()

    metrics["policy_loss"] = ploss.detach().item()
    metrics["policy_loss_abs"] = ploss_abs.detach().item()
    ratio_det = ratio.detach()
    if lm_bool is not None:
        # lm_bool may be [N,C,1] vs ratio [N,C,A]: expand explicitly so boolean
        # indexing and clip_fraction denominators count every token.
        valid = lm_bool.expand_as(ratio_det)
        valid_ratio = ratio_det[valid] if valid.any() else ratio_det
        ratio_abs = (ratio_det - 1.0).abs()
        valid_ratio_abs = ratio_abs[valid] if valid.any() else ratio_abs
        loss_mask_count = valid.sum().clamp(min=1)
        metrics["clip_fraction"] = (
            (clip_mask & valid).sum().float() / loss_mask_count.float()
        ).detach().item()
        metrics["ratio_mean"] = valid_ratio.mean().item()
        metrics["ratio_max"] = valid_ratio.max().item()
        metrics["ratio_min"] = valid_ratio.min().item()
        metrics["ratio_abs"] = valid_ratio_abs.mean().item()
    else:
        metrics["clip_fraction"] = clip_mask.float().mean().item()
        metrics["ratio_mean"] = ratio_det.mean().item()
        metrics["ratio_max"] = ratio_det.max().item()
        metrics["ratio_min"] = ratio_det.min().item()
        metrics["ratio_abs"] = (ratio_det - 1.0).abs().mean().item()
    metrics["advantages_mean"] = advantages.detach().mean().item()

    total_loss = ploss

    if critic_enabled:
        vs = prepped["values"]
        rets = prepped["returns"]
        pv = prepped["prev_values"]
        if vs is None or rets is None:
            raise ValueError(
                f"loss_type={loss_type!r} requires values and returns "
                f"(got values={'set' if vs is not None else None}, "
                f"returns={'set' if rets is not None else None})"
            )
        vs = vs.float()
        rets = rets.detach().float()
        if pv is not None:
            pv = pv.detach().float()
            vc = pv + torch.clamp(vs - pv, -value_clip, value_clip)
            vf1 = F.huber_loss(vs, rets, delta=huber_delta, reduction="none")
            vf2 = F.huber_loss(vc, rets, delta=huber_delta, reduction="none")
            vloss = _agg(torch.max(vf1, vf2), lm_for_agg)
        else:
            vloss = _agg(
                F.huber_loss(vs, rets, delta=huber_delta, reduction="none"),
                lm_for_agg,
            )
        total_loss = total_loss + 0.5 * vloss
        metrics["value_loss"] = vloss.detach().item()
        metrics["values_mean"] = _agg(vs.detach(), lm_for_agg).item()
        metrics["returns_mean"] = _agg(rets, lm_for_agg).item()

    if entropy is not None and entropy_bonus > 0:
        ent_flat = entropy.reshape(entropy.shape[0], -1).mean(dim=-1)
        emean = masked_mean(ent_flat, loss_mask)
        total_loss = total_loss - entropy_bonus * emean
        metrics["entropy"] = emean.detach().item()

    metrics["total_loss"] = total_loss.detach().item()
    return total_loss, metrics
