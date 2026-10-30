"""Early validation for embodied training configs.

Run at process start (after ``torch.distributed`` init) so placement / world
size mismatches and batch arithmetic errors fail fast instead of after rollout.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.distributed as dist

from joyrlinf.core.launcher import (
    parse_placement,
    _get_nproc_per_node,
    _group_ranks_by_node,
    _resolve_gpus_per_node,
)
from joyrlinf.core.actor_batch import resolve_actor_batch

logger = logging.getLogger(__name__)


def _nproc_per_node_for_config(cfg: Any, placement: Any = None) -> int:
    """Machines-local GPU count for placement grouping (especially comm_mode v2).

    Under ``torchrun``, ``LOCAL_WORLD_SIZE`` / ``dist`` supply the true value.
    For offline ``validate_embodied_config.py``, infer from YAML or from a
    contiguous single-node rank layout (e.g. global ranks 0–7 → 8 GPUs/node).
    """
    c = getattr(cfg, "cluster", None)
    if c is not None:
        basis = str(getattr(c, "placement_basis", "global") or "global").lower().strip()
        if basis == "per_node":
            return _resolve_gpus_per_node(c)

    n = _get_nproc_per_node()
    if n > 1:
        return n

    if placement is not None and c is not None:
        num_nodes = int(getattr(c, "num_nodes", 1) or 1)
        all_ranks = sorted(
            set(placement.env_ranks)
            | set(placement.rollout_ranks)
            | set(placement.actor_ranks)
        )
        if num_nodes == 1 and all_ranks:
            max_rank = max(all_ranks)
            if all_ranks == list(range(max_rank + 1)):
                return max_rank + 1

    return n


def validate_embodied_training_config(
    cfg: Any, *, check_world_ranks: bool | None = None,
) -> None:
    """Raise ``ValueError`` if common embodied + comm settings are inconsistent.

    Args:
        check_world_ranks: If ``True``, require placement union to match
            ``dist.get_world_size()`` and ranks ``0..ws-1``. If ``None``, auto:
            run this check only when ``torch.distributed`` is initialized.

    Checks (non-exhaustive but covers frequent foot-guns):

    - ``runner.task_type == embodied`` (otherwise no-op).
    - Placement ranks are unique and cover ``0 .. world_size-1`` when
      ``check_world_ranks`` is enabled and distributed is initialized.
    - ``total_num_envs`` divisible by rollout GPU count.
    - ``gbs % actor_ws == 0`` for well-defined per-actor logical batch.
    - GRPO: actor batch width divisible by ``group_size``.
    - **comm_mode v2**: each node must have both rollout and actor ranks;
      envs per rollout GPU divisible by per-node actor count; per-actor
      batch consistent across nodes for FSDP correctness.
    - **comm_mode v1**: same divisibility with ``B_actor = envs per rollout GPU``.
    """
    if str(cfg.runner.get("task_type", "embodied")) != "embodied":
        return

    if not hasattr(cfg, "cluster") or not hasattr(cfg.cluster, "component_placement"):
        raise ValueError("embodied config: missing ``cluster.component_placement``.")
    placement = parse_placement(cfg.cluster)
    n_env = len(placement.env_ranks)
    n_roll = len(placement.rollout_ranks)
    n_act = len(placement.actor_ranks)
    pipeline_mode = str(cfg.algorithm.get("pipeline_mode", "sync")).lower().strip()
    if pipeline_mode not in ("sync", "async"):
        raise ValueError(
            f"algorithm.pipeline_mode must be 'sync' or 'async', got {pipeline_mode!r}."
        )
    if pipeline_mode == "sync" and not placement.sync_colocated:
        raise ValueError(
            "pipeline_mode: sync requires env == rollout == actor on the same ranks "
            "(full-GPU temporal reuse). For spatial split (e.g. rollout 0-5, actor 6-7) "
            "or disaggregated placement, use pipeline_mode: async."
        )
    if n_roll == 0 or n_act == 0:
        raise ValueError(
            "embodied config: need at least one rollout rank and one actor rank; "
            f"got rollout_ranks={placement.rollout_ranks} actor_ranks={placement.actor_ranks}"
        )

    if check_world_ranks is None:
        check_world_ranks = dist.is_initialized()
    if check_world_ranks:
        if not dist.is_initialized():
            raise RuntimeError(
                "validate_embodied_training_config(..., check_world_ranks=True) "
                "requires torch.distributed to be initialized."
            )
        ws = dist.get_world_size()
        all_ranks = sorted(
            set(placement.env_ranks)
            | set(placement.rollout_ranks)
            | set(placement.actor_ranks)
        )
        if len(all_ranks) != ws or all_ranks != list(range(ws)):
            raise ValueError(
                "embodied config: component_placement must list each global rank "
                f"0..{ws - 1} exactly once (union of env, rollout, actor). "
                f"world_size={ws}, union(sorted)={all_ranks}"
            )

    total_envs = int(cfg.env.train.total_num_envs)
    if total_envs <= 0:
        raise ValueError(f"env.train.total_num_envs must be > 0, got {total_envs}")
    if total_envs % n_roll != 0:
        raise ValueError(
            f"total_num_envs ({total_envs}) must be divisible by number of "
            f"rollout GPUs ({n_roll}) so each rollout sees the same env count."
        )

    envs_per_rollout = total_envs // n_roll

    if not hasattr(cfg, "actor"):
        raise ValueError(
            "embodied config: missing ``actor`` section "
            "(gradient_accumulation_steps, micro_batch_size)."
        )
    batch_cfg = resolve_actor_batch(cfg.actor, n_act)
    mbs = batch_cfg.micro_batch_size
    ga = batch_cfg.gradient_accumulation_steps
    per_actor = batch_cfg.per_actor_batch
    gbs = batch_cfg.effective_global_batch
    actor_ws = n_act
    if mbs <= 0 or ga < 1:
        raise ValueError(
            f"actor.micro_batch_size and gradient_accumulation_steps must be "
            f"positive; got mbs={mbs}, ga={ga}"
        )

    comm_mode = str(cfg.algorithm.get("comm_mode", "v1"))
    rollout_epoch = int(cfg.algorithm.get("rollout_epoch", 1))
    t_steps = int(cfg.env.train.max_steps_per_rollout_epoch)
    if rollout_epoch < 1 or t_steps < 1:
        raise ValueError(
            f"algorithm.rollout_epoch ({rollout_epoch}) and "
            f"env.train.max_steps_per_rollout_epoch ({t_steps}) must be >= 1."
        )
    gs = int(cfg.algorithm.get("group_size", 8))
    adv_type = str(cfg.algorithm.get("adv_type", "grpo")).lower().strip()
    loss_type = str(cfg.algorithm.get("loss_type", "actor")).lower().strip()
    filter_rewards = bool(cfg.algorithm.get("filter_rewards", False))
    reward_type = str(cfg.algorithm.get("reward_type", "action_level")).lower().strip()
    if reward_type not in ("action_level", "chunk_level"):
        raise ValueError(
            f"algorithm.reward_type must be 'action_level' or 'chunk_level', "
            f"got {reward_type!r}."
        )
    num_action_chunks = int(cfg.actor.model.get("num_action_chunks", 1))
    if num_action_chunks < 1:
        raise ValueError(
            f"actor.model.num_action_chunks must be >= 1, got {num_action_chunks}."
        )
    if t_steps % num_action_chunks != 0:
        raise ValueError(
            f"env.train.max_steps_per_rollout_epoch ({t_steps}) must be divisible "
            f"by actor.model.num_action_chunks ({num_action_chunks}); rollout "
            "collects whole action chunks only."
        )
    # Actor training flattens by chunk step, not by individual env sub-step.
    # A row contains either C action-level targets (OpenVLA PPO) or one
    # chunk-level scalar target (OpenPI/Pi0.5 PPO/GRPO).
    t_eff = (t_steps // num_action_chunks) * rollout_epoch

    if adv_type == "grpo":
        # Sync GRPO is on-policy (actor). Async GRPO may add proximal stale
        # correction via decoupled_actor (still no critic / value head).
        allowed_grpo = (
            ("actor",) if pipeline_mode == "sync"
            else ("actor", "decoupled_actor")
        )
        if loss_type not in allowed_grpo:
            raise ValueError(
                f"adv_type=grpo with pipeline_mode={pipeline_mode} requires "
                f"loss_type in {allowed_grpo} (no critic); "
                f"got loss_type={loss_type!r}."
            )
    elif adv_type == "gae":
        # Sync PPO is strictly on-policy (actor_critic); async PPO uses the
        # staleness/proximal stale-correction path (decoupled_actor_critic).
        expected_loss = (
            "actor_critic" if pipeline_mode == "sync" else "decoupled_actor_critic"
        )
        if loss_type != expected_loss:
            raise ValueError(
                f"adv_type=gae (PPO) with pipeline_mode={pipeline_mode} requires "
                f"loss_type={expected_loss!r}; got loss_type={loss_type!r}."
            )
        if not bool(cfg.actor.model.get("add_value_head", False)):
            raise ValueError(
                "adv_type=gae (PPO) requires actor.model.add_value_head: true "
                "so rollout produces prev_values for GAE."
            )
        if int(cfg.algorithm.get("rollout_epoch", 1)) != 1:
            raise ValueError(
                "adv_type=gae requires rollout_epoch=1: prev_values carries a "
                "bootstrap row per epoch and multi-epoch merge would corrupt "
                "the [T+1,B,C] layout."
            )
    else:
        raise ValueError(
            f"algorithm.adv_type must be 'grpo' or 'gae', got {adv_type!r}."
        )

    if pipeline_mode == "sync":
        b_actor = envs_per_rollout
    elif comm_mode == "v2":
        nproc = _nproc_per_node_for_config(cfg, placement)
        rollout_by_node = _group_ranks_by_node(placement.rollout_ranks, nproc)
        actor_by_node = _group_ranks_by_node(placement.actor_ranks, nproc)
        node_ids = sorted(set(rollout_by_node) | set(actor_by_node))
        for nid in node_ids:
            nr_node = rollout_by_node.get(nid, [])
            na_node = actor_by_node.get(nid, [])
            if not nr_node or not na_node:
                raise ValueError(
                    f"comm_mode=v2 node {nid}: need both rollout and actor ranks "
                    f"on the same node for node-local all2all; got rollout={nr_node} actor={na_node}"
                )
            n_roll_node = len(nr_node)
            n_act_node = len(na_node)
            envs_per_rollout_node = total_envs // n_roll
            if envs_per_rollout_node % n_act_node != 0:
                raise ValueError(
                    f"comm_mode=v2 node {nid}: envs per rollout GPU "
                    f"({envs_per_rollout_node}) must be divisible by per-node "
                    f"n_actors ({n_act_node}) so each actor gets an equal slice."
                )
        b_actors_per_node = {}
        for nid in node_ids:
            nr = len(rollout_by_node[nid])
            na = len(actor_by_node[nid])
            b_actors_per_node[nid] = nr * (envs_per_rollout // na)
        b_values = list(b_actors_per_node.values())
        if len(set(b_values)) > 1:
            raise ValueError(
                f"comm_mode=v2: per-actor batch size differs across nodes "
                f"({b_actors_per_node}). Each node must yield the same "
                f"B_actor so FSDP gradient all-reduce is well-defined. "
                f"Adjust placement so n_rollout/n_actor ratio is uniform."
            )
        b_actor = b_values[0]
    else:
        b_actor = envs_per_rollout

    # Group divisibility only matters when groups are actually used:
    # GRPO score reshaping, or the group-based reward filter.
    if (adv_type == "grpo" or filter_rewards) and b_actor % gs != 0:
        raise ValueError(
            f"adv_type={adv_type} filter_rewards={filter_rewards}: effective "
            f"batch width on each actor ({b_actor} envs) must be divisible by "
            f"algorithm.group_size ({gs}) for group reshaping."
        )

    total_flat = t_eff * b_actor
    if total_flat % per_actor != 0:
        raise ValueError(
            f"Flattened training samples per actor ({total_flat} = "
            f"T_eff {t_eff} × B_actor {b_actor}) must be divisible by "
            f"per-actor logical batch (mbs {mbs} × ga {ga} = {per_actor}). "
            f"comm_mode={comm_mode}. Adjust total_num_envs, max_steps_per_rollout_epoch, "
            f"rollout_epoch, gradient_accumulation_steps, micro_batch_size, or placement."
        )

    val_interval = int(cfg.runner.get("val_check_interval", -1))
    if val_interval > 0:
        eval_section = getattr(cfg.env, "eval", None)
        if eval_section is None:
            raise ValueError(
                "runner.val_check_interval > 0 requires env.eval section "
                "(with total_num_envs divisible by rollout GPU count)."
            )
        eval_total = int(eval_section.total_num_envs)
        if eval_total <= 0:
            raise ValueError(
                f"env.eval.total_num_envs must be > 0, got {eval_total}"
            )
        if eval_total % n_roll != 0:
            raise ValueError(
                f"env.eval.total_num_envs ({eval_total}) must be divisible by "
                f"rollout GPUs ({n_roll})."
            )
        save_interval = int(cfg.runner.get("save_interval", -1))
        if save_interval > 0 and save_interval % val_interval != 0:
            raise ValueError(
                f"runner.save_interval ({save_interval}) must be divisible by "
                f"runner.val_check_interval ({val_interval})."
            )

    from joyrlinf.utils.logging_setup import GLOBAL_LOG_LEADER, current_rank, log_at_rank

    log_at_rank(
        logger, current_rank(), GLOBAL_LOG_LEADER, logging.INFO,
        "Config validation OK: pipeline=%s comm_mode=%s n_roll=%d n_act=%d envs/rollout=%d "
        "B_actor=%d T_eff=%d total_flat=%d gbs=%d mbs=%d per_actor=%d ga=%d",
        pipeline_mode, comm_mode, n_roll, n_act, envs_per_rollout, b_actor, t_eff, total_flat,
        gbs, mbs, per_actor, ga,
    )
