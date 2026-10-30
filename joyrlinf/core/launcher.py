"""Process launcher and placement utilities.

Designed for torchrun: each rank is assigned a role based on the
component_placement config. Supports both co-located (env+rollout same GPU)
and fully disaggregated (env, rollout, actor on separate GPUs) modes.
Homogeneous clusters can set ``cluster.placement_basis: per_node`` to describe
GPU layout once per machine instead of repeating global ranks.
No Ray dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import torch
import torch.distributed as dist

from joyrlinf.utils.logging_setup import GLOBAL_LOG_LEADER, current_rank, log_at_rank

logger = logging.getLogger(__name__)


class Role(Enum):
    ENV = auto()      # standalone env simulation (disaggregated mode)
    ROLLOUT = auto()  # rollout inference (co-located with env, or standalone)
    ACTOR = auto()    # training


@dataclass
class Placement:
    """Parsed placement configuration."""
    env_ranks: list[int] = field(default_factory=list)
    rollout_ranks: list[int] = field(default_factory=list)
    actor_ranks: list[int] = field(default_factory=list)

    @property
    def colocated(self) -> bool:
        """Whether env and rollout are on the same GPUs."""
        return set(self.env_ranks) == set(self.rollout_ranks)

    @property
    def sync_colocated(self) -> bool:
        """Whether env, rollout, and actor share the same rank set (sync mode)."""
        s_env = set(self.env_ranks)
        s_roll = set(self.rollout_ranks)
        s_act = set(self.actor_ranks)
        return s_env == s_roll == s_act and len(s_env) > 0

    @property
    def num_rollout(self) -> int:
        return len(self.rollout_ranks)

    @property
    def num_actor(self) -> int:
        return len(self.actor_ranks)

    def get_paired_actor(self, rollout_rank: int) -> int:
        """Get the first actor rank paired with a rollout rank."""
        idx = self.rollout_ranks.index(rollout_rank)
        return self.actor_ranks[idx % len(self.actor_ranks)]

    def get_all_paired_actors(self, rollout_rank: int) -> list[int]:
        """Get all actor ranks paired with a rollout rank.

        With N rollouts and M actors (M >= N), each rollout feeds
        actors whose index mod N equals this rollout's index.
        """
        r_idx = self.rollout_ranks.index(rollout_rank)
        return [a for i, a in enumerate(self.actor_ranks)
                if i % len(self.rollout_ranks) == r_idx]

    def get_paired_rollout(self, actor_rank: int) -> int:
        """Get the rollout rank paired with an actor rank."""
        idx = self.actor_ranks.index(actor_rank)
        return self.rollout_ranks[idx % len(self.rollout_ranks)]

    def get_paired_rollout_for_env(self, env_rank: int) -> int:
        """Get the rollout rank paired with an env rank (disaggregated mode)."""
        idx = self.env_ranks.index(env_rank)
        return self.rollout_ranks[idx % len(self.rollout_ranks)]

    def get_paired_env_for_rollout(self, rollout_rank: int) -> int:
        """Get the env rank paired with a rollout rank (disaggregated mode)."""
        idx = self.rollout_ranks.index(rollout_rank)
        return self.env_ranks[idx % len(self.env_ranks)]


def _parse_range(spec: str) -> list[int]:
    """Parse '0-3' or '0,1,2,3' into a list of ints."""
    result = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return result


def _resolve_gpus_per_node(cluster_cfg) -> int:
    """Infer GPUs per machine for tiling (``torchrun`` / config).

    Order: explicit ``gpus_per_node`` in YAML ``LOCAL_WORLD_SIZE`` env,
    then ``dist.get_world_size() // cluster.num_nodes`` when both are known.
    """
    import os

    g = getattr(cluster_cfg, "gpus_per_node", None)
    if g is not None:
        return int(g)
    lws = os.environ.get("LOCAL_WORLD_SIZE")
    if lws is not None:
        return int(lws)
    if dist.is_initialized():
        nn = getattr(cluster_cfg, "num_nodes", None)
        ws = dist.get_world_size()
        if nn is not None:
            nn = int(nn)
            if ws % nn != 0:
                raise ValueError(
                    f"Cannot infer gpus_per_node: world_size={ws} not divisible "
                    f"by cluster.num_nodes={nn}; set cluster.gpus_per_node or "
                    f"launch with LOCAL_WORLD_SIZE."
                )
            return ws // nn
    raise ValueError(
        "placement_basis=per_node requires gpus_per_node (YAML), LOCAL_WORLD_SIZE "
        "(torchrun), or (torch.distributed + cluster.num_nodes) to infer tile size."
    )


def _resolve_num_nodes_tiled(cluster_cfg, *, ws: int, nproc: int) -> int:
    """Infer how many identical nodes replicate ``component_placement``.

    Prefer ``torch.distributed`` world_size / ``gpus_per_node``. If cluster
    ``num_nodes`` is set, it must match the inferred count.
    """
    if ws % nproc != 0:
        raise ValueError(
            f"world_size ({ws}) must be divisible by gpus_per_node ({nproc}) "
            "for homogeneous tiling."
        )
    inferred = ws // nproc
    cfg_nn = getattr(cluster_cfg, "num_nodes", None)
    if cfg_nn is not None and int(cfg_nn) != inferred:
        raise ValueError(
            f"cluster.num_nodes={cfg_nn} does not match world_size/nproc="
            f"{inferred} (world_size={ws}, gpus_per_node={nproc}). "
            "Fix num_nodes or the torchrun topology."
        )
    return inferred


def _maybe_expand_per_node(
    *,
    cluster_cfg,
    raw: Placement,
) -> Placement:
    """Duplicate *local* rank layout across nodes into global ranks."""
    basis = str(getattr(cluster_cfg, "placement_basis", "global") or "global")
    basis = basis.lower().strip()
    if basis in ("global", ""):
        return raw
    if basis != "per_node":
        raise ValueError(
            f"cluster.placement_basis must be 'global' or 'per_node', got {basis!r}."
        )

    # Local rank sets must lie on one node and not wrap.
    nproc = _resolve_gpus_per_node(cluster_cfg)
    for role, ranks in (
        ("env", raw.env_ranks),
        ("rollout", raw.rollout_ranks),
        ("actor", raw.actor_ranks),
    ):
        for r in ranks:
            if r < 0 or r >= nproc:
                raise ValueError(
                    f"placement_basis=per_node: {role} local rank {r} out of "
                    f"range [0, {nproc - 1}] for gpus_per_node={nproc}."
                )

    num_nodes = None
    ws = None
    if dist.is_initialized():
        ws = dist.get_world_size()
        num_nodes = _resolve_num_nodes_tiled(cluster_cfg, ws=ws, nproc=nproc)
    else:
        nn = getattr(cluster_cfg, "num_nodes", None)
        if nn is None:
            raise ValueError(
                "placement_basis=per_node without torch.distributed: set "
                "cluster.num_nodes (and cluster.gpus_per_node) so placement "
                "can be expanded for offline validation."
            )
        num_nodes = int(nn)

    def _tile(side: list[int]) -> list[int]:
        acc: list[int] = []
        for nid in range(int(num_nodes)):
            base = nid * nproc
            acc.extend(r + base for r in sorted(side))
        acc.sort()
        return acc

    expanded = Placement(
        env_ranks=_tile(raw.env_ranks),
        rollout_ranks=_tile(raw.rollout_ranks),
        actor_ranks=_tile(raw.actor_ranks),
    )
    if logger.isEnabledFor(logging.INFO):
        from joyrlinf.utils.logging_setup import GLOBAL_LOG_LEADER, current_rank, log_at_rank

        log_at_rank(
            logger, current_rank(), GLOBAL_LOG_LEADER, logging.INFO,
            "placement_basis=per_node: tiled %d nodes × %d GPUs/node → ws=%s "
            "env=%s rollout=%s actor=%s",
            num_nodes,
            nproc,
            ws if dist.is_initialized() else "(offline)",
            expanded.env_ranks,
            expanded.rollout_ranks,
            expanded.actor_ranks,
        )
    return expanded


def parse_placement(cluster_or_placement_cfg) -> Placement:
    """Parse cluster placement into ranks.

    **Two call shapes**

    - **Full cluster** (recommended): pass ``cfg.cluster``. Honor optional
      ``cluster.placement_basis``:

      - ``global`` (default): ``component_placement`` lists **torchrun global**
        ranks, same as before.
      - ``per_node``: ``env`` / ``rollout`` / ``actor`` strings describe **local**
        ranks on one machine only (0 … ``gpus_per_node``-1). They are replicated
        for ``num_nodes`` machines. Needs ``torch.distributed`` after launch
        unless you validate offline — then set ``num_nodes`` and
        ``gpus_per_node`` explicitly.

    - **Bare ``component_placement``** (backward compatible): behaves as
      ``placement_basis=global``; cannot use ``per_node`` tiling — pass the full
      ``cluster`` object instead.
    """
    if hasattr(cluster_or_placement_cfg, "component_placement"):
        cluster_cfg = cluster_or_placement_cfg
        placement_cfg = cluster_cfg.component_placement
        raw = Placement(
            env_ranks=_parse_range(placement_cfg.env),
            rollout_ranks=_parse_range(placement_cfg.rollout),
            actor_ranks=_parse_range(placement_cfg.actor),
        )
        return _maybe_expand_per_node(cluster_cfg=cluster_cfg, raw=raw)

    placement_cfg = cluster_or_placement_cfg
    return Placement(
        env_ranks=_parse_range(placement_cfg.env),
        rollout_ranks=_parse_range(placement_cfg.rollout),
        actor_ranks=_parse_range(placement_cfg.actor),
    )


def get_role(rank: int, placement: Placement) -> Role:
    in_env = rank in placement.env_ranks
    in_rollout = rank in placement.rollout_ranks
    in_actor = rank in placement.actor_ranks

    if in_env and in_rollout:
        return Role.ROLLOUT
    if in_env and not in_rollout:
        return Role.ENV
    if in_rollout:
        return Role.ROLLOUT
    if in_actor:
        return Role.ACTOR
    raise ValueError(f"Rank {rank} not in any placement group")


def init_distributed() -> tuple[int, int, torch.device]:
    """Initialize torch.distributed and return (rank, world_size, device).

    Expects CUDA_VISIBLE_DEVICES to be set per-process (by train.py) so that
    each rank sees exactly one GPU as cuda:0.

    If the default process group was already initialized (e.g. by scripts/train.py),
    this only sets device and returns rank / world_size.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    from joyrlinf.utils.logging_setup import GLOBAL_LOG_LEADER, log_at_rank

    log_at_rank(
        logger, rank, GLOBAL_LOG_LEADER, logging.INFO,
        "Rank %d/%d on %s", rank, world_size, device,
    )
    logger.debug("Distributed init: rank=%d world=%d device=%s", rank, world_size, device)
    return rank, world_size, device


def _get_nproc_per_node() -> int:
    """Return LOCAL_WORLD_SIZE (GPUs per node). Falls back to world_size."""
    import os
    lws = os.environ.get("LOCAL_WORLD_SIZE")
    if lws is not None:
        return int(lws)
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _group_ranks_by_node(
    ranks: list[int], nproc_per_node: int,
) -> dict[int, list[int]]:
    """Partition *ranks* into ``{node_id: sorted_ranks_on_that_node}``."""
    by_node: dict[int, list[int]] = {}
    for r in ranks:
        nid = r // nproc_per_node
        by_node.setdefault(nid, []).append(r)
    for v in by_node.values():
        v.sort()
    return by_node


def create_process_groups(
    placement: Placement,
    *,
    comm_mode: str = "v1",
    pipeline_mode: str = "sync",
) -> dict[str, dist.ProcessGroup]:
    """Create NCCL process groups for different communication patterns.

    Args:
        comm_mode: ``"v1"`` uses per-pair NCCL groups for rollout->actor
            transfer.  ``"v2"`` creates **per-node**
            ``rollout_actor_all2all`` NCCL groups so that batch exchange
            stays within a single machine while FSDP all-reduce in the
            global ``actor`` group handles cross-node gradient sync.
        pipeline_mode: ``"sync"`` with ``placement.sync_colocated`` uses a
            minimal group set (no batch P2P / weight broadcast). ``"async"``
            creates full comm groups for spatial split or disaggregated runs.
    """
    if (
        not placement.colocated
        and len(placement.env_ranks) != len(placement.rollout_ranks)
    ):
        raise ValueError(
            "Fully disaggregated env/rollout currently requires a 1:1 rank "
            "mapping; got env_ranks="
            f"{placement.env_ranks}, rollout_ranks={placement.rollout_ranks}. "
            "Use equal-sized rank sets (for example env=0-1, rollout=2-3)."
        )
    groups = {}
    all_ranks = sorted(set(
        placement.env_ranks + placement.rollout_ranks + placement.actor_ranks
    ))
    sync_minimal = pipeline_mode == "sync" and placement.sync_colocated

    # Actor-only group for FSDP all-reduce (cross-node)
    if placement.actor_ranks:
        groups["actor"] = dist.new_group(ranks=placement.actor_ranks, backend="nccl")

    # All rollout ranks (for aggregating per-env stats like success counts).
    # Gloo: only tiny scalars are exchanged (success counts, eval metrics).
    # NCCL with socket transport across 24+ ranks wastes tens of GB GPU memory
    # on proxy/ring buffers that are never used for large tensor traffic.
    if placement.rollout_ranks:
        groups["rollout"] = dist.new_group(
            ranks=sorted(placement.rollout_ranks), backend="gloo"
        )
    else:
        groups["rollout"] = None

    if sync_minimal:
        groups["rollout_actor_all2all"] = None
        groups["_a2a_node_groups"] = {}
        groups["weight_broadcast"] = None
        groups["_wb_node_groups"] = {}
        groups["shutdown"] = dist.new_group(ranks=all_ranks, backend="gloo")
        log_at_rank(
            logger, current_rank(), GLOBAL_LOG_LEADER, logging.INFO,
            "pipeline_mode=sync (colocated): minimal process groups "
            "(actor + rollout + shutdown only)",
        )
        return groups

    rollout_sorted = sorted(placement.rollout_ranks)
    actor_sorted = sorted(placement.actor_ranks)

    if comm_mode == "v2" and rollout_sorted and actor_sorted:
        nproc = _get_nproc_per_node()
        rollout_by_node = _group_ranks_by_node(rollout_sorted, nproc)
        actor_by_node = _group_ranks_by_node(actor_sorted, nproc)
        node_ids = sorted(set(rollout_by_node) | set(actor_by_node))

        groups["rollout_actor_all2all"] = None
        groups["_a2a_node_groups"] = {}

        for nid in node_ids:
            nr = rollout_by_node.get(nid, [])
            na = actor_by_node.get(nid, [])
            if not nr or not na:
                logger.warning(
                    "comm_mode=v2 node %d: rollout=%s actor=%s — "
                    "skipping (need both on same node for node-local all2all)",
                    nid, nr, na,
                )
                continue
            a2a_ranks = nr + na
            pg = dist.new_group(ranks=a2a_ranks, backend="nccl")
            groups["_a2a_node_groups"][nid] = {
                "group": pg,
                "ranks": a2a_ranks,
                "n_rollouts": len(nr),
                "n_actors": len(na),
            }
            log_at_rank(
                logger, current_rank(), GLOBAL_LOG_LEADER, logging.INFO,
                "comm_mode=v2 node %d: created node-local all2all group "
                "ranks=%s (rollout[0:%d] actor[%d:%d])",
                nid, a2a_ranks, len(nr), len(nr), len(a2a_ranks),
            )
    else:
        groups["rollout_actor_all2all"] = None
        groups["_a2a_node_groups"] = {}

    if comm_mode != "v2":
        # v1: Rollout-Actor transfer pairs (one NCCL group per pair)
        for a_rank in placement.actor_ranks:
            r_rank = placement.get_paired_rollout(a_rank)
            pair_ranks = sorted(set([r_rank, a_rank]))
            if len(pair_ranks) > 1:
                key = f"pair_{r_rank}_{a_rank}"
                if key not in groups:
                    groups[key] = dist.new_group(
                        ranks=pair_ranks, backend="nccl"
                    )

    # Weight broadcast: each node's first actor broadcasts to that node's
    # rollouts.  After FSDP all-reduce all actors hold identical weights,
    # so per-node broadcast avoids cross-network weight transfer entirely.
    # Uses gloo backend to avoid CUDA stream contention.
    nproc_wb = _get_nproc_per_node()
    rollout_by_node_wb = _group_ranks_by_node(placement.rollout_ranks, nproc_wb)
    actor_by_node_wb = _group_ranks_by_node(placement.actor_ranks, nproc_wb)
    wb_node_ids = sorted(set(rollout_by_node_wb) | set(actor_by_node_wb))
    groups["_wb_node_groups"] = {}
    for nid in wb_node_ids:
        nr = rollout_by_node_wb.get(nid, [])
        na = actor_by_node_wb.get(nid, [])
        if not nr or not na:
            continue
        src_actor = na[0]
        bcast_ranks = sorted(set(nr + [src_actor]))
        if len(bcast_ranks) > 1:
            pg = dist.new_group(ranks=bcast_ranks, backend="gloo")
            groups["_wb_node_groups"][nid] = {
                "group": pg,
                "src_actor": src_actor,
                "ranks": bcast_ranks,
            }
            log_at_rank(
                logger, current_rank(), GLOBAL_LOG_LEADER, logging.INFO,
                "weight_broadcast node %d: src_actor=%d ranks=%s (gloo)",
                nid, src_actor, bcast_ranks,
            )
    groups["weight_broadcast"] = None

    # Env-Rollout transfer pairs (disaggregated mode only)
    if not placement.colocated:
        for i, e_rank in enumerate(placement.env_ranks):
            r_rank = placement.rollout_ranks[i % len(placement.rollout_ranks)]
            pair_ranks = sorted(set([e_rank, r_rank]))
            if len(pair_ranks) > 1:
                groups[f"env_rollout_pair_{e_rank}_{r_rank}"] = dist.new_group(
                    ranks=pair_ranks, backend="nccl"
                )

    # Gloo group for shutdown barrier (all ranks).
    # Multi-node: default PG is already Gloo, but we keep this explicit group
    # so single-node (NCCL default) also avoids the 600s NCCL watchdog that
    # aborts env ranks finishing long before actors in async mode.
    groups["shutdown"] = dist.new_group(ranks=all_ranks, backend="gloo")

    return groups
