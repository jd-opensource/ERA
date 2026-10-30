"""Resolve actor micro-batching and gradient accumulation from config."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActorBatchConfig:
    micro_batch_size: int
    gradient_accumulation_steps: int
    per_actor_batch: int
    effective_global_batch: int


def resolve_actor_batch(actor_cfg: Any, actor_world_size: int) -> ActorBatchConfig:
    """Derive batch sizes from ``micro_batch_size`` + ``gradient_accumulation_steps``.

    ``per_actor_batch = micro_batch_size * gradient_accumulation_steps``
    ``effective_global_batch = per_actor_batch * actor_world_size``

    ``global_batch_size`` is accepted as a deprecated alias and converted with a
    warning.
    """
    if actor_world_size < 1:
        raise ValueError(f"actor_world_size must be >= 1, got {actor_world_size}")

    mbs = int(actor_cfg.micro_batch_size)
    if mbs <= 0:
        raise ValueError(f"actor.micro_batch_size must be > 0, got {mbs}")

    ga_raw = actor_cfg.get("gradient_accumulation_steps", None)
    gbs_raw = actor_cfg.get("global_batch_size", None)

    if ga_raw is not None:
        ga = int(ga_raw)
        if ga < 1:
            raise ValueError(
                f"actor.gradient_accumulation_steps must be >= 1, got {ga}"
            )
        per_actor = ga * mbs
        gbs = per_actor * actor_world_size
    elif gbs_raw is not None:
        gbs = int(gbs_raw)
        if gbs <= 0:
            raise ValueError(f"actor.global_batch_size must be > 0, got {gbs}")
        if gbs % actor_world_size != 0:
            raise ValueError(
                f"actor.global_batch_size ({gbs}) must be divisible by actor world "
                f"size ({actor_world_size}); prefer gradient_accumulation_steps."
            )
        per_actor = gbs // actor_world_size
        ga = max(1, per_actor // mbs)
        logger.warning(
            "actor.global_batch_size is deprecated; set "
            "gradient_accumulation_steps=%d with micro_batch_size=%d instead "
            "(per-actor batch %d).",
            ga,
            mbs,
            per_actor,
        )
    else:
        raise ValueError(
            "actor section must set gradient_accumulation_steps (and micro_batch_size)."
        )

    if mbs > per_actor:
        raise ValueError(
            f"actor.micro_batch_size ({mbs}) must be <= per-actor batch "
            f"(micro_batch_size * gradient_accumulation_steps = {per_actor})."
        )

    return ActorBatchConfig(
        micro_batch_size=mbs,
        gradient_accumulation_steps=ga,
        per_actor_batch=per_actor,
        effective_global_batch=gbs,
    )
