"""Canonical tensor layouts for embodied actor training (rollout → flat batch → loss)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

# Rollout canonical: [T_chunk, B, C, ...]
# chunk_level training sample: one (chunk_step, env) → scalar fields [N]
# action_level training sample: one (chunk_step, env) → per-action-chunk [N, C]

_SCALAR_CHUNK_FIELDS = frozenset({
    "advantages",
    "loss_mask",
    "returns",
    "values",
    "prev_values",
    "versions",
})


@dataclass(frozen=True)
class ActorBatchLayout:
    """Single source of truth for embodied actor tensor shapes."""

    t_chunk: int
    batch_size: int
    num_action_chunks: int
    action_dim: int
    reward_type: str
    logprob_type: str

    @property
    def n_samples(self) -> int:
        return self.t_chunk * self.batch_size

    @property
    def is_chunk_level(self) -> bool:
        return self.reward_type == "chunk_level"

    @property
    def is_action_level(self) -> bool:
        return (
            self.reward_type == "action_level"
            and self.num_action_chunks > 1
        )

    def expected_flat_shape(self, field: str, trailing: tuple[int, ...]) -> tuple[int, ...]:
        if self.is_chunk_level and field in _SCALAR_CHUNK_FIELDS:
            return (self.n_samples,)
        return (self.n_samples, *trailing)

    def flatten_field(self, field: str, tensor: torch.Tensor) -> torch.Tensor:
        """[T, B, ...] → flat training layout for ``field``."""
        if tensor.shape[0] != self.t_chunk or tensor.shape[1] != self.batch_size:
            raise ValueError(
                f"Cannot flatten {field}: expected leading dims "
                f"[{self.t_chunk}, {self.batch_size}], got {tuple(tensor.shape[:2])}"
            )
        n = self.n_samples
        if self.is_chunk_level and field in _SCALAR_CHUNK_FIELDS:
            if tensor.dim() == 2:
                return tensor.reshape(n)
            if tensor.dim() == 3 and tensor.shape[-1] == 1:
                return tensor.reshape(n)
            raise ValueError(
                f"chunk_level field '{field}' must be [T,B] or [T,B,1], "
                f"got {tuple(tensor.shape)}"
            )
        return tensor.reshape(n, *tensor.shape[2:])

    def validate_flat_batch(self, flat: dict[str, torch.Tensor]) -> None:
        ref = flat.get("prev_logprobs")
        if not isinstance(ref, torch.Tensor):
            raise ValueError("flat batch missing prev_logprobs")
        n = ref.shape[0]
        if n != self.n_samples:
            raise ValueError(
                f"prev_logprobs n_samples={n}, expected {self.n_samples}"
            )
        for field in ("advantages", "loss_mask"):
            t = flat.get(field)
            if not isinstance(t, torch.Tensor):
                continue
            if t.shape[0] != n:
                raise ValueError(
                    f"{field} length {t.shape[0]} != prev_logprobs length {n}"
                )
            if self.is_chunk_level and field in _SCALAR_CHUNK_FIELDS:
                if t.dim() != 1:
                    raise ValueError(
                        f"chunk_level {field} must be 1D [N], got {tuple(t.shape)}"
                    )
            elif self.is_action_level and field in _SCALAR_CHUNK_FIELDS:
                if t.dim() != 2 or t.shape[1] != self.num_action_chunks:
                    raise ValueError(
                        f"action_level {field} must be [N, C={self.num_action_chunks}], "
                        f"got {tuple(t.shape)}"
                    )


def layout_from_batch(
    batch: dict[str, torch.Tensor],
    *,
    reward_type: str,
    logprob_type: str,
    action_dim: int,
) -> ActorBatchLayout:
    ref = batch.get("prev_logprobs")
    if not isinstance(ref, torch.Tensor) or ref.dim() < 2:
        raise ValueError("batch must contain prev_logprobs [T, B, ...]")
    t_chunk, batch_size = int(ref.shape[0]), int(ref.shape[1])

    rewards = batch.get("rewards")
    if isinstance(rewards, torch.Tensor) and rewards.dim() == 3:
        num_action_chunks = int(rewards.shape[2])
    elif ref.dim() >= 3 and action_dim > 0:
        num_action_chunks = int(ref.shape[2] // action_dim)
    else:
        num_action_chunks = 1

    return ActorBatchLayout(
        t_chunk=t_chunk,
        batch_size=batch_size,
        num_action_chunks=num_action_chunks,
        action_dim=action_dim,
        reward_type=reward_type,
        logprob_type=logprob_type,
    )


def resolve_loss_mask_canonical(
    batch: dict[str, torch.Tensor],
    layout: ActorBatchLayout,
    *,
    compute_loss_mask_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build [T, B] (chunk_level) or [T, B, C] (action_level) mask before flatten."""
    t, b = layout.t_chunk, layout.batch_size
    rewards = batch.get("rewards")

    if isinstance(batch.get("loss_mask"), torch.Tensor):
        loss_mask = batch["loss_mask"][:t]
        loss_mask_sum = batch.get("loss_mask_sum")
        loss_mask_sum = (
            loss_mask_sum[:t] if isinstance(loss_mask_sum, torch.Tensor) else None
        )
    elif layout.is_action_level and isinstance(batch.get("dones"), torch.Tensor):
        loss_mask, loss_mask_sum = compute_loss_mask_fn(batch["dones"])
    elif isinstance(rewards, torch.Tensor) and rewards.dim() == 3:
        if layout.is_chunk_level:
            loss_mask = torch.ones(
                t, b, dtype=rewards.dtype, device=rewards.device,
            )
            loss_mask_sum = None
        else:
            loss_mask = torch.ones_like(rewards[:t])
            loss_mask_sum = None
    else:
        loss_mask = torch.ones(t, b)
        loss_mask_sum = None

    if layout.is_chunk_level:
        if loss_mask.dim() == 3:
            if loss_mask_sum is not None and loss_mask_sum.dim() == 3:
                loss_mask_sum = loss_mask_sum.squeeze(-1)
            loss_mask = loss_mask.any(dim=-1)

    return loss_mask, loss_mask_sum


def flatten_for_training(
    batch: dict[str, torch.Tensor],
    layout: ActorBatchLayout,
    *,
    compute_loss_mask_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Single entry: rollout canonical dict → flat micro-batch dict for PPO."""
    t, b = layout.t_chunk, layout.batch_size
    loss_mask, loss_mask_sum = resolve_loss_mask_canonical(
        batch, layout, compute_loss_mask_fn=compute_loss_mask_fn,
    )

    flat: dict[str, torch.Tensor] = {}
    skip = frozenset({"loss_mask", "loss_mask_sum"})

    for key, value in batch.items():
        if key in skip or not isinstance(value, torch.Tensor) or value.dim() < 2:
            flat[key] = value
            continue

        n, bs = int(value.shape[0]), int(value.shape[1])
        if bs == b and n == t + 1:
            value = value[:t]
            n = t
        if bs == b and n == t:
            flat[key] = layout.flatten_field(key, value)
        else:
            flat[key] = value

    flat["loss_mask"] = layout.flatten_field("loss_mask", loss_mask)
    if isinstance(loss_mask_sum, torch.Tensor):
        if layout.is_action_level and loss_mask_sum.dim() == 3:
            flat["loss_mask_sum"] = layout.flatten_field(
                "loss_mask_sum", loss_mask_sum,
            )
        elif layout.is_chunk_level:
            flat["loss_mask_sum"] = loss_mask_sum.reshape(layout.n_samples)
        else:
            flat["loss_mask_sum"] = loss_mask_sum.reshape(layout.n_samples)

    layout.validate_flat_batch(flat)
    return flat


def expand_to_target_dim(
    tensor: Optional[torch.Tensor],
    target_shape: torch.Size,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    while tensor.dim() < len(target_shape):
        tensor = tensor.unsqueeze(-1)
    return tensor


def preprocess_loss_inputs(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    layout: ActorBatchLayout,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    prev_values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    versions: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Align all loss tensors to the same shape as reduced logprobs."""
    if layout.is_chunk_level:
        advantages = advantages.reshape(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.reshape(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.reshape(-1)
        if values is not None:
            values = values.reshape(-1)
        if prev_values is not None:
            prev_values = prev_values.reshape(-1)
        if returns is not None:
            returns = returns.reshape(-1)
        if versions is not None:
            versions = versions.reshape(-1)

    bsz = logprobs.shape[0]
    action_dim = layout.action_dim
    logprob_type = layout.logprob_type

    if logprob_type == "token_level":
        logprobs = logprobs.reshape(bsz, -1, action_dim)
        old_logprobs = old_logprobs.reshape(bsz, -1, action_dim)
        advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.unsqueeze(-1)
        if versions is not None:
            versions = versions.unsqueeze(-1)
    elif logprob_type == "action_level":
        logprobs = logprobs.reshape(bsz, -1, action_dim).sum(dim=-1)
        old_logprobs = old_logprobs.reshape(bsz, -1, action_dim).sum(dim=-1)
    elif logprob_type == "chunk_level":
        logprobs = logprobs.reshape(bsz, -1, action_dim).sum(dim=[1, 2])
        old_logprobs = old_logprobs.reshape(bsz, -1, action_dim).sum(dim=[1, 2])

    target_shape = logprobs.shape
    return {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": expand_to_target_dim(advantages, target_shape),
        "loss_mask": expand_to_target_dim(loss_mask, target_shape),
        "loss_mask_sum": expand_to_target_dim(loss_mask_sum, target_shape),
        "values": expand_to_target_dim(values, target_shape),
        "prev_values": expand_to_target_dim(prev_values, target_shape),
        "returns": expand_to_target_dim(returns, target_shape),
        "versions": expand_to_target_dim(versions, target_shape),
    }
