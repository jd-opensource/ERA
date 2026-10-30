"""Runner scheduling helpers (aligned with RLinf check_progress)."""
from __future__ import annotations

from typing import Union


def safe_is_divisible(a: int, b: int) -> bool:
    """Divisibility check that treats 0 as non-divisible."""
    if a == 0 or b == 0:
        return False
    return a % b == 0


def check_progress(
    step: int,
    max_steps: int,
    val_check_interval: int,
    save_interval: int,
    limit_val_batches: Union[int, float, None] = None,
    run_time_exceeded: bool = False,
) -> tuple[bool, bool, bool]:
    """Decide whether to run validation / save checkpoint at *step*.

    Args:
        step: Global training step, **1-indexed** (第 1 步 => ``step=1``).
            First eval/save with ``interval=2`` happens at ``step=2``.
        max_steps: Total steps target.
        val_check_interval: Run eval every N steps; <=0 disables.
        save_interval: Save every N steps; <=0 disables.
        limit_val_batches: Unused; kept for RLinf API compatibility (0 disables).

    Returns:
        (run_val, save_model, is_train_end)
    """
    if step < 1:
        return False, False, False

    is_validation_enabled = (
        limit_val_batches != 0 and val_check_interval > 0
    )
    is_save_enabled = save_interval > 0
    is_train_end = step >= max_steps

    if is_validation_enabled and save_interval > 0:
        assert save_interval % val_check_interval == 0, (
            f"save_interval={save_interval} must be divisible by "
            f"val_check_interval={val_check_interval}"
        )

    # 1-indexed: interval=2 -> eval/save at steps 2, 4, 6, ...
    run_val = (
        step % val_check_interval == 0
        or is_train_end
        or run_time_exceeded
    )
    run_val &= is_validation_enabled

    save_model = (
        step % save_interval == 0
        or is_train_end
        or run_time_exceeded
    )
    save_model &= is_save_enabled

    return run_val, save_model, is_train_end


def prefix_metrics(metrics: dict, prefix: str) -> dict:
    """Prefix metric keys, e.g. ``policy_loss`` -> ``train/policy_loss``."""
    if not prefix:
        return dict(metrics)
    p = prefix if prefix.endswith("/") else f"{prefix}/"
    return {f"{p}{k}": v for k, v in metrics.items()}
