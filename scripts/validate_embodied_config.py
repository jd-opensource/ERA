#!/usr/bin/env python3
"""Validate embodied YAML (batch math, comm_mode, placement) without a full train run.

Usage::

    python scripts/validate_embodied_config.py --config configs/maniskill_grpo_pi05_async_test1.yaml
    python scripts/validate_embodied_config.py --config configs/maniskill_grpo_gr00t_async_test1.yaml

Offline validation infers GPUs-per-node from:

- ``cluster.gpus_per_node`` (required for ``placement_basis: per_node`` multi-node), or
- contiguous global ranks ``0..N-1`` on a single-node config (e.g. rollout 0–5, actor 6–7 → 8).

This does **not** verify ``component_placement`` against ``torch.distributed``
world size (that runs automatically when you start ``scripts/train.py`` under
``torchrun``, via ``EmbodiedRunner``).

Exit code 0 if validation passes, non-zero on ``ValueError``.
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running from repo root without installing the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    from omegaconf import OmegaConf

    from joyrlinf.core.config_validation import validate_embodied_training_config

    cfg = OmegaConf.load(args.config)
    # Standalone: skip placement vs world_size (requires torchrun + init).
    validate_embodied_training_config(cfg, check_world_ranks=False)
    print("validate_embodied_training_config: OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        print(f"validate_embodied_training_config: FAILED\n{e}", file=sys.stderr)
        raise SystemExit(1)
