#!/usr/bin/env python3
"""Compare old vs RLinf dones layouts for compute_loss_mask (CPU only)."""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from joyrlinf.algorithms.embodied_advantages import compute_loss_mask


def _post_chunk_dones(
    n_chunks: int,
    bsz: int,
    n_actions: int,
    truncate_at: Optional[int],
) -> torch.Tensor:
    rows = []
    for c in range(n_chunks):
        row = torch.zeros(bsz, n_actions, dtype=torch.bool)
        if truncate_at is not None and c == truncate_at:
            row[:, -1] = True
        rows.append(row)
    return torch.stack(rows)


def joyrlinf_old_dones(
    n_chunks: int, bsz: int, n_actions: int, truncate_at: Optional[int],
) -> torch.Tensor:
    post = _post_chunk_dones(n_chunks, bsz, n_actions, truncate_at)
    bootstrap = torch.zeros(bsz, n_actions, dtype=torch.bool)
    return torch.cat([post, bootstrap.unsqueeze(0)], dim=0)


def rlinf_dones(
    n_chunks: int, bsz: int, n_actions: int, truncate_at: Optional[int],
) -> torch.Tensor:
    zero = torch.zeros(bsz, n_actions, dtype=torch.bool)
    post = _post_chunk_dones(n_chunks, bsz, n_actions, truncate_at)
    return torch.cat([zero.unsqueeze(0), post], dim=0)


def summarize(name: str, dones: torch.Tensor) -> None:
    loss_mask, _ = compute_loss_mask(dones)
    valid = int(loss_mask.sum().item())
    total = loss_mask.numel()
    print(
        f"{name:16s} dones={list(dones.shape)} "
        f"loss_mask_valid={valid}/{total} ({100.0 * valid / total:.1f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-chunks", type=int, default=10)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--n-actions", type=int, default=8)
    args = parser.parse_args()

    print(
        f"Simulating {args.n_chunks} chunk steps, "
        f"bsz={args.bsz}, num_action_chunks={args.n_actions}\n"
    )
    for trunc_at in [0, 4, args.n_chunks - 1, None]:
        label = "full_rollout" if trunc_at is None else f"trunc@chunk{trunc_at}"
        print(f"--- {label} ---")
        old = joyrlinf_old_dones(args.n_chunks, args.bsz, args.n_actions, trunc_at)
        new = rlinf_dones(args.n_chunks, args.bsz, args.n_actions, trunc_at)
        summarize("joyrlinf_old", old)
        summarize("rlinf/fixed", new)
        diff = (compute_loss_mask(old)[0] != compute_loss_mask(new)[0]).sum().item()
        print(f"mask_diff_entries={int(diff)}\n")


if __name__ == "__main__":
    main()
