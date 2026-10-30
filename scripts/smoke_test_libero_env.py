#!/usr/bin/env python3
"""Minimal LIBERO env smoke test (no torchrun / no OpenPI model).

Usage (inside xxl1, openpi venv)::

    source switch_env openpi
    export PYTHONPATH=/ufs/xxl/joyrlinf:$PYTHONPATH
    export MUJOCO_GL=egl
    export MUJOCO_EGL_DEVICE_ID=0
    python scripts/smoke_test_libero_env.py --suite libero_10 --num-envs 1

First run may take 1–3 minutes (MuJoCo subprocess startup).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
from omegaconf import OmegaConf

from joyrlinf.core.launcher import parse_placement
from joyrlinf.envs.build_env import build_env
from joyrlinf.envs.action_utils import prepare_actions_for_libero


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="libero_10",
        choices=["libero_10", "libero_90", "libero_130"],
        help="Smaller libero_10 is fastest for first test.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=2)
    args = parser.parse_args()

    if args.num_envs < args.group_size or args.num_envs % args.group_size != 0:
        print("num_envs must be >= group_size and divisible by group_size")
        return 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.create(
        {
            "cluster": {
                "component_placement": {"env": "0", "rollout": "0", "actor": "0"},
            },
            "env": {
                "train": {
                    "env_type": "libero",
                    "task_suite_name": args.suite,
                    "group_size": args.group_size,
                    "auto_reset": True,
                    "use_fixed_reset_state_ids": False,
                    "max_episode_steps": 512,
                    "use_rel_reward": True,
                    "reward_coef": 5.0,
                    "reset_gripper_open": True,
                    "seed": 0,
                    "init_params": {
                        "camera_heights": 128,
                        "camera_widths": 128,
                    },
                },
            },
        }
    )
    placement = parse_placement(cfg.cluster)

    print(f"[smoke] building LiberoEnv suite={args.suite} num_envs={args.num_envs}")
    t0 = time.time()
    env = build_env(
        cfg.env.train,
        num_envs=args.num_envs,
        rank=0,
        placement=placement,
        device=device,
        colocated=True,
    )
    print(f"[smoke] env built in {time.time() - t0:.1f}s")

    t1 = time.time()
    obs, _ = env.reset()
    print(f"[smoke] reset in {time.time() - t1:.1f}s")
    print(
        f"  main_images: {obs['main_images'].shape} dtype={obs['main_images'].dtype}"
    )
    print(
        f"  wrist_images: {obs['wrist_images'].shape} "
        f"task: {obs['task_descriptions'][0][:60]}..."
    )

    chunk = prepare_actions_for_libero(
        torch.zeros(args.num_envs, args.chunks, 7),
        num_action_chunks=args.chunks,
        action_dim=7,
        device=device,
    )
    t2 = time.time()
    obs, rew, term, trunc, info = env.chunk_step(chunk)
    print(f"[smoke] chunk_step in {time.time() - t2:.1f}s")
    print(f"  rewards: {rew.shape} sample={rew[0].tolist()}")
    print(f"  term(last chunk): {term[:, -1].tolist()}")

    print("[smoke] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
