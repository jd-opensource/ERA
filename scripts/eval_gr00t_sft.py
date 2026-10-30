#!/usr/bin/env python3
"""Evaluate a GR00T SFT checkpoint in ManiSkill (no RL / no distributed training).

Use this to verify whether the SFT model can complete the task in sim before
debugging GRPO training. Typical usage inside the gr00t venv::

    cd /home/gyc/gitcode/joyrlinf
    python scripts/eval_gr00t_sft.py \\
      --config configs/maniskill_grpo_gr00t_panda_peginsertion.yaml \\
      --num-envs 32 --num-action-chunks 16 --mode eval

Compare action-chunk horizons (SFT dataset uses 16-step chunks)::

    python scripts/eval_gr00t_sft.py \\
      --config configs/maniskill_grpo_gr00t_panda_peginsertion.yaml \\
      --compare-chunks 5 16 --mode eval

Exit code 0 always; inspect printed success_rate.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from joyrlinf.envs.action_utils import prepare_actions
from joyrlinf.envs.maniskill_env import ManiSkillEnv
from joyrlinf.models.model_factory import build_vla_model

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    num_envs: int
    num_action_chunks: int
    mode: str
    max_steps: int
    success_count: int
    success_rate: float
    mean_return: float
    max_return: float
    elapsed_s: float
    proprio_mode: str
    use_eval_env: bool


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("mani_skill").setLevel(logging.WARNING)


def _set_action_chunks(model, num_action_chunks: int) -> None:
    """Align model output width with env execution horizon."""
    core = model.model
    core.output_action_chunks = num_action_chunks
    if hasattr(core, "action_head"):
        core.action_head.action_chunk = num_action_chunks


def _set_noise_level(model, noise_level: float | None) -> None:
    if noise_level is None:
        return
    head = model.model.action_head
    head.rl_config.noise_level = noise_level
    head.rl_config.noise_anneal = False


def _build_env_cfg(cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    section = "eval" if args.use_eval_env else "train"
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env[section], resolve=True))
    with open_dict(env_cfg):
        env_cfg.total_num_envs = args.num_envs
        env_cfg.init_params.num_envs = args.num_envs
        if args.proprio_mode is not None:
            env_cfg.proprio_mode = args.proprio_mode
        if args.max_steps is not None:
            env_cfg.max_episode_steps = args.max_steps
            env_cfg.max_steps_per_rollout_epoch = args.max_steps
        if args.seed is not None:
            env_cfg.seed = args.seed
    return env_cfg


def _prepare_obs(obs: dict) -> dict:
    return {
        "main_images": obs.get("main_images"),
        "wrist_images": obs.get("wrist_images"),
        "states": obs.get("states"),
        "task_descriptions": obs.get("task_descriptions"),
    }


def run_single_eval(
    cfg: DictConfig,
    args: argparse.Namespace,
    *,
    num_action_chunks: int,
    mode: str,
) -> EvalResult:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    env_cfg = _build_env_cfg(cfg, args)
    max_steps = int(env_cfg.max_steps_per_rollout_epoch)
    if max_steps % num_action_chunks != 0:
        raise ValueError(
            f"max_steps={max_steps} must be divisible by "
            f"num_action_chunks={num_action_chunks}"
        )
    n_chunk_steps = max_steps // num_action_chunks

    policy_setup = cfg.actor.model.get("policy_setup", "panda-ee-dpose")
    action_dim = int(cfg.actor.model.get("action_dim", 7))

    logger.info(
        "Eval start: envs=%d chunks=%d mode=%s steps=%d chunk_loops=%d "
        "proprio=%s eval_env=%s device=%s",
        args.num_envs,
        num_action_chunks,
        mode,
        max_steps,
        n_chunk_steps,
        env_cfg.get("proprio_mode", "default"),
        args.use_eval_env,
        device,
    )

    t0 = time.time()
    env = ManiSkillEnv(cfg=env_cfg, num_envs=args.num_envs, seed_offset=0)

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))
    with open_dict(model_cfg):
        model_cfg.num_action_chunks = num_action_chunks
    model = build_vla_model(model_cfg, device, mode="inference")
    _set_action_chunks(model, num_action_chunks)
    _set_noise_level(model, args.noise_level)

    obs, _ = env.reset()
    total_returns = torch.zeros(args.num_envs, device=env.device)

    for chunk_idx in range(n_chunk_steps):
        actions_np, _ = model.predict(
            _prepare_obs(obs),
            mode=mode,
            compute_values=False,
        )
        if actions_np.shape[0] != args.num_envs:
            raise RuntimeError(
                f"Expected actions batch {args.num_envs}, got {actions_np.shape[0]}"
            )
        if actions_np.shape[1] != num_action_chunks:
            raise RuntimeError(
                f"Model returned {actions_np.shape[1]} chunks, expected {num_action_chunks}"
            )
        if actions_np.shape[2] != action_dim:
            raise RuntimeError(
                f"Model returned action_dim={actions_np.shape[2]}, expected {action_dim}"
            )

        chunk_actions = prepare_actions(
            actions_np,
            env_type=env_cfg.get("env_type", "maniskill"),
            num_action_chunks=num_action_chunks,
            action_dim=action_dim,
            policy=policy_setup,
        )
        obs, chunk_rewards, _, _, info = env.chunk_step(chunk_actions)
        total_returns += chunk_rewards.sum(dim=1)

        if args.verbose and chunk_idx % max(1, n_chunk_steps // 10) == 0:
            succ_now = int(env.segment_success_once.sum().item())
            logger.debug(
                "chunk %d/%d interim_success=%d/%d mean_return=%.4f",
                chunk_idx + 1,
                n_chunk_steps,
                succ_now,
                args.num_envs,
                total_returns.mean().item(),
            )
            if "success" in info:
                logger.debug(
                    "last_step_success=%d action[0,0]=%s",
                    int(info["success"].sum().item()),
                    np.round(actions_np[0, 0], 4),
                )

    success_count = int(env.segment_success_once.sum().item())
    elapsed = time.time() - t0
    return EvalResult(
        num_envs=args.num_envs,
        num_action_chunks=num_action_chunks,
        mode=mode,
        max_steps=max_steps,
        success_count=success_count,
        success_rate=success_count / max(args.num_envs, 1),
        mean_return=float(total_returns.mean().item()),
        max_return=float(total_returns.max().item()),
        elapsed_s=elapsed,
        proprio_mode=str(env_cfg.get("proprio_mode", "default")),
        use_eval_env=args.use_eval_env,
    )


def _print_result(r: EvalResult) -> None:
    env_tag = "eval" if r.use_eval_env else "train"
    print(
        f"  chunks={r.num_action_chunks:2d}  mode={r.mode:5s}  "
        f"proprio={r.proprio_mode:16s}  env={env_tag:5s}  "
        f"success={r.success_count:4d}/{r.num_envs:<4d} ({100*r.success_rate:5.1f}%)  "
        f"return mean={r.mean_return:.3f} max={r.max_return:.3f}  "
        f"time={r.elapsed_s:.1f}s"
    )


def _print_header() -> None:
    print("\n=== GR00T SFT eval summary ===")
    print(
        "  chunks  mode   proprio           env    "
        "success              return           time"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=str,
        default="configs/maniskill_grpo_gr00t_panda_peginsertion.yaml",
        help="Training YAML (env + actor.model sections are reused)",
    )
    p.add_argument("--num-envs", type=int, default=32, help="Parallel ManiSkill envs")
    p.add_argument(
        "--num-action-chunks",
        type=int,
        default=16,
        help="Action steps executed per model call (SFT dataset horizon is 16)",
    )
    p.add_argument(
        "--compare-chunks",
        type=int,
        nargs="+",
        default=None,
        help="Run multiple chunk sizes, e.g. --compare-chunks 5 16",
    )
    p.add_argument(
        "--mode",
        choices=("eval", "train"),
        default="eval",
        help="eval = deterministic denoise; train = flow_sde exploration noise",
    )
    p.add_argument(
        "--compare-modes",
        choices=("eval", "train"),
        nargs="+",
        default=None,
        help="Run both modes for each chunk setting",
    )
    p.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="Override gr00t.rl_head_config.noise_level (only affects mode=train)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode length override (default: from env.train/eval in config)",
    )
    p.add_argument(
        "--proprio-mode",
        type=str,
        default=None,
        choices=("panda_tcp_pose8", "panda_qpos8", "default"),
        help="Override env proprio_mode",
    )
    p.add_argument(
        "--use-eval-env",
        action="store_true",
        help="Use env.eval section (fixed reset ids) instead of env.train",
    )
    p.add_argument("--seed", type=int, default=None, help="Env seed override")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    p.add_argument("--verbose", action="store_true", help="Per-chunk debug logs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)

    cfg = OmegaConf.load(args.config)
    chunk_list = args.compare_chunks or [args.num_action_chunks]
    mode_list = args.compare_modes or [args.mode]

    results: list[EvalResult] = []
    for chunks in chunk_list:
        for mode in mode_list:
            results.append(
                run_single_eval(cfg, args, num_action_chunks=chunks, mode=mode)
            )

    _print_header()
    for r in results:
        _print_result(r)

    best = max(results, key=lambda r: (r.success_count, r.mean_return))
    print(
        f"\nBest: chunks={best.num_action_chunks} mode={best.mode} "
        f"success={best.success_count}/{best.num_envs} "
        f"({100*best.success_rate:.1f}%)"
    )
    if best.success_count == 0:
        print(
            "\nNo successes — SFT checkpoint likely does not transfer to this sim loop.\n"
            "Next checks:\n"
            "  1. Try --compare-chunks 5 16 (RL config uses 5, SFT uses 16)\n"
            "  2. Compare --proprio-mode panda_tcp_pose8 vs panda_qpos8\n"
            "  3. Confirm checkpoint path and task text match SFT dataset\n"
            "  4. If eval>0 but train=0, reduce --noise-level for RL rollout"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
