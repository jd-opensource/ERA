"""Standalone env worker for fully disaggregated deployment.

When env and rollout run on separate GPUs, the EnvWorker drives the
ManiSkill environment and communicates observations/actions with its
paired RolloutOnlyWorker via NCCLPipe.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import record_function
from omegaconf import DictConfig

from joyrlinf.core.direct_pipe import NCCLPipe
from joyrlinf.core.launcher import parse_placement
from joyrlinf.algorithms.embodied_advantages import pack_chunk_step_signals
from joyrlinf.envs.build_env import build_env

logger = logging.getLogger(__name__)


def _prepare_env_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Format obs dict for model input. Adds missing keys expected by openpi."""
    return {
        "main_images": obs.get("main_images"),
        "states": obs.get("states"),
        "task_descriptions": obs.get("task_descriptions"),
        "wrist_images": obs.get("wrist_images"),
        "extra_view_images": obs.get("extra_view_images"),
    }


def _obs_to_tensor_dict(env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract only GPU tensors from obs dict for NCCLPipe transfer."""
    tensors = {}
    for k, v in env_obs.items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v
    return tensors


class EnvWorker:
    """Runs env simulation on a dedicated GPU, communicates with rollout via NCCL.

    Protocol per rollout epoch:
      1. reset -> send task_descriptions (object_list) + obs tensors (NCCLPipe)
      2. for each chunk step:
         - recv chunk_actions from rollout
         - env.chunk_step(actions) -> new obs, rewards, terms, truncs
         - send step_results dict to rollout
      3. send final obs for bootstrap value estimation
    """

    def __init__(self, rank: int, cfg: DictConfig, device: torch.device):
        self.rank = rank
        self.cfg = cfg
        self.device = device

        placement = parse_placement(cfg.cluster)
        num_env_gpus = len(placement.env_ranks)
        self.num_envs_per_gpu = cfg.env.train.total_num_envs // num_env_gpus
        self.num_action_chunks = cfg.actor.model.num_action_chunks
        logger.info(
            "EnvWorker init: rank=%d device=%s num_envs_per_gpu=%d",
            self.rank, self.device, self.num_envs_per_gpu,
        )

        self.env = build_env(
            cfg.env.train,
            num_envs=self.num_envs_per_gpu,
            rank=self.rank,
            placement=placement,
            device=self.device,
            colocated=False,
        )
        logger.info("EnvWorker env ready on device=%s", self.env.device)

        # Continuous rollout semantics (aligned with RLinf):
        # With auto_reset=True, only the first epoch does a full reset.
        self._rollout_env_ready = False
        self._cached_obs: dict[str, Any] | None = None
        self._cached_info: dict[str, Any] = {}

    def _train_auto_reset_enabled(self) -> bool:
        return bool(self.cfg.env.train.get("auto_reset", True))

    def _needs_full_reset_at_epoch_start(self) -> bool:
        if not self._train_auto_reset_enabled():
            return True
        return not self._rollout_env_ready

    @staticmethod
    def _zero_segment_success_metrics(env: Any) -> None:
        seg = getattr(env, "segment_success_once", None)
        if seg is not None and hasattr(seg, "zero_"):
            seg.zero_()

    def _finish_train_rollout_epoch(self) -> None:
        train = self.cfg.env.train
        if not train.get("auto_reset", True):
            return
        if train.get("use_fixed_reset_state_ids", False):
            return
        self.env.update_reset_state_ids()

    def run_epoch(
        self,
        send_pipe: NCCLPipe,
        recv_pipe: NCCLPipe,
        env_rollout_group: dist.ProcessGroup,
        *,
        return_timeline: bool = False,
    ) -> dict[str, Any] | None:
        """Drive one rollout epoch: reset, step loop, send data to rollout.

        Args:
            send_pipe: NCCLPipe env->rollout for sending obs/results.
            recv_pipe: NCCLPipe rollout->env for receiving actions.
            env_rollout_group: process group containing this env + paired rollout
                               (used for broadcast_object_list of string metadata).
        """
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.num_action_chunks
        )
        logger.debug("EnvWorker epoch: n_chunk_steps=%d", n_chunk_steps)

        # 1. Reset or continue (aligned with CombinedEnvRolloutWorker / RLinf)
        need_full_reset = self._needs_full_reset_at_epoch_start()
        use_fixed_ids = bool(self.cfg.env.train.get("use_fixed_reset_state_ids", False))
        env_reset_start_ns = None
        env_reset_end_ns = None
        if need_full_reset:
            if not use_fixed_ids:
                self.env.update_reset_state_ids()
            env_reset_start_ns = time.time_ns()
            obs, info = self.env.reset()
            env_reset_end_ns = time.time_ns()            
            self._rollout_env_ready = True
        else:
            if self._cached_obs is None:
                raise RuntimeError(
                    "Continuing rollout without cached obs; call full reset first."
                )
            obs, info = self._cached_obs, self._cached_info
            self._zero_segment_success_metrics(self.env)
            logger.debug("EnvWorker: reset skipped (auto_reset continuous)")
        env_obs = _prepare_env_obs(obs)

        # 2. Send task_descriptions (strings) via broadcast_object_list
        task_descs = env_obs.get("task_descriptions", [])
        dist.broadcast_object_list(
            [task_descs], src=self.rank, group=env_rollout_group)

        # 3. Send initial obs tensors
        obs_tensors = _obs_to_tensor_dict(env_obs)
        send_pipe.send_dict(obs_tensors)

        # 4. Chunk step loop
        interactions_timeline = []
        for chunk_idx in range(n_chunk_steps):
            interaction_start_ns = time.time_ns()
            # Recv actions from rollout (reverse direction)
            with record_function("env/recv_actions"):
                action_data = recv_pipe.recv_dict()
            chunk_actions = action_data["chunk_actions"]

            # Step env (PhysX physics + camera rendering, see ManiSkillEnv.step)
            env_step_start_ns = time.time_ns()
            with record_function("env/chunk_step"):
                obs, chunk_rewards, chunk_terms, chunk_truncs, info = (
                    self.env.chunk_step(chunk_actions)
                )
            env_step_end_ns = time.time_ns()
            env_obs = _prepare_env_obs(obs)

            step_rewards, step_dones, step_terms, step_truncs = (
                pack_chunk_step_signals(chunk_rewards, chunk_terms, chunk_truncs)
            )

            # Build results dict: rewards/terms/truncs + new obs
            results = {
                "rewards": step_rewards,
                "dones": step_dones,
                "terminations": step_terms,
                "truncations": step_truncs,
            }
            results.update(_obs_to_tensor_dict(env_obs))
            with record_function("env/send_results"):
                send_pipe.send_dict(results)

            interaction_end_ns = time.time_ns()
            interactions_timeline.append({
                "interaction_idx": chunk_idx,
                "interaction_time": {
                    "start_wall_ns": interaction_start_ns,
                    "end_wall_ns": interaction_end_ns,
                },
                "env_step": {
                    "start_wall_ns": env_step_start_ns,
                    "end_wall_ns": env_step_end_ns,
                },
            })

        # One message per epoch: how many parallel envs achieved success at least
        # once during this segment (ManiSkillEnv.segment_success_once).
        stat = torch.tensor(
            [int(self.env.segment_success_once.sum().item())],
            dtype=torch.int64,
            device=self.device,
        )
        send_pipe.send_dict({"rollout_local_success_count": stat})

        # Cache last env state for continuous rollout.
        self._cached_obs = obs
        self._cached_info = info
        self._finish_train_rollout_epoch()

        logger.debug("EnvWorker epoch done")
        
        if not return_timeline:
            return None
        return {
            "env_reset": {
                "performed": need_full_reset,
                "start_wall_ns": env_reset_start_ns,
                "end_wall_ns": env_reset_end_ns,
            },
            "interactions": interactions_timeline,
            "bootstrap_infer": {},
        }

