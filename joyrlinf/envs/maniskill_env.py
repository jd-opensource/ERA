"""ManiSkill environment wrapper for GPU-vectorized simulation.

Adapted for joyrlinf: no Channel dependency, all data stays on GPU.
Supports GRPO group_size by initializing groups of envs with shared states.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch.profiler import record_function
from omegaconf import OmegaConf, open_dict

logger = logging.getLogger(__name__)

try:
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils import common
    from mani_skill.utils.common import torch_clone_dict
    from mani_skill.utils.visualization.misc import images_to_video, tile_images
except ImportError:
    logger.warning("mani_skill not installed; ManiSkillEnv unavailable")

_CUSTOM_TASK_MODULES = [
    "joyrlinf.envs.tasks.put_on_in_scene_multi",
    "joyrlinf.envs.tasks.put_carrot_on_plate",
    "joyrlinf.envs.tasks.panda_table_agent",
]

_tasks_registered = False


def _ensure_custom_tasks_registered():
    global _tasks_registered
    if _tasks_registered:
        return
    import importlib
    for mod_name in _CUSTOM_TASK_MODULES:
        try:
            importlib.import_module(mod_name)
            logger.debug("Registered custom tasks from %s", mod_name)
        except ImportError as e:
            logger.warning(
                "Failed to import custom task module '%s': %s. "
                "If you need custom envs like PutOnPlateInScene25Main-v3, "
                "ensure joyrlinf is installed or on PYTHONPATH",
                mod_name, e,
            )
    _tasks_registered = True


def _extract_termination(info: dict, num_envs: int, device: torch.device) -> torch.Tensor:
    success = info.get("success")
    fail = info.get("fail")
    if success is not None and fail is not None:
        return torch.logical_or(success, fail)
    if success is not None:
        return success.clone()
    if fail is not None:
        return fail.clone()
    return torch.zeros(num_envs, dtype=torch.bool, device=device)


class ManiSkillEnv:
    """GPU-vectorized ManiSkill environment wrapper.

    Key design: obs tensors stay on GPU. No .cpu() conversion.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int = 0,
        *,
        reset_partition: str = "per_rank",
        slot_index: int = 0,
        slot_count: int = 1,
    ):
        _ensure_custom_tasks_registered()

        self.cfg = cfg
        self.seed = cfg.seed + seed_offset
        # Base seed WITHOUT the per-rank offset, used by the topology-independent
        # "global" reset partition so every rank draws the same global scene list.
        self._base_seed = int(cfg.seed)
        # Reset-scene partitioning strategy:
        #   "per_rank" (default, legacy): each rank draws num_group scenes with a
        #     rank-offset seed -> the scene *union* depends on the rollout GPU
        #     count (sync-8-GPU vs async-4-GPU sample different scenes).
        #   "global": all ranks draw one shared global list of (num_group *
        #     slot_count) scenes from a single seed, then this rank slices its own
        #     contiguous segment -> the union is identical regardless of GPU count,
        #     making sync/async success rates directly comparable.
        self.reset_partition = str(reset_partition)
        self.slot_index = int(slot_index)
        self.slot_count = max(1, int(slot_count))
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.get("use_rel_reward", True)
        self.ignore_terminations = cfg.get("ignore_terminations", False)
        self.group_size = cfg.get("group_size", 1)
        self.num_group = num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.get("use_fixed_reset_state_ids", False)
        self._num_envs = num_envs

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs
        env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        env_args = {k: v for k, v in env_args.items() if v is not None}
        logger.debug(
            "Creating ManiSkill env: id=%s num_envs=%s control_mode=%s sim_backend=%s",
            env_args.get("id"),
            env_args.get("num_envs"),
            env_args.get("control_mode"),
            env_args.get("sim_backend"),
        )
        self.env: BaseEnv = gym.make(**env_args)
        logger.debug("ManiSkill env created on device=%s", self.env.unwrapped.device)

        self.prev_step_reward = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._init_reset_state_ids()
        self._init_metrics()

        # Video recording
        self.video_cfg = cfg.get("video_cfg", {})
        self.render_images: list[np.ndarray] = []
        self.video_cnt = 0

    @property
    def num_envs(self) -> int:
        return self.env.unwrapped.num_envs

    @property
    def device(self) -> torch.device:
        return self.env.unwrapped.device

    @property
    def elapsed_steps(self):
        return self.env.unwrapped.elapsed_steps

    @property
    def instruction(self) -> list[str]:
        return self.env.unwrapped.get_language_instruction()

    @property
    def total_num_group_envs(self) -> int:
        if hasattr(self.env.unwrapped, "total_num_trials"):
            return self.env.unwrapped.total_num_trials
        return 1000

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        # Round counter for the "global" partition so optional per-epoch
        # resampling (non-fixed / PPO) advances in lockstep across all ranks.
        self._reset_draw_round = 0
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        if self.reset_partition == "global":
            # Same global draw on every rank (seed independent of rank/GPU count),
            # then slice this rank's contiguous segment. Advancing the per-round
            # seed keeps ranks synchronized when resampling across epochs.
            g = torch.Generator()
            g.manual_seed(self._base_seed + self._reset_draw_round)
            global_num_group = self.num_group * self.slot_count
            all_ids = torch.randint(
                low=0, high=self.total_num_group_envs,
                size=(global_num_group,), generator=g,
            )
            start = self.slot_index * self.num_group
            ids = all_ids[start:start + self.num_group]
            self._reset_draw_round += 1
        else:
            ids = torch.randint(
                low=0, high=self.total_num_group_envs,
                size=(self.num_group,), generator=self._generator,
            )
        self.reset_state_ids = ids.repeat_interleave(self.group_size).to(self.device)

    def _init_metrics(self):
        self.success_once = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.segment_success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.returns[mask] = 0.0
        else:
            self.prev_step_reward.zero_()
            self.success_once.zero_()
            self.segment_success_once.zero_()
            self.returns.zero_()

    # ─── Observation processing ───

    def _default_task_descriptions(self) -> list[str]:
        try:
            return self.instruction
        except Exception:
            pass
        task = getattr(self.cfg, "default_task_description", None)
        if task:
            return [task] * self.num_envs
        return ["complete the task"] * self.num_envs

    def _extract_panda_qpos8(self, device: torch.device) -> torch.Tensor:
        """Legacy 8-dim joint-space proprio: 7 arm joints + 1 averaged gripper.

        Kept for backward compat with datasets that store qpos-style state.
        New Panda LeRobot datasets (e.g. ``maniskill_peginsertionside_240``)
        actually store TCP world pose + 2 fingers — use ``panda_tcp_pose8``.
        """
        qpos = self.env.unwrapped.agent.robot.get_qpos().to(device, dtype=torch.float32)
        if qpos.shape[-1] >= 9:
            arm = qpos[..., :7]
            gripper = qpos[..., 7:9].mean(dim=-1, keepdim=True)
            return torch.cat([arm, gripper], dim=-1)
        if qpos.shape[-1] >= 8:
            return qpos[..., :8]
        pad = torch.zeros(*qpos.shape[:-1], 8 - qpos.shape[-1], device=device)
        return torch.cat([qpos, pad], dim=-1)

    @staticmethod
    def _quat_wxyz_to_rotvec(q_wxyz: torch.Tensor) -> torch.Tensor:
        """Convert (..., 4) quaternion wxyz to (..., 3) axis-angle rotvec.

        Matches scipy ``Rotation.from_quat(xyzw).as_rotvec()`` semantics, which
        is what the ``rechim25/maniskill_peginsertionside_240`` SFT dataset
        stores in ``state[3:6]``.
        """
        w = q_wxyz[..., 0].clamp(-1.0, 1.0)
        xyz = q_wxyz[..., 1:]
        sin_half = torch.linalg.norm(xyz, dim=-1)
        angle = 2.0 * torch.atan2(sin_half, w)
        small = sin_half < 1e-8
        scale = torch.where(
            small,
            torch.full_like(sin_half, 2.0),
            angle / sin_half.clamp_min(1e-12),
        )
        return xyz * scale.unsqueeze(-1)

    def _extract_panda_tcp_pose8(self, device: torch.device) -> torch.Tensor:
        """8-dim proprio: TCP pose in robot-base frame (3 xyz + 3 axis-angle) + 2 fingers.

        ManiSkill ``agent.tcp.pose`` is the TCP link's world pose. The
        ``maniskill_peginsertionside_240`` SFT dataset stores TCP **relative
        to the robot base** (= ``robot.pose.inv() * tcp.pose``), which makes
        the state invariant to where the robot is mounted in the table scene.

        Verified against episode 0 frame 0:
          state = [0.6224, 0.0238, 0.1753, 2.1878, 2.2467, 0.0382, 0.04, 0.04]
        """
        agent = self.env.unwrapped.agent
        rel = agent.robot.pose.inv() * agent.tcp.pose
        tcp_p_local = rel.p.to(device, dtype=torch.float32)
        rotvec = self._quat_wxyz_to_rotvec(rel.q.to(device, dtype=torch.float32))
        qpos = agent.robot.get_qpos().to(device, dtype=torch.float32)
        fingers = qpos[..., -2:]
        return torch.cat([tcp_p_local, rotvec, fingers], dim=-1)

    def _extract_proprioception(self, raw_obs, device: torch.device) -> torch.Tensor:
        proprio_mode = getattr(self.cfg, "proprio_mode", "default")
        if proprio_mode == "panda_qpos8":
            return self._extract_panda_qpos8(device)
        if proprio_mode == "panda_tcp_pose8":
            return self._extract_panda_tcp_pose8(device)
        return self.env.unwrapped.agent.robot.get_qpos().to(device, dtype=torch.float32)

    def wrap_obs(self, raw_obs, info: Optional[dict] = None) -> dict[str, Any]:
        """Extract VLA-format obs from raw ManiSkill obs. All tensors on GPU."""
        wrap_obs_mode = getattr(self.cfg, "wrap_obs_mode", "default")
        if wrap_obs_mode == "raw":
            if info is None or "extracted_obs" not in info:
                raise ValueError("wrap_obs_mode=raw requires infos['extracted_obs']")
            return info["extracted_obs"]

        if wrap_obs_mode == "simple":
            sensor_data = raw_obs["sensor_data"]
            main_key = getattr(self.cfg, "main_camera_key", "base_camera")
            if main_key not in sensor_data:
                main_key = next(iter(sensor_data))
            obs_image = sensor_data[main_key]["rgb"].to(torch.uint8)
            wrist_key = getattr(self.cfg, "wrist_camera_key", None)
            wrist_images = None
            if wrist_key and wrist_key in sensor_data:
                wrist_images = sensor_data[wrist_key]["rgb"].to(torch.uint8)
            else:
                for key in ("hand_camera", "wrist_camera"):
                    if key in sensor_data:
                        wrist_images = sensor_data[key]["rgb"].to(torch.uint8)
                        break
                if wrist_images is None:
                    for key, value in sensor_data.items():
                        if key != main_key:
                            wrist_images = value["rgb"].to(torch.uint8)
                            break
            proprioception = self._extract_proprioception(raw_obs, obs_image.device)
            return {
                "main_images": obs_image,
                "wrist_images": wrist_images,
                "states": proprioception,
                "task_descriptions": self._default_task_descriptions(),
            }

        obs_image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        proprioception = self._extract_proprioception(raw_obs, obs_image.device)
        return {
            "main_images": obs_image,
            "states": proprioception,
            "task_descriptions": self._default_task_descriptions(),
        }

    # ─── Reward ───

    def _calc_step_reward(self, reward: torch.Tensor, info: dict) -> torch.Tensor:
        reward_mode = self.cfg.get("reward_mode", "default")
        if reward_mode == "raw":
            r = reward
        elif reward_mode == "only_success":
            r = info.get("success", torch.zeros_like(reward)).float()
        else:
            r = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            r += info.get("is_src_obj_grasped", torch.zeros_like(r)) * 0.1
            r += info.get("consecutive_grasp", torch.zeros_like(r)) * 0.1
            success = info.get("success", torch.zeros_like(r))
            grasp = info.get("is_src_obj_grasped", torch.zeros_like(r))
            r += (success & grasp).float() * 1.0

        if self.use_rel_reward:
            diff = r - self.prev_step_reward
            self.prev_step_reward = r
            return diff
        self.prev_step_reward = r
        return r

    def _record_metrics(self, step_reward: torch.Tensor, info: dict) -> dict:
        self.returns += step_reward
        episode_info = {"return": self.returns.clone()}
        if "success" in info:
            self.success_once |= info["success"]
            self.segment_success_once |= info["success"]
            episode_info["success_once"] = self.success_once.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        info["episode"] = episode_info
        return info

    # ─── Core API ───

    def reset(self, *, options: Optional[dict] = None) -> tuple[dict[str, Any], dict]:
        if options is None:
            options = (
                {"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids else {}
            )
        with record_function("env/reset_sim"):
            raw_obs, info = self.env.reset(seed=self.seed, options=options)
        with record_function("env/wrap_obs"):
            obs = self.wrap_obs(raw_obs, info)
        env_idx = options.get("env_idx")
        self._reset_metrics(env_idx)
        return obs, info

    def step(
        self,
        actions: torch.Tensor,
        auto_reset: bool = True,
        capture_obs: bool = True,
    ):
        base_env = self.env.unwrapped
        original_get_obs_sensor_data = None
        if not capture_obs:
            original_get_obs_sensor_data = base_env._get_obs_sensor_data
            base_env._get_obs_sensor_data = lambda *args, **kwargs: {}
        try:
            with record_function("env/sim_and_render"):
                raw_obs, raw_reward, terminations, truncations, info = self.env.step(actions)
        finally:
            if original_get_obs_sensor_data is not None:
                base_env._get_obs_sensor_data = original_get_obs_sensor_data

        obs = None
        if capture_obs:
            with record_function("env/wrap_obs"):
                obs = self.wrap_obs(raw_obs, info)

        with record_function("env/reward_metrics"):
            reward = self._calc_step_reward(raw_reward, info)
            info = self._record_metrics(reward, info)

        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if self.ignore_terminations:
            terminations[:] = False

        dones = torch.logical_or(terminations, truncations)
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, info = self._handle_auto_reset(dones, obs, info)

        return obs, reward, terminations, truncations, info

    def chunk_step(self, chunk_actions: torch.Tensor):
        """Execute multiple sub-steps from a chunked action prediction.

        Args:
            chunk_actions: [num_envs, chunk_size, action_dim]

        Returns:
            (obs, chunk_rewards, chunk_terminations, chunk_truncations, info)
        """
        chunk_size = chunk_actions.shape[1]
        chunk_rewards, chunk_terms, chunk_truncs = [], [], []
        render_last_only = self.cfg.get("render_obs_only_at_chunk_end", False)

        for i in range(chunk_size):
            capture_obs = not render_last_only or i == chunk_size - 1
            obs, reward, terms, truncs, info = self.step(
                chunk_actions[:, i],
                auto_reset=False,
                capture_obs=capture_obs,
            )
            chunk_rewards.append(reward)
            chunk_terms.append(terms)
            chunk_truncs.append(truncs)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        chunk_terms = torch.stack(chunk_terms, dim=1)
        chunk_truncs = torch.stack(chunk_truncs, dim=1)

        past_terms = chunk_terms.any(dim=1)
        past_truncs = chunk_truncs.any(dim=1)
        past_dones = torch.logical_or(past_terms, past_truncs)

        if past_dones.any() and self.auto_reset:
            obs, info = self._handle_auto_reset(past_dones, obs, info)

        # Collapse chunk-level done to last sub-step
        out_terms = torch.zeros_like(chunk_terms)
        out_terms[:, -1] = past_terms
        out_truncs = torch.zeros_like(chunk_truncs)
        out_truncs[:, -1] = past_truncs

        return obs, chunk_rewards, out_terms, out_truncs, info

    def _handle_auto_reset(self, dones, obs, info):
        final_obs = torch_clone_dict(obs)
        final_info = torch_clone_dict(info)
        env_idx = torch.arange(self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        if self.use_fixed_reset_state_ids:
            options["episode_id"] = self.reset_state_ids[env_idx]
        obs, info = self.reset(options=options)
        info["final_observation"] = final_obs
        info["final_info"] = final_info
        info["_final_info"] = dones
        info["_final_observation"] = dones
        return obs, info

    # ─── Video utils ───

    def capture_image(self) -> np.ndarray:
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]
        if len(img.shape) > 3 and len(img) > 1:
            img = tile_images(img, nrows=int(np.sqrt(self.num_envs)))
        elif len(img.shape) > 3:
            img = img[0]
        return img

    def flush_video(self, output_dir: str):
        if not self.render_images:
            return
        images_to_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
            fps=self.cfg.init_params.sim_config.control_freq,
            verbose=False,
        )
        self.video_cnt += 1
        self.render_images.clear()

    def sleep(self) -> None:
        """Park env between sync phases; PhysX scene stays allocated on GPU."""
        self.render_images.clear()
        self._reset_metrics()
