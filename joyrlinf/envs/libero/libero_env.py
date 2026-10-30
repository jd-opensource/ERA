"""LIBERO environment wrapper for joyrlinf (CPU subprocess sim, GPU obs tensors)."""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Optional, Union

import numpy as np

from joyrlinf.envs.libero.quiet import quiet_stderr, suppress_libero_startup_noise

suppress_libero_startup_noise()
import torch
from omegaconf import DictConfig, OmegaConf

from joyrlinf.envs.env_utils import list_of_dict_to_dict_of_list, to_tensor
from joyrlinf.envs.libero.utils import (
    get_benchmark_overridden,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from joyrlinf.envs.libero.venv import ReconfigureSubprocEnv

logger = logging.getLogger(__name__)


class LiberoEnv:
    """Vectorized LIBERO via subprocess workers; obs tensors on ``device`` for NCCL."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        device: torch.device | str | None = None,
    ):
        self.cfg = cfg
        self.seed_offset = seed_offset
        self.total_num_processes = max(1, total_num_processes)
        self.seed = int(cfg.seed) + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = int(cfg.get("group_size", 1))
        if num_envs < self.group_size or num_envs % self.group_size != 0:
            raise ValueError(
                f"num_envs={num_envs} must be >= group_size={self.group_size} "
                f"and divisible by group_size"
            )
        self.num_group = num_envs // self.group_size
        self.use_fixed_reset_state_ids = bool(cfg.get("use_fixed_reset_state_ids", False))
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        self.task_id_filter = cfg.get("task_id_filter", None)
        if self.task_id_filter is not None:
            self.task_id_filter = list(self.task_id_filter)

        self.ignore_terminations = bool(cfg.get("ignore_terminations", False))
        self.auto_reset = bool(cfg.get("auto_reset", True))
        self.use_rel_reward = bool(cfg.get("use_rel_reward", True))
        self.use_step_penalty = bool(cfg.get("use_step_penalty", False))
        self.reward_coef = float(cfg.get("reward_coef", 1.0))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        suite_name = cfg.get("task_suite_name", "libero_10")
        self.task_suite = get_benchmark_overridden(suite_name)()
        logger.info(
            "LiberoEnv: suite=%s num_envs=%d group_size=%d device=%s",
            suite_name, num_envs, self.group_size, self._device,
        )

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs, dtype=np.float32)
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.video_cfg = cfg.get("video_cfg", {})
        self.current_raw_obs: list | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    def _init_env(self) -> None:
        logger.info(
            "Spawning %d LIBERO subprocess envs (first startup is slow; "
            "no stderr spam — wait for 'LIBERO subprocess envs ready' on each rank)",
            self.num_envs,
        )
        self.env = ReconfigureSubprocEnv(self.get_env_fns())
        logger.info("LIBERO subprocess envs ready (%d on this rank)", self.num_envs)

    def get_env_fns(self) -> list:
        env_fn_params = self.get_env_fn_params()
        env_fns = []

        def make_env_fn(param: dict):
            def _fn(p=dict(param)):
                suppress_libero_startup_noise()
                with quiet_stderr():
                    from libero.libero.envs import OffScreenRenderEnv

                    seed = p.pop("seed", self.seed)
                    env = OffScreenRenderEnv(**p)
                env.seed(seed)
                return env

            return _fn

        for param in env_fn_params:
            env_fns.append(make_env_fn(param))
        return env_fns

    def get_env_fn_params(self, env_idx: Optional[np.ndarray] = None) -> list[dict]:
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)
        from libero.libero import get_libero_path

        bddl_root = get_libero_path("bddl_files")
        task_descriptions: list[str] = []

        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        env_fn_params = []
        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(
                    self.task_descriptions[env_id]
                    if hasattr(self, "task_descriptions")
                    else ""
                )
                continue

            task = self.task_suite.get_task(int(self.task_ids[env_id]))
            bddl_path = os.path.join(
                bddl_root, task.problem_folder, task.bddl_file
            )
            env_fn_params.append(
                {**base_env_args, "bddl_file_name": bddl_path, "seed": self.seed}
            )
            task_descriptions.append(task.language)

        self.task_descriptions = task_descriptions
        return env_fn_params

    def _compute_total_num_group_envs(self) -> None:
        self.trial_id_bins = []
        for task_id in range(self.task_suite.get_num_tasks()):
            self.trial_id_bins.append(
                len(self.task_suite.get_task_init_states(task_id))
            )
        self.total_num_group_envs = int(sum(self.trial_id_bins))
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

        self._valid_reset_state_ids = None
        if self.task_id_filter is not None:
            validated = sorted({int(t) for t in self.task_id_filter})
            num_tasks = len(self.trial_id_bins)
            for tid in validated:
                if tid < 0 or tid >= num_tasks:
                    raise ValueError(f"task_id {tid} out of range [0, {num_tasks - 1}]")
            ids = []
            for tid in validated:
                start = self.cumsum_trial_id_bins[tid - 1] if tid > 0 else 0
                end = self.cumsum_trial_id_bins[tid]
                ids.extend(range(start, end))
            self._valid_reset_state_ids = np.array(ids, dtype=np.int64)

    def update_reset_state_ids(self) -> None:
        if self.cfg.get("is_eval", False) or self.cfg.get(
            "use_ordered_reset_state_ids", False
        ):
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self) -> None:
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states: int) -> np.ndarray:
        if self.specific_reset_id is not None:
            return self.specific_reset_id * np.ones((num_reset_states,), dtype=int)
        if self._valid_reset_state_ids is not None:
            indices = self._generator.integers(
                0, len(self._valid_reset_state_ids), size=(num_reset_states,)
            )
            return self._valid_reset_state_ids[indices]
        return self._generator.integers(
            0, self.total_num_group_envs, size=(num_reset_states,)
        )

    def get_reset_state_ids_all(self) -> np.ndarray:
        if self._valid_reset_state_ids is not None:
            reset_state_ids = self._valid_reset_state_ids.copy()
        else:
            reset_state_ids = np.arange(self.total_num_group_envs)

        if not self.cfg.get("is_eval", False):
            self._generator_ordered.shuffle(reset_state_ids)

        if len(reset_state_ids) < self.total_num_processes:
            repeats = (self.total_num_processes // len(reset_state_ids)) + 1
            reset_state_ids = np.tile(reset_state_ids, repeats)

        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        reset_state_ids = reset_state_ids[:valid_size]
        return reset_state_ids.reshape(self.total_num_processes, -1)

    def _get_ordered_reset_state_ids(self, num_reset_states: int) -> np.ndarray:
        if self.specific_reset_id is not None:
            return self.specific_reset_id * np.ones((self.num_group,), dtype=int)
        if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
            self.reset_state_ids_all = self.get_reset_state_ids_all()
            self.start_idx = 0
        reset_state_ids = self.reset_state_ids_all[self.seed_offset][
            self.start_idx : self.start_idx + num_reset_states
        ]
        self.start_idx += num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(
        self, reset_state_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        task_ids, trial_ids = [], []
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if start_pivot <= reset_state_id < end_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot
        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx: np.ndarray) -> list:
        return [
            self.task_suite.get_task_init_states(self.task_ids[e])[
                self.trial_ids[e]
            ]
            for e in env_idx
        ]

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    @property
    def segment_success_once(self) -> torch.Tensor:
        return self._segment_success_once

    def _init_metrics(self) -> None:
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        self.success_episode_len = np.zeros(self.num_envs, dtype=np.int32)
        self._segment_success_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _reset_metrics(self, env_idx: Optional[np.ndarray] = None) -> None:
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.returns[mask] = 0.0
            self.success_episode_len[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.returns[:] = 0.0
            self.success_episode_len[:] = 0
            self._elapsed_steps[:] = 0
            self._segment_success_once.zero_()

    def _extract_image_and_state(self, obs: dict) -> dict:
        return {
            "full_image": get_libero_image(obs),
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }

    def _wrap_obs(self, obs_list: list) -> dict[str, Any]:
        images_and_states_list = [
            self._extract_image_and_state(obs) for obs in obs_list
        ]
        batched = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list),
            device=self.device,
        )
        main_images = torch.stack(
            [v.clone() for v in batched["full_image"]], dim=0
        ).to(torch.uint8)
        wrist_images = torch.stack(
            [v.clone() for v in batched["wrist_image"]], dim=0
        ).to(torch.uint8)
        states = batched["state"].to(dtype=torch.float32)
        return {
            "main_images": main_images,
            "wrist_images": wrist_images,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }

    def _reconfigure(self, reset_state_ids: np.ndarray, env_idx: np.ndarray) -> None:
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            env_id = int(env_id)
            task_changed = self.task_ids[env_id] != task_ids[j]
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
            if task_changed or not self.cfg.get("is_eval", False):
                reconfig_env_idx.append(env_id)

        if reconfig_env_idx:
            params = self.get_env_fn_params(np.array(reconfig_env_idx))
            self.env.reconfigure_env_fns(params, reconfig_env_idx)
            self.env.seed(self.seed * max(len(env_idx), 1))
            self.env.reset(id=reconfig_env_idx)
            init_state = self._get_reset_states(np.array(reconfig_env_idx))
            self.env.set_init_state(init_state=init_state, id=reconfig_env_idx)

    def reset(
        self, *, options: Optional[dict] = None
    ) -> tuple[dict[str, Any], dict]:
        options = options or {}
        env_idx = options.get("env_idx")
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        else:
            env_idx = np.asarray(env_idx)

        reset_state_ids = options.get("episode_id", options.get("reset_state_ids"))
        if self._is_start:
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            reset_state_ids = self._get_random_reset_state_ids(len(env_idx))

        self._reconfigure(np.asarray(reset_state_ids), env_idx)

        for _ in range(15):
            zero_actions = np.zeros((len(env_idx), 7), dtype=np.float32)
            if self.cfg.get("reset_gripper_open", True):
                zero_actions[:, -1] = -1.0
            raw_obs, _, _, _ = self.env.step(zero_actions, env_idx)
            if self.current_raw_obs is None:
                self.current_raw_obs = [None] * self.num_envs
            for i, idx in enumerate(env_idx):
                self.current_raw_obs[int(idx)] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        return obs, {}

    def step(
        self, actions: torch.Tensor | np.ndarray, auto_reset: bool = True
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _, terminations, info_batch = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_batch)
        truncations = self._elapsed_steps >= int(self.cfg.max_episode_steps)
        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(terminations)
        infos = self._record_metrics(step_reward, terminations, infos)

        term_t = torch.as_tensor(terminations, device=self.device, dtype=torch.bool)
        trunc_t = torch.as_tensor(truncations, device=self.device, dtype=torch.bool)

        if self.ignore_terminations:
            term_t = torch.zeros_like(term_t)

        dones = term_t | trunc_t
        if auto_reset and self.auto_reset and dones.any().item():
            obs, infos = self._handle_auto_reset(dones.cpu().numpy(), obs, infos)
            term_t = torch.zeros_like(term_t)
            trunc_t = torch.zeros_like(trunc_t)

        return (
            obs,
            torch.as_tensor(step_reward, device=self.device, dtype=torch.float32),
            term_t,
            trunc_t,
            infos,
        )

    def chunk_step(
        self, chunk_actions: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        chunk_size = chunk_actions.shape[1]
        chunk_rewards, chunk_terms, chunk_truncs = [], [], []
        obs, info = None, {}

        for i in range(chunk_size):
            obs, reward, terms, truncs, info = self.step(
                chunk_actions[:, i], auto_reset=False
            )
            chunk_rewards.append(reward)
            chunk_terms.append(terms)
            chunk_truncs.append(truncs)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        chunk_terms = torch.stack(chunk_terms, dim=1)
        chunk_truncs = torch.stack(chunk_truncs, dim=1)

        past_terms = chunk_terms.any(dim=1)
        past_truncs = chunk_truncs.any(dim=1)
        past_dones = past_terms | past_truncs

        if past_dones.any().item() and self.auto_reset:
            obs, info = self._handle_auto_reset(past_dones.cpu().numpy(), obs, info)

        out_terms = torch.zeros_like(chunk_terms)
        out_terms[:, -1] = past_terms
        out_truncs = torch.zeros_like(chunk_truncs)
        out_truncs[:, -1] = past_truncs

        return obs, chunk_rewards, out_terms, out_truncs, info

    def _handle_auto_reset(
        self, dones: np.ndarray, _final_obs: dict, infos: dict
    ) -> tuple[dict, dict]:
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        if self.cfg.get("is_eval", False):
            new_reset_state_ids = self._get_ordered_reset_state_ids(len(env_idx))
            self.reset_state_ids[env_idx] = new_reset_state_ids
        elif self.use_fixed_reset_state_ids:
            self.update_reset_state_ids()
        obs, infos = self.reset(
            options={
                "env_idx": env_idx,
                "reset_state_ids": (
                    self.reset_state_ids[env_idx]
                    if self.use_fixed_reset_state_ids or self.cfg.get("is_eval", False)
                    else None
                ),
            }
        )
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        dones_t = torch.as_tensor(dones, device=self.device, dtype=torch.bool)
        infos["_final_info"] = dones_t
        infos["_final_observation"] = dones_t
        return obs, infos

    def _calc_step_reward(self, terminations: np.ndarray) -> np.ndarray:
        step_penalty = -1.0 if self.use_step_penalty else 0.0
        reward = step_penalty + self.reward_coef * terminations.astype(np.float32)
        if self.use_rel_reward:
            diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return diff.astype(np.float32)
        return reward.astype(np.float32)

    def _record_metrics(
        self, step_reward: np.ndarray, terminations: np.ndarray, infos: dict
    ) -> dict:
        self.returns += step_reward * (~self.success_once)
        new_success_mask = terminations & ~self.success_once
        if new_success_mask.any():
            self.success_episode_len[new_success_mask] = self._elapsed_steps[
                new_success_mask
            ]
        self.success_once |= terminations
        self._segment_success_once |= torch.as_tensor(
            terminations, device=self.device, dtype=torch.bool
        )
        episode_len_for_reward = np.where(
            self.success_once, self.success_episode_len, self._elapsed_steps
        )
        infos["episode"] = {
            "success_once": self.success_once.copy(),
            "return": self.returns.copy(),
            "episode_len": self._elapsed_steps.copy(),
            "reward": self.returns.copy() / np.maximum(episode_len_for_reward, 1),
        }
        return infos
