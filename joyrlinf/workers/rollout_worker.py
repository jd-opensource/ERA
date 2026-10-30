"""Rollout workers for co-located and disaggregated deployment.

CombinedEnvRolloutWorker: env and rollout share the same GPU, obs pass by
    Python reference (zero-copy). Used when placement.colocated is True.

RolloutOnlyWorker: model inference only, communicates with a remote EnvWorker
    via NCCLPipe. Used when placement.colocated is False (fully disaggregated).
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Union
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import record_function
from omegaconf import DictConfig

from joyrlinf.core.direct_pipe import NCCLPipe
from joyrlinf.core.gpu_phase_gate import NullGpuPhaseGate
from joyrlinf.core.gpu_memory_guard import take_memory_snapshot
from joyrlinf.core.launcher import parse_placement
from joyrlinf.utils.logging_setup import log_at_rank
from joyrlinf.algorithms.embodied_advantages import pack_chunk_step_signals
from joyrlinf.core.types import (
    ChunkStepResult,
    EmbodiedRolloutBatch,
    PrefixKvHostPool,
    async_d2h_tensor_tree,
)
from joyrlinf.envs.action_utils import prepare_actions
from joyrlinf.envs.build_env import build_env
from joyrlinf.models.openpi_model import OpenPiModel
from joyrlinf.models.model_factory import build_vla_model

logger = logging.getLogger(__name__)

# When set, each profiled rollout section inserts a cuda.synchronize() so the
# accumulated wall-clock time reflects real GPU work per section. This slightly
# perturbs kernel pipelining, so keep it OFF for throughput runs and ON only
# when you want an accurate env-vs-inference breakdown.
#   JOYRLINF_ROLLOUT_TIMING=1  -> accurate per-section GPU timing (adds syncs)
_ROLLOUT_TIMING = os.environ.get("JOYRLINF_ROLLOUT_TIMING", "0") == "1"


def _resolve_rollout_buffer_d2h(cfg: DictConfig | dict[str, Any]) -> str:
    """Parse model-specific rollout_buffer_d2h → ``end`` | ``per_chunk``."""
    model_options = None
    if hasattr(cfg, "get"):
        actor = cfg.get("actor")
        if actor is not None and hasattr(actor, "get"):
            model_cfg = actor.get("model")
            if model_cfg is not None and hasattr(model_cfg, "get"):
                model_type = str(model_cfg.get("model_type", "openpi"))
                model_options = model_cfg.get(model_type)
                if model_options is None:
                    model_options = model_cfg.get("openpi")
        if model_options is None:
            model_options = cfg.get("openpi")
    mode = "end"
    if model_options is not None and hasattr(model_options, "get"):
        mode = str(model_options.get("rollout_buffer_d2h", "end")).strip().lower()
    if mode not in ("end", "per_chunk"):
        raise ValueError(
            "rollout_buffer_d2h must be 'end' or 'per_chunk', "
            f"got {mode!r}."
        )
    return mode


def _tensor_to_cpu_sync(t: torch.Tensor | None) -> torch.Tensor | None:
    if t is None:
        return None
    if isinstance(t, torch.Tensor) and t.is_cuda:
        return t.detach().cpu()
    return t


# Heavy KV fields offloaded during env.chunk_step overlap; everything else
# stays on GPU until buffer.to_dict (avoids end-of-rollout wall).
_PREFIX_KV_D2H_KEYS = ("prefix_kv", "prefix_pad_masks")


def _schedule_d2h_prefix_kv_fields(
    forward_inputs: dict[str, Any] | None,
    *,
    pin: bool = True,
) -> dict[str, Any]:
    """Non-blocking D2H of prefix_kv / prefix_pad_masks only (caller syncs)."""
    if not forward_inputs:
        return {}
    out: dict[str, Any] = {}
    for key in _PREFIX_KV_D2H_KEYS:
        val = forward_inputs.get(key)
        if val is not None:
            out[key] = async_d2h_tensor_tree(val, pin=pin)
    return out


def _apply_prefix_kv_offload(
    forward_inputs: dict[str, Any] | None,
    kv_cpu: dict[str, Any],
) -> dict[str, Any] | None:
    """Shallow-copy forward_inputs and replace KV keys with CPU (pinned) tensors."""
    if forward_inputs is None:
        return None
    fi = dict(forward_inputs)
    fi.update(kv_cpu)
    return fi


def ensure_kv_host_pool(
    pool: PrefixKvHostPool | None,
    forward_inputs: dict[str, Any] | None,
    *,
    capacity: int,
    pin: bool,
) -> PrefixKvHostPool | None:
    """Allocate/reuse host KV rows (may ``cudaHostAlloc``); call off the copy stream."""
    if not forward_inputs:
        return pool
    if (
        forward_inputs.get("prefix_kv") is None
        and forward_inputs.get("prefix_pad_masks") is None
    ):
        return pool
    if pool is None:
        pool = PrefixKvHostPool()
    pool.ensure(
        capacity,
        kv=forward_inputs.get("prefix_kv"),
        pad=forward_inputs.get("prefix_pad_masks"),
        pin=pin,
    )
    return pool


def schedule_kv_into_host_pool(
    pool: PrefixKvHostPool | None,
    row: int,
    forward_inputs: dict[str, Any] | None,
    *,
    capacity: int,
    pin: bool,
) -> tuple[PrefixKvHostPool | None, dict[str, Any]]:
    """Ensure host pool then async-copy row (caller owns stream + sync)."""
    pool = ensure_kv_host_pool(
        pool, forward_inputs, capacity=capacity, pin=pin,
    )
    if pool is None or not forward_inputs:
        return pool, {}
    if (
        forward_inputs.get("prefix_kv") is None
        and forward_inputs.get("prefix_pad_masks") is None
    ):
        return pool, {}
    return pool, pool.schedule_row(row, forward_inputs)


def finalize_host_kv_fields(
    pool: PrefixKvHostPool | None, buffer: EmbodiedRolloutBatch,
) -> None:
    if pool is None:
        return
    n = len(buffer.forward_inputs)
    if n == 0:
        return
    stacked = pool.stacked(n)
    if stacked:
        buffer.host_kv_fields = stacked


@contextmanager
def _prof_section(
    name: str,
    device: torch.device,
    accum: dict[str, float] | None,
) -> Iterator[None]:
    """Annotate a rollout section for the Chrome trace and (optionally) time it.

    - Always emits a ``torch.profiler.record_function(name)`` marker so the
      section is separable inside any active profiler trace (near-zero overhead
      when no profiler is running).
    - When *accum* is provided, adds the section's wall-clock time into
      ``accum[name]``. With ``JOYRLINF_ROLLOUT_TIMING=1`` a cuda.synchronize()
      is inserted first so async GPU kernels are attributed to this section.
    """
    with record_function(name):
        if accum is None:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if _ROLLOUT_TIMING and device.type == "cuda":
                torch.cuda.synchronize(device)
            accum[name] = accum.get(name, 0.0) + (time.perf_counter() - t0)


def _log_rollout_breakdown(
    accum: dict[str, float],
    *,
    rank: int,
    leader: int,
    mode: str,
    n_chunk_steps: int,
) -> None:
    """Emit a one-line env-vs-inference timing breakdown for the leader rank."""
    if not accum:
        return
    total = sum(accum.values())
    order = [
        "rollout/reset",
        "rollout/infer_predict",
        "rollout/prepare_actions",
        "rollout/env_chunk_step",
        "rollout/prepare_obs",
        "rollout/bootstrap_predict",
        "rollout/to_cpu",
    ]
    keys = order + [k for k in accum if k not in order]
    parts = []
    for k in keys:
        v = accum.get(k)
        if v is None:
            continue
        pct = 100.0 * v / total if total > 0 else 0.0
        parts.append(f"{k.split('/')[-1]}={v:.3f}s({pct:.0f}%)")
    infer = accum.get("rollout/infer_predict", 0.0) + accum.get(
        "rollout/bootstrap_predict", 0.0
    )
    env = accum.get("rollout/env_chunk_step", 0.0)
    log_at_rank(
        logger, rank, leader, logging.INFO,
        "[ROLLOUT-BREAKDOWN] mode=%s chunks=%d total=%.3fs "
        "| infer=%.3fs env=%.3fs (infer/env=%.2f) sync=%s | %s",
        mode, n_chunk_steps, total, infer, env,
        (infer / env if env > 0 else float("nan")),
        _ROLLOUT_TIMING, " ".join(parts),
    )


def _gpu_mem_mb() -> tuple[float, float]:
    """Return (allocated_MB, reserved_MB) for current device."""
    return (
        torch.cuda.memory_allocated() / 1e6,
        torch.cuda.memory_reserved() / 1e6,
    )


def _log_vram_share_after_predict(label: str, device: torch.device | int | None) -> None:
    """INFO: pt_alloc / frag / non_pt / free as MiB and % of device total."""
    if isinstance(device, torch.device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
    elif device is None:
        idx = torch.cuda.current_device()
    else:
        idx = int(device)
    snap = take_memory_snapshot(idx)
    logger.info("[%s] %s", label, snap.format_share())


def _infer_action_chunks_per_env(
    actions_np: np.ndarray, num_envs: int, action_dim: int
) -> int:
    a = np.asarray(actions_np)
    if a.ndim == 3:
        if a.shape[0] != num_envs or a.shape[2] != action_dim:
            raise ValueError(
                f"actions shape {a.shape} incompatible with "
                f"num_envs={num_envs} action_dim={action_dim}"
            )
        return int(a.shape[1])
    if a.ndim == 2:
        if a.shape[0] != num_envs:
            raise ValueError(
                f"actions batch {a.shape[0]} != num_envs {num_envs}"
            )
        if a.shape[1] % action_dim != 0:
            raise ValueError(
                f"actions width {a.shape[1]} not divisible by action_dim={action_dim}"
            )
        return int(a.shape[1] // action_dim)
    raise ValueError(f"Unexpected actions ndim={a.ndim} shape={a.shape}")


def _zero_action_level_dones(
    num_envs: int, num_action_chunks: int, device: torch.device,
) -> torch.Tensor:
    """RLinf bootstrap_step: [B, num_action_chunks] all False before first action."""
    return torch.zeros(
        num_envs, num_action_chunks, dtype=torch.bool, device=device,
    )


def _assert_action_chunks_match(
    actions_np: np.ndarray,
    num_envs: int,
    action_dim: int,
    expected_chunks: int,
) -> None:
    got = _infer_action_chunks_per_env(actions_np, num_envs, action_dim)
    if got != expected_chunks:
        raise RuntimeError(
            f"Policy outputs {got} action steps per env, but "
            f"actor.model.num_action_chunks={expected_chunks}. "
            f"Set num_action_chunks to {got} (and ensure "
            f"max_steps_per_rollout_epoch % {got} == 0)."
        )


class CombinedEnvRolloutWorker:
    """Runs env simulation + model inference on a single GPU.

    Zero-copy: obs from env.step() is passed directly to model.predict()
    as a Python dict of GPU tensors. No serialization, no IPC.
    """

    def __init__(self, rank: int, cfg: DictConfig, device: torch.device,
                 gpu_guard=None, defer_build: bool = False,
                 gpu_phase_gate=None):
        self.rank = rank
        self.cfg = cfg
        self.device = device
        self.gpu_guard = gpu_guard
        self.gpu_phase_gate = gpu_phase_gate or NullGpuPhaseGate()

        placement = parse_placement(cfg.cluster)
        num_rollout_gpus = len(placement.rollout_ranks)
        self._rollout_leader = placement.rollout_ranks[0]

        self.num_envs_per_gpu = cfg.env.train.total_num_envs // num_rollout_gpus
        self.num_action_chunks = cfg.actor.model.num_action_chunks
        self.action_dim = cfg.actor.model.get("action_dim", 7)
        self.policy_setup = cfg.actor.model.get("policy_setup", "widowx_bridge")
        self.env_type = cfg.env.train.get("env_type", "maniskill")
        self._wm_env_type = cfg.env.train.get(
            "wm_env_type", cfg.env.train.get("env_ft_type")
        )
        # Async PPO staleness: actor-step version of the currently loaded
        # weights (updated by the runner on each weight load).
        self.weight_version: int = 0
        # Track weight versions for any staleness-aware path: GAE/PPO always,
        # and async GRPO (proximal decoupled_actor) which also needs versions.
        self._track_versions = (
            str(cfg.algorithm.get("adv_type", "grpo")) == "gae"
            or str(cfg.algorithm.get("pipeline_mode", "sync")) == "async"
        )
        self._buffer_d2h_mode = _resolve_rollout_buffer_d2h(cfg)
        mem_cfg = cfg.get("memory", None) or {}
        self._release_after_predict = (
            bool(mem_cfg.get("release_after_predict", False))
            and device.type == "cuda"
        )
        # Background VRAM releaser (see _release_idle_vram).
        self._release_event = threading.Event()
        self._release_thread: threading.Thread | None = None
        self._d2h_stream: torch.cuda.Stream | None = None
        if self._buffer_d2h_mode == "per_chunk" and device.type == "cuda":
            self._d2h_stream = torch.cuda.Stream(device=device)
        # Reused across steps: pinned [T(+1), ...] host KV written per chunk.
        # When SharedHost IPC is enabled the pool binds to shm write slots.
        self._kv_host_pool: PrefixKvHostPool | None = None
        self._kv_shm_channel = None
        log_at_rank(
            logger, self.rank, self._rollout_leader, logging.DEBUG,
            "RolloutWorker init: rank=%d device=%s num_envs_per_gpu=%d chunks=%d "
            "action_dim=%d defer_build=%s buffer_d2h=%s",
            self.rank, self.device, self.num_envs_per_gpu, self.num_action_chunks,
            self.action_dim, defer_build, self._buffer_d2h_mode,
        )

        self.env = None
        self.model: OpenPiModel | None = None
        self._eval_env = None
        eval_section = cfg.env.get("eval", None)
        self._has_eval_cfg = eval_section is not None
        if self._has_eval_cfg:
            eval_total = int(eval_section.total_num_envs)
            self.eval_num_envs_per_gpu = eval_total // num_rollout_gpus
        else:
            self.eval_num_envs_per_gpu = self.num_envs_per_gpu

        if not defer_build:
            self.build_rollout()

        # Continuous rollout semantics (aligned with RLinf):
        # With auto_reset=True, only the first train epoch performs a full reset.
        # Later epochs continue simulation and rely on partial resets on done envs.
        self._rollout_env_ready = False
        self._cached_obs: dict[str, Any] | None = None
        self._cached_info: dict[str, Any] = {}

    def _train_auto_reset_enabled(self) -> bool:
        return bool(self.cfg.env.train.get("auto_reset", True))

    def _needs_full_reset_at_epoch_start(self, mode: str) -> bool:
        if mode != "train":
            return True
        if not self._train_auto_reset_enabled():
            return True
        return not self._rollout_env_ready

    @staticmethod
    def _zero_segment_success_metrics(env: Any) -> None:
        seg = getattr(env, "segment_success_once", None)
        if seg is not None and hasattr(seg, "zero_"):
            seg.zero_()

    @staticmethod
    def _empty_env_metric_sums() -> dict[str, float]:
        return {
            "num_trajectories": 0.0,
            "success_once_sum": 0.0,
            "return_sum": 0.0,
            "episode_len_sum": 0.0,
            "reward_sum": 0.0,
        }

    @staticmethod
    def _accumulate_rlinf_env_metrics(
        metric_sums: dict[str, float],
        info: dict[str, Any],
        chunk_dones: torch.Tensor,
    ) -> None:
        """Accumulate RLinf-style completed-trajectory env metrics."""
        if not isinstance(info, dict) or "final_info" not in info:
            return
        final_info = info.get("final_info")
        if not isinstance(final_info, dict):
            return
        episode = final_info.get("episode")
        if not isinstance(episode, dict):
            return

        mask = chunk_dones.detach().cpu().bool().reshape(-1)
        n_traj = int(mask.sum().item())
        if n_traj == 0:
            return

        metric_sums["num_trajectories"] += float(n_traj)
        for key, out_key in (
            ("success_once", "success_once_sum"),
            ("return", "return_sum"),
            ("episode_len", "episode_len_sum"),
            ("reward", "reward_sum"),
        ):
            if key not in episode:
                continue
            values = torch.as_tensor(episode[key]).detach().cpu().reshape(-1)
            metric_sums[out_key] += float(values[mask].float().sum().item())

    def _finish_train_rollout_epoch(self) -> None:
        """Keep RNG/reset semantics aligned with RLinf continuous rollout."""
        return

    @staticmethod
    def _park_env(env) -> None:
        """Park env between sync phases when supported (e.g. ManiSkill sleep)."""
        if hasattr(env, "sleep"):
            env.sleep()

    def build_rollout(
        self,
        weight_staging: dict[str, torch.Tensor] | None = None,
        cached_model: OpenPiModel | None = None,
    ) -> dict[str, float]:
        """Create env + inference model; optionally load weights from CPU staging.

        When *cached_model* is provided (sync colocated reuse), skip the slow
        safetensors rebuild and only onload weights to GPU + ``load_state_dict``.
        """
        timings: dict[str, float] = {}
        if self.env is None:
            t0 = time.time()
            logger.debug("CombinedEnvRolloutWorker building env")
            self.env = self._build_env()
            timings["build_env"] = time.time() - t0
            log_at_rank(
                logger, self.rank, self._rollout_leader, logging.INFO,
                "CombinedEnvRolloutWorker env ready (%.2fs)", timings["build_env"],
            )
        else:
            t0 = time.time()
            self._park_env(self.env)
            timings["reuse_env"] = time.time() - t0
            logger.debug(
                "CombinedEnvRolloutWorker reusing pooled env (sleep %.2fs)",
                timings["reuse_env"],
            )
        if self.model is None:
            t0 = time.time()
            if cached_model is not None:
                logger.debug("CombinedEnvRolloutWorker restoring cached inference model")
                cached_model.onload_to_device(self.device)
                self.model = cached_model
                timings["restore_model"] = time.time() - t0
                log_at_rank(
                    logger, self.rank, self._rollout_leader, logging.INFO,
                    "CombinedEnvRolloutWorker model restored (%.2fs)",
                    timings["restore_model"],
                )
            else:
                logger.debug("CombinedEnvRolloutWorker building model from model_path")
                self.model = self._build_model()
                timings["build_model"] = time.time() - t0
                log_at_rank(
                    logger, self.rank, self._rollout_leader, logging.INFO,
                    "CombinedEnvRolloutWorker model ready (%.2fs)",
                    timings["build_model"],
                )
        if weight_staging is not None:
            t0 = time.time()
            self.load_weights(weight_staging)
            timings["load_weights"] = time.time() - t0
            logger.debug(
                "CombinedEnvRolloutWorker weights loaded from staging (%.2fs)",
                timings["load_weights"],
            )
        return timings

    def teardown_rollout(
        self,
        *,
        release_cuda_cache: bool = False,
        keep_model: bool = False,
        keep_env: bool = False,
    ) -> OpenPiModel | None:
        """Release rollout resources to free GPU memory for actor phase.

        When *keep_model* is True, offload inference to CPU and return it for reuse
        on the next rollout phase (avoids rebuilding from ``model_path``).

        When *keep_env* is True, park the PhysX env (no ``close()``) for reuse on
        the next rollout phase (avoids ~10s rebuild and allocator fragmentation).
        """
        if self._eval_env is not None:
            try:
                self._eval_env.env.close()
            except Exception:
                pass
            del self._eval_env
            self._eval_env = None
        if self.env is not None and not keep_env:
            try:
                self.env.env.close()
            except Exception:
                pass
            del self.env
            self.env = None
        elif self.env is not None and keep_env:
            self._park_env(self.env)
        kept_model: OpenPiModel | None = None
        if self.model is not None:
            if keep_model:
                self.model.offload_to_cpu()
                kept_model = self.model
                self.model = None
            else:
                del self.model
                self.model = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            if release_cuda_cache:
                torch.cuda.empty_cache()
        gc.collect()
        logger.info(
            "CombinedEnvRolloutWorker teardown complete: rank=%d "
            "release_cuda_cache=%s keep_model=%s keep_env=%s",
            self.rank, release_cuda_cache, keep_model, keep_env,
        )
        return kept_model

    def _ensure_rollout_ready(self) -> None:
        if self.env is None or self.model is None:
            raise RuntimeError(
                "CombinedEnvRolloutWorker not built; call build_rollout() first."
            )

    def _release_idle_vram(self, label: str) -> None:
        """Signal the background releaser to return predict's idle reserve.

        The predict transient peak (attention scores + MLP activations) leaves
        several GiB of idle reserve behind; under MPS the colocated actor
        cannot use it unless this process actually releases it. The
        empty_cache (~30ms with ``expandable_segments:True``: unmap idle
        pages only, VA ranges stay reserved, live cuBLAS workspace untouched)
        runs on a daemon thread so signal_train_ok fires immediately — the
        release completes long before the actor's microbatch peak ramps up.
        Back-to-back signals coalesce on the event.
        """
        if not self._release_after_predict:
            return
        if self._release_thread is None:
            self._release_thread = threading.Thread(
                target=self._idle_vram_release_loop,
                name=f"rollout-vram-release-{self.rank}",
                daemon=True,
            )
            self._release_thread.start()
        self._release_event.set()

    def _idle_vram_release_loop(self) -> None:
        while True:
            self._release_event.wait()
            self._release_event.clear()
            start = time.perf_counter()
            try:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            except Exception:
                logger.exception(
                    "[rank %d] background empty_cache failed", self.rank,
                )
                continue
            logger.debug(
                "[rank %d] async empty_cache: idle reserve released in %.1fms",
                self.rank, (time.perf_counter() - start) * 1e3,
            )

    def _build_env(self):
        placement = parse_placement(self.cfg.cluster)
        return build_env(
            self.cfg.env.train,
            num_envs=self.num_envs_per_gpu,
            rank=self.rank,
            placement=placement,
            device=self.device,
            colocated=True,
        )

    def _build_model(self):
        return build_vla_model(
            self.cfg.actor.model,
            self.device,
            mode="inference",
        )

    def _get_eval_env(self):
        """Lazy-build eval env to avoid doubling PhysX memory at startup."""
        if self._eval_env is None:
            eval_cfg = self.cfg.env.get("eval", self.cfg.env.train)
            placement = parse_placement(self.cfg.cluster)
            logger.info(
                "Building eval env: rank=%d num_envs=%d",
                self.rank, self.eval_num_envs_per_gpu,
            )
            self._eval_env = build_env(
                eval_cfg,
                num_envs=self.eval_num_envs_per_gpu,
                rank=self.rank,
                placement=placement,
                device=self.device,
                colocated=True,
            )

        return self._eval_env

    @staticmethod
    def make_train_runner_metrics(success_count: int, denominator: int) -> dict[str, float]:
        """Fraction of train envs with ≥1 success in this rollout step.

        *success_count* is cluster-wide (all rollout ranks); *denominator* must
        match the terminal log: ``rollout_epoch * total_num_envs``.
        """
        denom = max(int(denominator), 1)
        return {
            "train/success_rate": float(success_count) / denom,
            "success_rate": float(success_count) / denom,
        }

    def load_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._ensure_rollout_ready()
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        logger.debug("loaded new weights (%d keys)", len(state_dict))

    def _make_versions_tensor(self) -> torch.Tensor | None:
        """Version tensor with the loaded weight version (PPO/gae only)."""
        if not self._track_versions:
            return None
        version_chunks = (
            1
            if str(self.cfg.algorithm.get("reward_type", "action_level")) == "chunk_level"
            else self.num_action_chunks
        )
        return torch.full(
            (self.num_envs_per_gpu, version_chunks),
            float(self.weight_version),
            dtype=torch.float32,
            device=self.device,
        )

    def set_kv_shm_channel(self, channel) -> None:
        """Attach a SharedHostKvChannel so per-chunk D2H lands in shm slots."""
        self._kv_shm_channel = channel
        if self._kv_host_pool is not None:
            self._kv_host_pool.set_shm_channel(channel)

    def begin_kv_shm_epoch(self) -> None:
        """Wait for a free write slot and re-bind the host pool (owner side)."""
        ch = self._kv_shm_channel
        if ch is None:
            return
        ch.begin_epoch_write()
        if ch.ready:
            if self._kv_host_pool is None:
                self._kv_host_pool = PrefixKvHostPool()
                self._kv_host_pool.set_shm_channel(ch)
            ch.bind_pool(self._kv_host_pool)

    def _ensure_kv_host_pool(
        self,
        forward_inputs: dict[str, Any] | None,
        *,
        capacity: int,
        pin: bool,
    ) -> None:
        if self._kv_host_pool is None:
            self._kv_host_pool = PrefixKvHostPool()
            if self._kv_shm_channel is not None:
                self._kv_host_pool.set_shm_channel(self._kv_shm_channel)
        elif self._kv_shm_channel is not None:
            self._kv_host_pool.set_shm_channel(self._kv_shm_channel)
        # Private pool: pin=True → full pinned rows.
        # SharedHost: slots stay pageable; pin=True → one-row pinned staging
        # (async D2H), then commit_staged_row H2H into shm.
        self._kv_host_pool = ensure_kv_host_pool(
            self._kv_host_pool,
            forward_inputs,
            capacity=capacity,
            pin=pin,
        )

    def _copy_kv_row_into_host_pool(
        self, row: int, forward_inputs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Enqueue D2H into staging or pool[row] on the *current* CUDA stream."""
        if self._kv_host_pool is None or not forward_inputs:
            return {}
        if (
            forward_inputs.get("prefix_kv") is None
            and forward_inputs.get("prefix_pad_masks") is None
        ):
            return {}
        return self._kv_host_pool.schedule_row(row, forward_inputs)

    def _commit_kv_row_from_staging(self, row: int) -> dict[str, Any]:
        """After D2H sync: flush pinned staging → shm[row] (no-op if unused)."""
        if self._kv_host_pool is None:
            return {}
        return self._kv_host_pool.commit_staged_row(row)

    def _finalize_host_kv_fields(self, buffer: EmbodiedRolloutBatch) -> None:
        finalize_host_kv_fields(self._kv_host_pool, buffer)

    def _append_bootstrap_predict(
        self,
        buffer: EmbodiedRolloutBatch,
        final_result: dict[str, Any],
    ) -> None:
        """Keep only the bootstrap value (GAE); drop its forward_inputs/KV.

        Training samples cover chunk rows [0, T) — the actor's
        flatten_for_training truncates every [T+1]-row field back to [T]
        before use, so persisting the bootstrap predict's prefix KV (one
        full host-pool/shm row, ~2.2GiB at 128 envs) or its forward_inputs
        would be dead weight on both host RAM and the batch transfer.
        """
        if final_result.get("prev_values") is not None:
            buffer.prev_values.append(final_result["prev_values"])

    def set_global_step(self, global_step: int) -> None:
        self._ensure_rollout_ready()
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def rollout_epoch(
        self, mode: str = "train", *, return_timeline: bool = False,
    ) -> (
        tuple[dict[str, Any], dict[str, int]]
        | tuple[dict[str, Any], dict[str, int], dict[str, Any]]
    ):
        self._ensure_rollout_ready()
        """Run one rollout epoch: optional reset -> step loop -> collect batch.

        Returns:
            (batch dict for actor training, stats dict with ``local_success_count``:
            number of envs on this GPU that achieved task success at least once
            during the segment, via ``ManiSkillEnv.success_once``.)
        """
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.num_action_chunks
        )
        compute_values = self.cfg.algorithm.get("adv_type") == "gae"
        logger.debug(
            "Rollout epoch start: mode=%s n_chunk_steps=%d compute_values=%s",
            mode, n_chunk_steps, compute_values,
        )

        buffer = EmbodiedRolloutBatch(rollout_epochs=1)
        # Per-section wall-clock accumulator for the env-vs-inference breakdown.
        sect: dict[str, float] = {}
        env_metric_sums = self._empty_env_metric_sums()

        alloc0, rsv0 = _gpu_mem_mb()
        logger.debug("VRAM before reset: alloc=%.0fM reserved=%.0fM", alloc0, rsv0)
        if self.gpu_guard is not None:
            self.gpu_guard.log_profile("rollout_epoch_start")

        # Reset / continue env (aligned with RLinf continuous rollout semantics).
        # NOTE: in async mode, hidden backpressure often surfaces here: pending
        # NCCL work from previous step can block before env.reset() actually runs.
        t_pre_reset = time.time()
        env_reset_start_ns = None
        env_reset_end_ns = None
        t_update_ids = 0.0
        reset_call = 0.0
        need_full_reset = self._needs_full_reset_at_epoch_start(mode)

        if need_full_reset:
            logger.debug("env.reset() starting (full)")
            t_reset0 = time.time()
            env_reset_start_ns = time.time_ns()
            with _prof_section("rollout/reset", self.device, sect):
                obs, info = self.env.reset()
            env_reset_end_ns = time.time_ns()
            reset_call = time.time() - t_reset0
            if mode == "train":
                self._rollout_env_ready = True
        else:
            if self._cached_obs is None:
                raise RuntimeError(
                    "Continuing rollout without cached obs; call full reset first."
                )
            obs, info = self._cached_obs, self._cached_info
            self._zero_segment_success_metrics(self.env)
            logger.debug("env.reset() skipped (auto_reset continuous rollout)")

        t_pre_reset_total = time.time() - t_pre_reset
        alloc1, rsv1 = _gpu_mem_mb()
        logger.debug("VRAM after reset: alloc=%.0fM reserved=%.0fM (delta=+%.0fM)", alloc1, rsv1, alloc1 - alloc0)
        logger.debug(
            "Rollout pre-reset timing: total=%.2fs update_ids=%.2fs "
            "pre_reset_wait=%.2fs reset_call=%.2fs full_reset=%s",
            t_pre_reset_total,
            t_update_ids,
            max(
                (t_pre_reset_total - t_update_ids - reset_call)
                if need_full_reset
                else 0.0,
                0.0,
            ),
            reset_call,
            need_full_reset,
        )

        # Pre-process first obs for model
        with _prof_section("rollout/prepare_obs", self.device, sect):
            env_obs = self._prepare_env_obs(obs)

        # RLinf env_worker.bootstrap_step: leading zero dones row before any action.
        # dones layout: [pre-chunk0, post-chunk0, ..., post-chunk_{N-1}] (N+1 rows).
        zero_device = (
            torch.device("cpu")
            if self._buffer_d2h_mode == "per_chunk"
            else self.device
        )
        zero_dones = _zero_action_level_dones(
            self.num_envs_per_gpu, self.num_action_chunks, zero_device,
        )
        buffer.dones.append(zero_dones)
        buffer.terminations.append(torch.zeros_like(zero_dones))
        buffer.truncations.append(torch.zeros_like(zero_dones))

        per_chunk_d2h = self._buffer_d2h_mode == "per_chunk"
        pin = self.device.type == "cuda"
        # One host KV row per chunk predict. The bootstrap predict's KV is
        # deliberately not stored (see _append_bootstrap_predict).
        kv_capacity = n_chunk_steps

        interactions_timeline = []
        for chunk_idx in range(n_chunk_steps):
            interaction_start_ns = time.time_ns()
            # Predict ⊥ train: wait for GPU grant before model.predict.
            self.gpu_phase_gate.wait_predict_ok()
            rollout_infer_start_ns = time.time_ns()
            with _prof_section("rollout/infer_predict", self.device, sect):
                actions_np, result = self.model.predict(
                    env_obs, mode=mode, compute_values=compute_values,
                )
            rollout_infer_end_ns = time.time_ns()
            # Hand back predict's idle reserve before granting the actor.
            self._release_idle_vram(f"chunk_{chunk_idx}_post_predict")
            # Release GPU to actor train; cached actions drive chunk_step only.
            self.gpu_phase_gate.signal_train_ok()
            if chunk_idx == 0:
                a2, r2 = _gpu_mem_mb()
                logger.debug("VRAM after 1st predict: alloc=%.0fM reserved=%.0fM", a2, r2)
            _log_vram_share_after_predict(
                f"chunk_{chunk_idx}/{n_chunk_steps}_post_predict",
                self.device,
            )
            if self.gpu_guard is not None:
                self.gpu_guard.log_profile(
                    f"chunk_{chunk_idx}/{n_chunk_steps}_post_predict",
                    level=logging.DEBUG,
                )

            _assert_action_chunks_match(
                actions_np, self.num_envs_per_gpu, self.action_dim,
                self.num_action_chunks,
            )
            with _prof_section("rollout/prepare_actions", self.device, sect):
                chunk_actions = prepare_actions(
                    actions_np,
                    env_type=self.env_type,
                    model_type=self.cfg.actor.model.get("model_type"),
                    num_action_chunks=self.num_action_chunks,
                    action_dim=self.action_dim,
                    policy=self.policy_setup,
                    device=self.device,
                    wm_env_type=self._wm_env_type,
                )

            versions = self._make_versions_tensor()
            pending_kv: dict[str, Any] | None = None
            if per_chunk_d2h:
                # Only prefix_kv(+pad) D2H on copy stream; overlap with env.
                fi0 = result.get("forward_inputs")
                with _prof_section("rollout/prefix_kv_d2h", self.device, sect):
                    self._ensure_kv_host_pool(
                        fi0, capacity=kv_capacity, pin=pin,
                    )
                    if self._d2h_stream is not None:
                        # D2H stream must wait for predict (default stream) that
                        # produced prefix_kv before enqueueing non_blocking copy_.
                        self._d2h_stream.wait_stream(
                            torch.cuda.current_stream(self.device)
                        )
                        with torch.cuda.stream(self._d2h_stream):
                            pending_kv = self._copy_kv_row_into_host_pool(
                                chunk_idx, fi0,
                            )
                    else:
                        pending_kv = self._copy_kv_row_into_host_pool(
                            chunk_idx, fi0,
                        )

            env_step_start_ns = time.time_ns()
            with _prof_section("rollout/env_chunk_step", self.device, sect):
                obs, chunk_rewards, chunk_terms, chunk_truncs, info = (
                    self.env.chunk_step(chunk_actions)
                )
            env_step_end_ns = time.time_ns()
            chunk_dones = torch.logical_or(chunk_terms, chunk_truncs)[:, -1]
            self._accumulate_rlinf_env_metrics(env_metric_sums, info, chunk_dones)
            if self.gpu_guard is not None:
                self.gpu_guard.log_profile(
                    f"chunk_{chunk_idx}/{n_chunk_steps}_post_step",
                    level=logging.DEBUG,
                )
                if not self.gpu_guard.check(
                    label=f"chunk_{chunk_idx}_post_step"
                ):
                    logger.error(
                        "GPU memory CRITICAL after chunk_step %d/%d, "
                        "consider reducing num_envs or increasing "
                        "pytorch_fraction",
                        chunk_idx, n_chunk_steps,
                    )
            with _prof_section("rollout/prepare_obs", self.device, sect):
                env_obs = self._prepare_env_obs(obs)

            step_rewards, step_dones, step_terms, step_truncs = (
                pack_chunk_step_signals(chunk_rewards, chunk_terms, chunk_truncs)
            )

            if per_chunk_d2h:
                with _prof_section("rollout/prefix_kv_d2h", self.device, sect):
                    if self._d2h_stream is not None:
                        self._d2h_stream.synchronize()
                    # SharedHost: flush pinned staging → pageable shm[row].
                    # Private pool: no-op that returns durable row views.
                    pending_kv = self._commit_kv_row_from_staging(chunk_idx)
                assert pending_kv is not None
                with _prof_section("rollout/buffer_append", self.device, sect):
                    fi = _apply_prefix_kv_offload(
                        result.get("forward_inputs"), pending_kv,
                    )
                    buffer.append(
                        ChunkStepResult(
                            prev_logprobs=result["prev_logprobs"],
                            prev_values=result.get("prev_values"),
                            dones=_tensor_to_cpu_sync(step_dones),
                            terminations=_tensor_to_cpu_sync(step_terms),
                            truncations=_tensor_to_cpu_sync(step_truncs),
                            rewards=_tensor_to_cpu_sync(step_rewards),
                            versions=versions,
                            forward_inputs=fi,
                        )
                    )
                    del result, pending_kv, fi
            else:
                with _prof_section("rollout/buffer_append", self.device, sect):
                    buffer.append(
                        ChunkStepResult(
                            prev_logprobs=result["prev_logprobs"],
                            prev_values=result.get("prev_values"),
                            dones=step_dones,
                            terminations=step_terms,
                            truncations=step_truncs,
                            rewards=step_rewards,
                            versions=versions,
                            forward_inputs=result.get("forward_inputs"),
                        )
                    )
            interaction_end_ns = time.time_ns()
            interactions_timeline.append({
                "interaction_idx": chunk_idx,
                    "interaction_time": {
                        "start_wall_ns": interaction_start_ns,
                        "end_wall_ns": interaction_end_ns,
                    },
                    "rollout_infer": {
                        "start_wall_ns": rollout_infer_start_ns,
                        "end_wall_ns": rollout_infer_end_ns,
                    },
                    "env_step": {
                        "start_wall_ns": env_step_start_ns,
                        "end_wall_ns": env_step_end_ns,
                    },
            })

        bootstrap_infer_start_ns = time.time_ns()
        # Bootstrap value on last obs (GAE). RLinf appends prev_values only here.
        self.gpu_phase_gate.wait_predict_ok()
        with _prof_section("rollout/bootstrap_predict", self.device, sect):
            _, final_result = self.model.predict(
                env_obs, mode=mode, compute_values=compute_values,
            )
        bootstrap_infer_end_ns = time.time_ns()
        self._release_idle_vram("bootstrap_post_predict")
        _log_vram_share_after_predict("bootstrap_post_predict", self.device)
        # No more predicts this epoch — actor may finish remaining microbatches.
        self.gpu_phase_gate.signal_epoch_predicts_done()
        self._append_bootstrap_predict(buffer, final_result)
        self._finalize_host_kv_fields(buffer)

        alloc_peak, rsv_peak = _gpu_mem_mb()
        logger.debug(
            "VRAM before to_cpu: alloc=%.0fM reserved=%.0fM (buffer has %d steps)",
            alloc_peak, rsv_peak, len(buffer.rewards),
        )
        with _prof_section("rollout/to_cpu", self.device, sect):
            batch = buffer.to_dict(device="cpu")
        # Idle-cache release is handled per predict by _release_idle_vram()
        # (memory.release_after_predict). With expandable_segments:True,
        # empty_cache() only unmaps idle pages and keeps the live cuBLAS
        # workspace, so the old autotune cold-start concern (observed under
        # max_split_size_mb-only configs) no longer applies. Note that
        # garbage_collection_threshold is inert while pytorch_fraction=0:
        # it measures against the whole-device cap, which is never reached.
        alloc_after, rsv_after = _gpu_mem_mb()
        logger.debug(
            "Rollout VRAM profile: init=%.0fM → after_reset=%.0fM → "
            "after_1st_predict=%.0fM → buffer_peak=%.0fM → after_to_cpu=%.0fM "
            "(n_chunk_steps=%d, num_envs=%d)",
            alloc0, alloc1,
            a2 if n_chunk_steps > 0 else alloc1,
            alloc_peak, alloc_after,
            n_chunk_steps, self.num_envs_per_gpu,
        )
        if self.gpu_guard is not None:
            self.gpu_guard.log_profile("rollout_epoch_end")

        # Cache last env state for continuous rollout.
        self._cached_obs = obs
        self._cached_info = info
        if mode == "train":
            self._finish_train_rollout_epoch()
        local_success = int(self.env.segment_success_once.sum().item())
        _log_rollout_breakdown(
            sect, rank=self.rank, leader=self._rollout_leader,
            mode=mode, n_chunk_steps=n_chunk_steps,
        )
        stats = {"local_success_count": local_success}
        stats.update(env_metric_sums)
        env_rollout_timeline = {
            "env_reset": {
                "performed": need_full_reset,
                "start_wall_ns": env_reset_start_ns,
                "end_wall_ns": env_reset_end_ns,
            },
            "interactions": interactions_timeline,
            "bootstrap_infer": {
                "start_wall_ns": bootstrap_infer_start_ns,
                "end_wall_ns": bootstrap_infer_end_ns,
            },
        }
        if return_timeline:
            return batch, stats, env_rollout_timeline
        return batch, stats

    def rollout_eval(self, *, use_train_env: bool = False) -> dict[str, float]:
        """Run evaluation rollout on eval env (or train env if eval not configured).

        Args:
            use_train_env: If True, never lazy-build a separate eval PhysX env.
                Used in sync colocated mode where train env + eval env cannot fit.
        """
        self._ensure_rollout_ready()
        eval_cfg = self.cfg.env.get("eval", self.cfg.env.train)
        if self._has_eval_cfg and not use_train_env:
            eval_env = self._get_eval_env()
            num_envs = self.eval_num_envs_per_gpu
        else:
            if self._has_eval_cfg and use_train_env:
                logger.info(
                    "rollout_eval: using train env (rank=%d, num_envs=%d); "
                    "sync colocated skips lazy eval PhysX",
                    self.rank, self.num_envs_per_gpu,
                )
            eval_env = self.env
            num_envs = self.num_envs_per_gpu

        n_steps = eval_cfg.get(
            "max_steps_per_rollout_epoch",
            eval_cfg.get("max_episode_steps", 80),
        )
        n_chunks = n_steps // self.num_action_chunks

        obs, info = eval_env.reset()
        env_obs = self._prepare_env_obs(obs)
        total_rewards = torch.zeros(num_envs, device=self.device)

        for _ in range(n_chunks):
            actions_np, _ = self.model.predict(
                env_obs, mode="eval", compute_values=False,
            )
            _assert_action_chunks_match(
                actions_np, num_envs, self.action_dim,
                self.num_action_chunks,
            )
            chunk_actions = prepare_actions(
                actions_np, env_type=self.env_type,
                model_type=self.cfg.actor.model.get("model_type"),
                num_action_chunks=self.num_action_chunks,
                action_dim=self.action_dim, policy=self.policy_setup,
                device=self.device,
                wm_env_type=self._wm_env_type,
            )
            obs, chunk_rewards, _, _, info = eval_env.chunk_step(chunk_actions)
            env_obs = self._prepare_env_obs(obs)
            total_rewards += chunk_rewards.sum(dim=1)

        success = eval_env.segment_success_once.float().mean().item()
        mean_return = total_rewards.mean().item()
        return {"eval/success_rate": success, "eval/mean_return": mean_return}

    def _prepare_env_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Format obs dict for model input. Adds missing keys expected by openpi."""
        # try:
        #     print(f'The keys of obs are {obs.keys()}')
        #     print(f'The obs is {obs}')
        # except Exception as e:
        #     print(f'An error occurred: {e}')
        #     print(f'The error obs are {obs}')
        #     print(f'The len of obs is {len(obs)}')
        prepared = {
            "main_images": obs.get("main_images"),
            "states": obs.get("states"),
            "task_descriptions": obs.get("task_descriptions"),
            "wrist_images": obs.get("wrist_images"),
            "extra_view_images": obs.get("extra_view_images"),
        }
        return prepared


class RolloutOnlyWorker:
    """Model inference worker for fully disaggregated deployment.

    Communicates with a remote EnvWorker via NCCLPipe to receive observations
    and send actions. No local environment.
    """

    def __init__(self, rank: int, cfg: DictConfig, device: torch.device):
        self.rank = rank
        self.cfg = cfg
        self.device = device

        placement = parse_placement(cfg.cluster)
        num_rollout_gpus = len(placement.rollout_ranks)
        num_env_gpus = len(placement.env_ranks)

        self.num_envs_per_gpu = cfg.env.train.total_num_envs // num_env_gpus
        self.num_action_chunks = cfg.actor.model.num_action_chunks
        self.action_dim = cfg.actor.model.get("action_dim", 7)
        self.policy_setup = cfg.actor.model.get("policy_setup", "widowx_bridge")
        self.env_type = cfg.env.train.get("env_type", "maniskill")
        self._wm_env_type = cfg.env.train.get(
            "wm_env_type", cfg.env.train.get("env_ft_type")
        )
        self.weight_version: int = 0
        # Track weight versions for any staleness-aware path: GAE/PPO always,
        # and async GRPO (proximal decoupled_actor) which also needs versions.
        self._track_versions = (
            str(cfg.algorithm.get("adv_type", "grpo")) == "gae"
            or str(cfg.algorithm.get("pipeline_mode", "sync")) == "async"
        )
        logger.info(
            "RolloutOnlyWorker init: rank=%d device=%s num_envs_per_gpu=%d",
            self.rank, self.device, self.num_envs_per_gpu,
        )

        logger.info("RolloutOnlyWorker building model")
        self.model = build_vla_model(
            cfg.actor.model,
            device,
            mode="inference",
        )
        logger.info("RolloutOnlyWorker model ready")

    # Keys the model's obs_processor expects; missing ones are set to None.
    _OBS_KEYS = ("main_images", "states", "task_descriptions",
                 "wrist_images", "extra_view_images")

    def _fill_obs(self, tensors: dict, task_descriptions: list) -> dict:
        """Ensure all expected obs keys exist (model crashes on missing keys)."""
        obs = {k: tensors.get(k) for k in self._OBS_KEYS}
        obs["task_descriptions"] = task_descriptions
        return obs

    def load_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        logger.debug("loaded new weights (%d keys)", len(state_dict))

    def _make_versions_tensor(self) -> torch.Tensor | None:
        """Version tensor with the loaded weight version (PPO/gae only)."""
        if not self._track_versions:
            return None
        version_chunks = (
            1
            if str(self.cfg.algorithm.get("reward_type", "action_level")) == "chunk_level"
            else self.num_action_chunks
        )
        return torch.full(
            (self.num_envs_per_gpu, version_chunks),
            float(self.weight_version),
            dtype=torch.float32,
            device=self.device,
        )

    def rollout_epoch(
        self,
        recv_pipe: NCCLPipe,
        send_pipe: NCCLPipe,
        env_rollout_group: dist.ProcessGroup,
        env_rank: int,
        mode: str = "train",
        *,
        return_timeline: bool = False,
    ) -> (
        tuple[dict[str, Any], dict[str, int]]
        | tuple[dict[str, Any], dict[str, int], dict[str, Any]]
    ):
        """Run one rollout epoch by communicating with remote EnvWorker.

        Args:
            recv_pipe: NCCLPipe env->rollout for receiving obs/results.
            send_pipe: NCCLPipe rollout->env for sending actions.
            env_rollout_group: process group for broadcast_object_list with env.
            env_rank: global rank of the paired env worker (broadcast source).
            mode: "train" or "eval".

        Returns:
            (batch dict for actor training, stats with ``local_success_count``).
        """
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.num_action_chunks
        )
        compute_values = self.cfg.algorithm.get("adv_type") == "gae"
        logger.debug(
            "RolloutOnlyWorker epoch: n_chunk_steps=%d compute_values=%s",
            n_chunk_steps, compute_values,
        )

        buffer = EmbodiedRolloutBatch(rollout_epochs=1)

        # 1. Recv task_descriptions (strings) via broadcast_object_list from env
        task_descs_list = [None]
        dist.broadcast_object_list(
            task_descs_list, src=env_rank, group=env_rollout_group)
        task_descriptions = task_descs_list[0] or []

        # 2. Recv initial obs tensors from env
        obs_tensors = recv_pipe.recv_dict()
        env_obs = self._fill_obs(obs_tensors, task_descriptions)

        zero_dones = _zero_action_level_dones(
            self.num_envs_per_gpu, self.num_action_chunks, self.device,
        )
        buffer.dones.append(zero_dones)
        buffer.terminations.append(torch.zeros_like(zero_dones))
        buffer.truncations.append(torch.zeros_like(zero_dones))

        # 3. Chunk step loop
        interactions_timeline = []
        for chunk_idx in range(n_chunk_steps):
            interaction_start_ns = time.time_ns()
            rollout_infer_start_ns = time.time_ns()
            with record_function("rollout/infer_predict"):
                actions_np, result = self.model.predict(
                    env_obs, mode=mode, compute_values=compute_values,
                )
            rollout_infer_end_ns = time.time_ns()

            _assert_action_chunks_match(
                actions_np, self.num_envs_per_gpu, self.action_dim,
                self.num_action_chunks,
            )
            with record_function("rollout/prepare_actions"):
                chunk_actions = prepare_actions(
                    actions_np,
                    env_type=self.env_type,
                    model_type=self.cfg.actor.model.get("model_type"),
                    num_action_chunks=self.num_action_chunks,
                    action_dim=self.action_dim,
                    policy=self.policy_setup,
                    device=self.device,
                    wm_env_type=self._wm_env_type,
                )

            with record_function("rollout/send_actions"):
                send_pipe.send_dict({"chunk_actions": chunk_actions})

            with record_function("rollout/recv_step"):
                step_results = recv_pipe.recv_dict()
            step_rewards = step_results["rewards"]
            step_terms = step_results["terminations"]
            step_truncs = step_results["truncations"]
            step_dones = torch.logical_or(step_terms, step_truncs)

            new_obs = {
                k: v for k, v in step_results.items()
                if k not in ("rewards", "dones", "terminations", "truncations")
            }
            env_obs = self._fill_obs(new_obs, task_descriptions)

            buffer.append(
                ChunkStepResult(
                    prev_logprobs=result["prev_logprobs"],
                    prev_values=result.get("prev_values"),
                    dones=step_dones,
                    terminations=step_terms,
                    truncations=step_truncs,
                    rewards=step_rewards,
                    versions=self._make_versions_tensor(),
                    forward_inputs=result.get("forward_inputs"),
                )
            )
            interaction_end_ns = time.time_ns()
            interactions_timeline.append({
                "interaction_idx": chunk_idx,
                "interaction_time": {
                    "start_wall_ns": interaction_start_ns,
                    "end_wall_ns": interaction_end_ns,
                },
                "rollout_infer": {
                    "start_wall_ns": rollout_infer_start_ns,
                    "end_wall_ns": rollout_infer_end_ns,
                },
            })

        stats_recv = recv_pipe.recv_dict()
        local_success = int(stats_recv["rollout_local_success_count"][0].item())

        bootstrap_infer_start_ns = time.time_ns()
        _, final_result = self.model.predict(
            env_obs, mode=mode, compute_values=compute_values,
        )
        bootstrap_infer_end_ns = time.time_ns()
        if final_result.get("prev_values") is not None:
            buffer.prev_values.append(final_result["prev_values"])
        if final_result.get("forward_inputs") is not None:
            buffer.forward_inputs.append(final_result["forward_inputs"])

        batch = buffer.to_dict(device="cpu")
        stats = {"local_success_count": local_success}
        env_rollout_timeline = {
            "interactions": interactions_timeline,
            "bootstrap_infer": {
                "start_wall_ns": bootstrap_infer_start_ns,
                "end_wall_ns": bootstrap_infer_end_ns,
            },
        }
        if return_timeline:
            return batch, stats, env_rollout_timeline
        return batch, stats
