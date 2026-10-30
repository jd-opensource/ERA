"""Actor worker for training on dedicated GPUs."""
from __future__ import annotations
import gc
import logging
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.profiler import record_function
from joyrlinf.algorithms.embodied_advantages import (
    compute_embodied_grpo_advantages,
    compute_loss_mask,
    preprocess_embodied_advantages_inputs,
)
from joyrlinf.algorithms.embodied_batch_layout import (
    flatten_for_training,
    layout_from_batch,
)
from joyrlinf.algorithms.embodied_utils import process_received_rollout_batch
from joyrlinf.algorithms.grpo import calculate_episode_scores
from joyrlinf.algorithms.loss import policy_loss
from joyrlinf.algorithms.ppo import compute_embodied_gae_advantages, masked_whiten
from joyrlinf.utils.logging import format_metrics_compact
from joyrlinf.core.launcher import parse_placement
from joyrlinf.core.actor_batch import resolve_actor_batch
from joyrlinf.core.gpu_phase_gate import NullGpuPhaseGate
from joyrlinf.core.gpu_memory_guard import take_memory_snapshot
from joyrlinf.utils.logging_setup import log_at_rank
from joyrlinf.models.model_factory import build_vla_model
from joyrlinf.training.checkpoint import load_actor_checkpoint, save_actor_checkpoint
from joyrlinf.training.fsdp_trainer import (
    setup_fsdp, setup_optimizer, setup_scheduler,
    diagnose_grad_norm,
    offload_params_and_grads, onload_params_and_grads,
    offload_optimizer, onload_optimizer,
    _refresh_orig_param_views,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FlatSlice:
    """Location of an original parameter inside an FSDP FlatParameter."""
    flat_name: str
    offset: int
    numel: int
    shape: tuple
    dtype: torch.dtype


def _normalize_sd_key(name: str) -> str:
    """Normalize wrapper segments while preserving real nested ``model`` names."""
    wrappers = {"module", "_fsdp_wrapped_module", "_orig_mod"}
    parts = [part for part in name.split(".") if part not in wrappers]
    while parts and parts[0] == "model":
        parts.pop(0)
    return ".".join(parts)


def _trainable_key_map(trainable_keys: set[str]) -> dict[str, str]:
    """Map normalized FSDP names back to rollout-compatible parameter names."""
    result: dict[str, str] = {}
    for key in trainable_keys:
        normalized = _normalize_sd_key(key)
        previous = result.get(normalized)
        if previous is not None and previous != key:
            raise RuntimeError(
                "Ambiguous normalized trainable parameter key: "
                f"{normalized!r} maps to both {previous!r} and {key!r}"
            )
        result[normalized] = key
    return result


def _move_to(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {k: _move_to(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_move_to(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_move_to(v, device) for v in data)
    return data


def _gather_microbatch(
    flat: dict,
    perm: torch.Tensor,
    start: int,
    end: int,
) -> dict:
    """Slice a shuffled micro-batch without materializing a full permutation.

    ``flat[k][perm]`` would copy every sample (catastrophic for multi‑GiB
    ``prefix_kv`` living in SharedHost shm). Indexing only ``perm[start:end]``
    copies ``(end - start)`` rows — typically ``micro_batch_size``.
    """
    mb_idx = perm[start:end]
    total = int(perm.shape[0])
    mb: dict = {}
    for key, value in flat.items():
        if (
            isinstance(value, torch.Tensor)
            and value.dim() >= 1
            and value.shape[0] == total
        ):
            mb[key] = value[mb_idx]
        else:
            mb[key] = value
    return mb


def _tensor_bytes(data):
    if isinstance(data, torch.Tensor):
        return data.numel() * data.element_size()
    if isinstance(data, dict):
        return sum(_tensor_bytes(v) for v in data.values())
    if isinstance(data, (list, tuple)):
        return sum(_tensor_bytes(v) for v in data)
    return 0


def _log_actor_vram(label: str, device) -> None:
    """Log this Actor process's VRAM snapshot (MPS: non_pt includes peer)."""
    if isinstance(device, torch.device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
    else:
        idx = int(device)
    snap = take_memory_snapshot(idx)
    logger.info("[%s] %s", label, snap.format_share())


class ActorWorker:
    def __init__(
        self, rank, cfg, device, actor_group, defer_build: bool = False,
        gpu_phase_gate=None,
    ):
        self.rank = rank
        self.cfg = cfg
        self.device = device
        self.actor_group = actor_group
        self.gpu_phase_gate = gpu_phase_gate or NullGpuPhaseGate()
        self.actor_ws = dist.get_world_size(actor_group)
        self.model_wrapper = None
        self.fsdp_model = None
        self.optimizer = None
        self.scheduler = None
        self._pre_fsdp_trainable_param_names: set[str] = set()
        self._flat_mapping = None
        self._built = False
        self._parked = False
        self._skip_cpu_park = False
        self.batch = None
        self.grad_clip = cfg.actor.optim.get("clip_grad", 1.0)
        batch_cfg = resolve_actor_batch(cfg.actor, self.actor_ws)
        self.mbs = batch_cfg.micro_batch_size
        self.ga_steps = batch_cfg.gradient_accumulation_steps
        self.per_actor_batch = batch_cfg.per_actor_batch
        fsdp_cfg = cfg.actor.get("fsdp_config", {})
        amp_cfg = fsdp_cfg.get("amp_autocast", {})
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            None: None,
        }
        self.train_autocast_dtype = None
        if bool(amp_cfg.get("enabled", False)):
            self.train_autocast_dtype = dtype_map.get(
                str(amp_cfg.get("precision", "bf16"))
            )
        self.enable_offload = cfg.actor.get("enable_offload", False)
        # Optimizer-only CPU offload (model stays on GPU). Preferred for
        # colocated async MPS where weights must remain resident.
        self.enable_optimizer_offload = bool(
            cfg.actor.get("enable_optimizer_offload", False)
        )
        # Hand idle cached VRAM (freed microbatch activations) back to the
        # driver when yielding to the paired rollout's predict. The actual
        # empty_cache runs on a background thread so the grant opens
        # immediately and the release overlaps the predict instead of
        # stalling it. Cheap with expandable_segments.
        mem_cfg = cfg.get("memory", None) or {}
        self._release_before_yield = (
            bool(mem_cfg.get("actor_release_before_yield", False))
            and getattr(device, "type", None) == "cuda"
        )
        self._release_event = threading.Event()
        self._release_thread: threading.Thread | None = None
        if self._release_before_yield:
            self.gpu_phase_gate.set_pre_yield_hook(self._signal_idle_vram_release)
        self.is_weight_offloaded = False
        self.is_optimizer_offloaded = False
        # Async PPO staleness: completed-update count of this actor's policy
        # (set by the runner each step; compared against batch versions).
        self.current_version: int = 0
        placement = parse_placement(self.cfg.cluster)
        self._actor_leader = placement.actor_ranks[0]
        log_at_rank(
            logger, self.rank, self._actor_leader, logging.DEBUG,
            "ActorWorker init: rank=%d device=%s actor_world_size=%d mbs=%d ga=%d "
            "per_actor=%d defer_build=%s",
            self.rank, self.device, self.actor_ws,
            self.mbs, self.ga_steps, self.per_actor_batch,
            defer_build,
        )
        if not defer_build:
            self.build_actor()

    def _ensure_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                "ActorWorker not built; call build_actor() first."
            )

    def build_actor(self) -> dict[str, float]:
        """Create VLA model + FSDP + optimizer (once per sync run)."""
        if self._built:
            return {}
        timings: dict[str, float] = {}
        t0 = time.time()
        model_type = self.cfg.actor.model.get("model_type", "openpi")
        log_at_rank(
            logger, self.rank, self._actor_leader, logging.DEBUG,
            "ActorWorker building VLA model (type=%s)", model_type,
        )
        self.model_wrapper = build_vla_model(
            self.cfg.actor.model, self.device, mode="training")
        self._pre_fsdp_trainable_param_names = {
            n for n, p in self.model_wrapper.model.named_parameters()
            if p.requires_grad
        }
        openpi_cfg = self.cfg.actor.model.get("openpi", {}) or {}
        gr00t_cfg = self.cfg.actor.model.get("gr00t", {}) or {}
        expert_only_actor = bool(
            openpi_cfg.get("expert_only_actor", False)
            or gr00t_cfg.get("expert_only_actor", False)
        )
        if expert_only_actor:
            underlying = self.model_wrapper.model
            if not getattr(underlying, "vlm_stripped", False):
                raise RuntimeError(
                    "expert_only_actor=true but actor VLM was not stripped"
                )
            log_at_rank(
                logger, self.rank, self._actor_leader, logging.INFO,
                "ActorWorker expert_only_actor: VLM stripped, "
                "trainable_params=%d",
                len(self._pre_fsdp_trainable_param_names),
            )
        log_at_rank(
            logger, self.rank, self._actor_leader, logging.DEBUG,
            "ActorWorker VLA model ready, wrapping with FSDP",
        )
        self.fsdp_model = setup_fsdp(
            self.model_wrapper.model,
            self.cfg.actor.get("fsdp_config", {}),
            self.actor_group, self.device)
        self.optimizer = setup_optimizer(
            self.fsdp_model, self.cfg.actor.optim)
        self.scheduler = setup_scheduler(
            self.optimizer, self.cfg.actor.optim)
        self._flat_mapping = self._build_flat_param_mapping()
        self._skip_cpu_park = bool(
            self.cfg.actor.get("fsdp_config", {}).get("use_orig_params", False)
        )
        self._built = True
        self._parked = False
        timings["build_actor"] = time.time() - t0
        if self.enable_offload and not self._skip_cpu_park:
            self._offload_all()
        elif self.enable_optimizer_offload:
            # Adam state is empty until the first step(); still mark intent.
            self._offload_optimizer()
        log_at_rank(
            logger, self.rank, self._actor_leader, logging.INFO,
            "ActorWorker build complete (%.2fs, offload=%s, "
            "optimizer_offload=%s, flat_mapping=%s)",
            timings["build_actor"],
            self.enable_offload,
            self.enable_optimizer_offload,
            "ready" if self._flat_mapping else "unavailable",
        )
        return timings

    def wake_actor(self) -> None:
        """Bring a parked actor back to GPU for the next sync training phase."""
        self._ensure_built()
        if not self._parked:
            return
        if self._skip_cpu_park:
            self._parked = False
            logger.info("ActorWorker wake (no-op, model stayed on GPU)")
            return
        t0 = time.time()
        self._onload_model()
        self._onload_optimizer()
        self._parked = False
        torch.cuda.synchronize(self.device)
        logger.info(
            "ActorWorker woke from park (%.2fs)", time.time() - t0,
        )

    def park_actor(self) -> None:
        """Park actor between sync phases; keeps FSDP structure alive."""
        if not self._built:
            return
        self.batch = None
        if self._parked:
            return
        if self._skip_cpu_park:
            self._parked = True
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            logger.info(
                "ActorWorker parked (model stays on GPU, use_orig_params=True, rank=%d)",
                self.rank,
            )
            return
        self._offload_all()
        self._parked = True
        logger.info("ActorWorker parked on CPU (rank=%d)", self.rank)

    # ---- offload / onload helpers ----

    def _offload_all(self):
        """Offload model params, grads, and optimizer to CPU."""
        offload_params_and_grads(self.fsdp_model, offload_grad=True)
        self.is_weight_offloaded = True
        self._offload_optimizer()

    def _onload_model(self):
        """Bring model params back to GPU (needed for weight broadcast & training)."""
        if self.is_weight_offloaded:
            onload_params_and_grads(self.fsdp_model, self.device, onload_grad=False)
            self.is_weight_offloaded = False

    def _offload_model(self):
        if not self.is_weight_offloaded:
            offload_params_and_grads(self.fsdp_model, offload_grad=True)
            self.is_weight_offloaded = True

    def _offload_optimizer(self):
        if not self.is_optimizer_offloaded:
            offload_optimizer(self.optimizer)
            self.is_optimizer_offloaded = True

    def _onload_optimizer(self):
        if self.is_optimizer_offloaded:
            onload_optimizer(self.optimizer, self.device)
            self.is_optimizer_offloaded = False

    def _signal_idle_vram_release(self) -> None:
        """Phase-gate pre-yield hook: wake the background VRAM releaser.

        After a microbatch's backward the activation blocks are free but stay
        reserved by this process; under MPS the paired rollout cannot use
        them for its predict transient peak (attention scores + MLP). The
        empty_cache itself takes O(100ms+) for tens of GiB, so it runs on a
        dedicated thread: the grant reopens immediately and the release
        overlaps the peer's predict, whose peak ramps up over the forward.
        Back-to-back yields coalesce on the event while a release is running.
        """
        if self._release_thread is None:
            self._release_thread = threading.Thread(
                target=self._idle_vram_release_loop,
                name=f"actor-vram-release-{self.rank}",
                daemon=True,
            )
            self._release_thread.start()
        self._release_event.set()

    def _idle_vram_release_loop(self) -> None:
        """Drain release requests; empty_cache only unmaps idle pages.

        With ``expandable_segments:True`` VA ranges stay reserved and the
        next microbatch remaps them; weights/grads/optimizer state and the
        live cuBLAS workspace are untouched.
        """
        while True:
            self._release_event.wait()
            self._release_event.clear()
            start = time.perf_counter()
            try:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            except Exception:
                logger.exception(
                    "[actor_rank%d] background empty_cache failed", self.rank,
                )
                continue
            logger.debug(
                "[actor_rank%d] async empty_cache: idle reserve released "
                "in %.1fms",
                self.rank, (time.perf_counter() - start) * 1e3,
            )

    def _optimizer_step_with_optional_onload(self) -> None:
        """Run ``optimizer.step``; onload Adam state only for this call if configured."""
        if self.enable_optimizer_offload:
            self._onload_optimizer()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        try:
            self.optimizer.step()
        finally:
            if self.enable_optimizer_offload:
                # First step creates Adam state on GPU; always park it after.
                self._offload_optimizer()

    def _align_grad_dtype_for_optimizer(self):
        """Ensure grad dtype matches parameter dtype before optimizer.step().

        In mixed-precision FSDP, some grads may be fp32 while trainable params
        are bf16. AdamW's internal lerp expects matching dtypes.
        """
        converted = 0
        for group in self.optimizer.param_groups:
            for p in group.get("params", []):
                if p is None or p.grad is None:
                    continue
                if p.grad.dtype != p.dtype:
                    p.grad.data = p.grad.data.to(dtype=p.dtype)
                    converted += 1
        if converted > 0:
            logger.info(
                "Aligned grad dtype for optimizer.step(): converted %d tensors",
                converted,
            )

    def _clip_grad_norm(self):
        """Clip grads; return pre-clip norm from FSDP/torch (RLinf contract)."""
        max_norm = self.grad_clip
        if hasattr(self.fsdp_model, "clip_grad_norm_"):
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm)
        else:
            params = [p for p in self.fsdp_model.parameters() if p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        if torch.is_tensor(grad_norm):
            return grad_norm
        return torch.tensor(
            float(grad_norm), device=self.device, dtype=torch.float32,
        )

    def _maybe_log_grad_norm_diagnostics(
        self,
        opt_step: int,
        scaled_loss: float,
    ) -> None:
        debug = os.getenv("JOYRLINF_DEBUG_GRAD_NORM", "0") == "1"
        if not debug or self.rank != 0 or opt_step != 1:
            return
        diag = diagnose_grad_norm(
            self.fsdp_model, self.optimizer, device=self.device,
        )
        diag["scaled_loss"] = float(scaled_loss)
        logger.warning(
            "Grad norm diagnostics (opt_step=1, RLinf contract path): %s",
            diag,
        )

    def _before_micro_batch(self, is_last_micro_batch: bool):
        fsdp_cfg = self.cfg.actor.get("fsdp_config", {})
        if (
            bool(fsdp_cfg.get("enable_gradient_accumulation", True))
            and hasattr(self.fsdp_model, "no_sync")
            and not is_last_micro_batch
        ):
            return self.fsdp_model.no_sync()
        return nullcontext()

    # ---- flat-param mapping (built once, avoids state_dict on every step) ----

    @staticmethod
    def _fsdp_clean_key(name: str) -> str:
        """Remove FSDP / DDP wrapper segments from a dotted param path."""
        return ".".join(
            p for p in name.split(".")
            if p not in ("_fsdp_wrapped_module", "module", "_orig_mod")
        )

    def _build_flat_param_mapping(self):
        """Map original param names to (flat_param, offset) slices.

        Reads FSDP ``FlatParameter`` metadata (``_param_infos``,
        ``_numels``, ``_shapes``) to compute offsets directly — no value
        comparison, no ``state_dict()`` call needed.

        Falls back to ``None`` if metadata is unavailable (non-FSDP model).
        """
        t0 = time.time()

        mod_to_path: dict[int, str] = {
            id(m): p for p, m in self.fsdp_model.named_modules()
        }

        mapping: dict[str, _FlatSlice] = {}
        n_flat = 0

        for fp_name, fp in self.fsdp_model.named_parameters():
            if not hasattr(fp, "_param_infos") or not hasattr(fp, "_numels"):
                continue
            n_flat += 1
            offset = 0
            for i in range(len(fp._param_infos)):
                pinfo = fp._param_infos[i]
                numel = fp._numels[i]
                shape = fp._shapes[i]

                mod_path = mod_to_path.get(id(pinfo.module), "")
                raw = f"{mod_path}.{pinfo.param_name}" if mod_path else pinfo.param_name
                clean = self._fsdp_clean_key(raw)

                mapping[clean] = _FlatSlice(
                    flat_name=fp_name,
                    offset=offset,
                    numel=numel,
                    shape=tuple(shape),
                    dtype=fp.dtype,
                )
                offset += numel

        if not mapping:
            logger.info("Actor rank %d: no FlatParameter metadata, "
                        "flat mapping unavailable", self.rank)
            return None

        trainable = self._pre_fsdp_trainable_param_names
        usable = {k: v for k, v in mapping.items() if k in trainable}

        elapsed = time.time() - t0
        logger.info(
            "Actor rank %d flat_param_mapping: %d from metadata, "
            "%d trainable-matched, %d flat_params (%.1fs)",
            self.rank, len(mapping), len(usable), n_flat, elapsed,
        )

        if not usable:
            logger.warning("Flat mapping: 0 trainable params matched — "
                           "falling back to state_dict() path. "
                           "mapping_sample=%s trainable_sample=%s",
                           list(mapping.keys())[:3],
                           list(trainable)[:3])
            return None

        return usable

    def _get_state_dict_flat_slice(self, trainable_only: bool = True):
        """Extract weights by slicing FlatParameters — no FSDP hooks."""
        needs_offload = self.enable_offload and self.is_weight_offloaded
        if needs_offload:
            self._onload_model()

        trainable_keys = self._pre_fsdp_trainable_param_names
        flat_params = dict(self.fsdp_model.named_parameters())
        sd: dict[str, torch.Tensor] = {}

        for orig_name, slc in self._flat_mapping.items():
            if trainable_only and orig_name not in trainable_keys:
                continue
            fp = flat_params[slc.flat_name]
            chunk = fp.data.view(-1)[slc.offset:slc.offset + slc.numel]
            sd[orig_name] = chunk.view(slc.shape).detach().cpu().clone()

        logger.info(
            "Actor rank %d state_dict_flat_slice: trainable=%d extracted=%d",
            self.rank, len(trainable_keys), len(sd),
        )
        if needs_offload:
            self._offload_model()
        return sd

    # ---- public API ----

    def get_state_dict(self, trainable_only: bool = True):
        if self._flat_mapping is not None:
            return self._get_state_dict_flat_slice(trainable_only)
        if self._skip_cpu_park:
            return self._get_state_dict_direct(trainable_only)
        return self._get_state_dict_fsdp(trainable_only)

    def _get_state_dict_direct(self, trainable_only: bool = True):
        """Extract weights via named_parameters() — avoids FSDP state_dict()
        which invalidates cuBLAS kernel caches and causes 60s+ re-compilation
        on the next forward pass.

        Falls back to the FSDP state_dict() path when FSDP has flattened
        parameters into ``_flat_param`` entries (the default when
        ``use_orig_params=False``), because the flat names cannot be matched
        to the original trainable parameter names that rollout workers expect.
        """
        has_flat = any(
            "_flat_param" in n
            for n, _ in self.fsdp_model.named_parameters()
        )
        if has_flat:
            logger.info(
                "Actor rank %d: FSDP flat params detected, "
                "falling back to state_dict() path",
                self.rank,
            )
            return self._get_state_dict_fsdp(trainable_only)

        needs_offload = self.enable_offload and self.is_weight_offloaded
        if needs_offload:
            self._onload_model()
        torch.cuda.synchronize()
        trainable_keys = self._pre_fsdp_trainable_param_names
        trainable_by_norm = _trainable_key_map(trainable_keys)
        sd = {}
        for name, param in self.fsdp_model.named_parameters():
            if not trainable_only or param.requires_grad:
                nk = _normalize_sd_key(name)
                if not trainable_only:
                    sd[name] = param.detach().cpu().clone()
                    continue
                target = trainable_by_norm.get(nk)
                if target is not None:
                    sd[target] = param.detach().cpu().clone()
        logger.info(
            "Actor rank %d state_dict_direct: trainable_target=%d extracted=%d",
            self.rank, len(trainable_keys) if trainable_only else -1, len(sd),
        )
        if trainable_only and len(sd) != len(trainable_keys):
            logger.warning(
                "Actor rank %d: direct extract matched %d/%d keys; "
                "falling back to state_dict() path",
                self.rank, len(sd), len(trainable_keys),
            )
            if needs_offload:
                self._offload_model()
            return self._get_state_dict_fsdp(trainable_only)
        if needs_offload:
            self._offload_model()
        return sd

    def _get_state_dict_fsdp(self, trainable_only: bool = True):
        """Original FSDP state_dict() path (may invalidate kernel caches)."""
        needs_offload = self.enable_offload and self.is_weight_offloaded
        if needs_offload:
            self._onload_model()
        torch.cuda.synchronize()
        raw_sd = self.fsdp_model.state_dict()
        if trainable_only:
            trainable_keys = set(self._pre_fsdp_trainable_param_names)
            trainable_by_norm = _trainable_key_map(trainable_keys)
            sd = {}
            for k, v in raw_sd.items():
                nk = _normalize_sd_key(k)
                target = trainable_by_norm.get(nk)
                if target is not None:
                    sd[target] = v.detach().cpu()
            logger.info(
                "Actor rank %d state_dict_fsdp filter: raw=%d trainable=%d filtered=%d",
                self.rank, len(raw_sd), len(trainable_keys), len(sd),
            )
            if len(sd) != len(trainable_keys):
                raw_preview = list(raw_sd.keys())[:8]
                train_preview = list(trainable_keys)[:8]
                raise RuntimeError(
                    "Filtered trainable state_dict is incomplete: "
                    f"matched={len(sd)}/{len(trainable_keys)} "
                    f"raw_keys(sample)={raw_preview} "
                    f"trainable_keys(sample)={train_preview}"
                )
        else:
            sd = {k: v.detach().cpu() for k, v in raw_sd.items()}
        if needs_offload:
            self._offload_model()
        return sd

    def receive_rollout_batch(self, batch):
        total_mb = _tensor_bytes(batch) / (1024 ** 2)
        logger.debug(
            "Actor rank %d received rollout batch: %.1f MiB tensor payload",
            self.rank, total_mb,
        )
        batch = _move_to(batch, "cpu")
        # non_blocking D2H copies complete asynchronously; without this sync the
        # CPU tensors after large fields (pixel_values) may still be zero-filled
        # when advantage computation reads them.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_env = self.cfg.env.train
        algo = self.cfg.algorithm
        batch, proc_metrics = process_received_rollout_batch(
            batch,
            auto_reset=bool(train_env.get("auto_reset", True)),
            ignore_terminations=bool(train_env.get("ignore_terminations", False)),
            reward_type=algo.get("reward_type", "action_level"),
            filter_rewards=bool(algo.get("filter_rewards", False)),
            rewards_lower_bound=float(algo.get("rewards_lower_bound", 0.0)),
            rewards_upper_bound=float(algo.get("rewards_upper_bound", 1.0)),
            group_size=int(algo.get("group_size", 8)),
        )
        self.batch = batch
        self._rollout_proc_metrics = proc_metrics
        self._batch_layout = layout_from_batch(
            batch,
            reward_type=algo.get("reward_type", "action_level"),
            logprob_type=algo.get("logprob_type", "chunk_level"),
            action_dim=int(self.cfg.actor.model.get("action_dim", 7)),
        )
        del batch

    def compute_advantages(self):
        adv_type = str(self.cfg.algorithm.get("adv_type", "grpo"))
        if adv_type == "grpo":
            return self._compute_advantages_grpo()
        if adv_type == "gae":
            return self._compute_advantages_gae()
        raise ValueError(
            f"Unsupported algorithm.adv_type={adv_type!r}; expected 'grpo' or 'gae'."
        )

    def _compute_advantages_gae(self):
        """PPO: GAE advantages + returns from rollout critic values."""
        b = self.batch
        algo = self.cfg.algorithm
        r, d = b["rewards"], b["dones"]
        reward_type = str(algo.get("reward_type", "action_level"))
        pv = b.get("prev_values")
        if pv is None:
            raise RuntimeError(
                "adv_type=gae requires prev_values in the rollout batch; "
                "set actor.model.add_value_head: true."
            )
        loss_mask = b.get("loss_mask")
        loss_mask_sum = b.get("loss_mask_sum")
        if loss_mask is None and r.dim() == 3:
            loss_mask, loss_mask_sum = compute_loss_mask(d)
            b["loss_mask"] = loss_mask
            b["loss_mask_sum"] = loss_mask_sum
        if reward_type == "chunk_level" and isinstance(loss_mask, torch.Tensor):
            if loss_mask.dim() == 3:
                loss_mask = loss_mask.any(dim=-1)
            b["loss_mask"] = loss_mask
            if isinstance(loss_mask_sum, torch.Tensor):
                if loss_mask_sum.dim() == 3:
                    loss_mask_sum = loss_mask_sum[..., -1]
                b["loss_mask_sum"] = loss_mask_sum

        adv, ret = compute_embodied_gae_advantages(
            rewards=r,
            dones=d,
            prev_values=pv,
            gamma=float(algo.get("gamma", 0.99)),
            gae_lambda=float(algo.get("gae_lambda", 0.95)),
            loss_mask=loss_mask,
            reward_type=reward_type,
        )
        if bool(algo.get("normalize_advantages", True)):
            adv = masked_whiten(adv, loss_mask)
        b["advantages"] = adv
        b["returns"] = ret

        global_metrics = self._allreduce_adv_metrics(adv.reshape(-1))
        lm_f = loss_mask.float() if loss_mask is not None else torch.ones_like(ret)
        denom = lm_f.sum().clamp(min=1.0)
        global_metrics["returns_mean"] = float((ret * lm_f).sum().item() / denom.item())
        pv_metric = pv[: r.shape[0]].float()
        if adv.dim() == 2 and pv_metric.dim() == 3 and pv_metric.shape[-1] == 1:
            pv_metric = pv_metric.squeeze(-1)
        global_metrics["values_mean"] = float(
            (pv_metric * lm_f).sum().item() / denom.item()
        )

        versions = b.get("versions")
        if isinstance(versions, torch.Tensor):
            stale = (float(self.current_version) - versions.float()).clamp(min=0.0)
            global_metrics["average_version"] = float(versions.float().mean().item())
            global_metrics["current_version"] = float(self.current_version)
            global_metrics["average_staleness"] = float(stale.mean().item())
            global_metrics["data_staleness_0_ratio"] = float((stale == 0).float().mean().item())
            global_metrics["data_staleness_1_ratio"] = float((stale == 1).float().mean().item())

        proc = getattr(self, "_rollout_proc_metrics", None) or {}
        if proc:
            global_metrics.update(proc)
        elif isinstance(b.get("loss_mask"), torch.Tensor):
            global_metrics["loss_mask_frac"] = float(
                b["loss_mask"].float().mean().item()
            )
        if self.rank == 0:
            logger.info(
                "[PPO adv] %s",
                format_metrics_compact(
                    global_metrics,
                    keys=[
                        "adv_mean", "adv_std", "values_mean", "returns_mean",
                        "average_version", "current_version", "average_staleness",
                        "data_staleness_0_ratio", "data_staleness_1_ratio",
                    ],
                ),
            )
        return global_metrics

    def _compute_advantages_grpo(self):
        b = self.batch
        gs = self.cfg.algorithm.get("group_size", 8)
        reward_type = self.cfg.algorithm.get("reward_type", "action_level")
        r, d = b["rewards"], b["dones"]
        loss_mask = b.get("loss_mask")
        loss_mask_sum = b.get("loss_mask_sum")
        if reward_type == "action_level" and r.dim() == 3 and loss_mask is None:
            loss_mask, loss_mask_sum = compute_loss_mask(d)
            b["loss_mask"] = loss_mask
            b["loss_mask_sum"] = loss_mask_sum
        adv = compute_embodied_grpo_advantages(
            rewards=r,
            dones=d,
            group_size=gs,
            reward_type=reward_type,
            loss_mask=loss_mask,
            loss_mask_sum=loss_mask_sum,
        )
        b["advantages"] = adv

        if r.dim() == 3:
            prep = preprocess_embodied_advantages_inputs(
                r, d, reward_type=reward_type, loss_mask=loss_mask,
                loss_mask_sum=loss_mask_sum,
            )
            grouped_scores = calculate_episode_scores(
                prep["rewards"],
                prep["dones"],
                prep["n_steps"],
                prep["batch_size"],
                gs,
            )
        else:
            grouped_scores = calculate_episode_scores(
                r, d, r.shape[0], r.shape[1], gs,
            )

        global_metrics = self._allreduce_adv_metrics(
            adv_tensor=adv,
            loss_mask=b.get("loss_mask"),
        )
        global_metrics.update(
            self._compute_grpo_diagnostics(r, d, grouped_scores, adv.reshape(-1), gs)
        )
        proc = getattr(self, "_rollout_proc_metrics", None) or {}
        if proc:
            global_metrics.update(proc)
        elif isinstance(b.get("loss_mask"), torch.Tensor):
            global_metrics["loss_mask_frac"] = float(
                b["loss_mask"].float().mean().item()
            )
        return global_metrics

    @torch.no_grad()
    def _allreduce_adv_metrics(
        self,
        adv_tensor: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> dict:
        adv = adv_tensor.reshape(-1)

        def _reduce_stats(values: torch.Tensor) -> tuple[float, float]:
            if values.numel() == 0:
                return 0.0, 0.0
            if not dist.is_initialized():
                return float(values.mean().item()), float(values.std().item())
            vals = values.to(torch.float64).to(self.device)
            buf = torch.cat(
                [
                    torch.tensor(
                        [vals.numel()], dtype=torch.float64, device=self.device,
                    ),
                    vals.sum().unsqueeze(0),
                    vals.square().sum().unsqueeze(0),
                ]
            )
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=self.actor_group)
            count, total, total_sq = buf[0], buf[1], buf[2]
            mean = (total / count).item() if count.item() > 0 else 0.0
            var = max((total_sq / count - mean ** 2).item(), 0.0) if count.item() > 0 else 0.0
            return mean, var ** 0.5

        metrics = {}
        mean_all, std_all = _reduce_stats(adv)
        metrics["adv_mean"] = mean_all
        metrics["adv_std"] = std_all

        if loss_mask is not None:
            mask = loss_mask
            while mask.dim() > adv_tensor.dim():
                if mask.shape[-1] == 1:
                    mask = mask.squeeze(-1)
                else:
                    mask = mask.any(dim=-1)
            while mask.dim() < adv_tensor.dim():
                mask = mask.unsqueeze(-1)
            if mask.shape != adv_tensor.shape:
                mask = torch.broadcast_to(mask, adv_tensor.shape)
            valid_adv = adv_tensor[mask.to(dtype=torch.bool, device=adv_tensor.device)]
            mean_masked, std_masked = _reduce_stats(valid_adv.reshape(-1))
            metrics["adv_mean_masked"] = mean_masked
            metrics["adv_std_masked"] = std_masked
        else:
            metrics["adv_mean_masked"] = mean_all
            metrics["adv_std_masked"] = std_all
        return metrics

    @torch.no_grad()
    def _compute_grpo_diagnostics(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        grouped_scores: torch.Tensor,
        adv: torch.Tensor,
        group_size: int,
    ) -> dict:
        if rewards.dim() == 3:
            n_steps, batch_size = rewards.shape[0], rewards.shape[1]
            reward_shape = f"[{n_steps},{batch_size},{rewards.shape[2]}]"
        else:
            n_steps, batch_size = rewards.shape[0], rewards.shape[1]
            reward_shape = f"[{n_steps},{batch_size}]"
        num_groups = batch_size // group_size

        reward_sum = rewards.sum().item()
        reward_nonzero = (rewards.abs() > 1e-8).sum().item()
        reward_total = rewards.numel()

        scores_flat = grouped_scores.reshape(-1)
        score_nonzero = (scores_flat.abs() > 1e-8).sum().item()
        group_stds = grouped_scores.std(dim=-1)
        groups_with_var = (group_stds > 1e-6).sum().item()

        done_count = (
            dones.sum().item()
            if dones.dtype == torch.bool
            else (dones > 0).sum().item()
        )
        local_adv_nonzero = (adv.abs() > 1e-6).sum().item()

        if dist.is_initialized():
            buf = torch.tensor(
                [
                    reward_nonzero, score_nonzero, groups_with_var,
                    reward_sum, done_count, local_adv_nonzero,
                ],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=self.actor_group)
            g_reward_nz = int(buf[0].item())
            g_score_nz = int(buf[1].item())
            g_groups_var = int(buf[2].item())
            g_reward_sum = buf[3].item()
            g_done_count = int(buf[4].item())
            g_adv_nz = int(buf[5].item())
            g_reward_total = reward_total * self.actor_ws
            g_envs = batch_size * self.actor_ws
            g_groups = num_groups * self.actor_ws
        else:
            g_reward_nz = reward_nonzero
            g_score_nz = score_nonzero
            g_groups_var = groups_with_var
            g_reward_sum = reward_sum
            g_done_count = int(done_count)
            g_adv_nz = local_adv_nonzero
            g_reward_total = reward_total
            g_envs = batch_size
            g_groups = num_groups

        logger.debug(
            "[GRPO diag] rewards: shape=%s nonzero=%d/%d (%.1f%%) sum=%.4f | "
            "scores: nonzero=%d/%d | groups_with_var=%d/%d | "
            "dones=%d | adv_nonzero=%d/%d",
            reward_shape, reward_nonzero, reward_total,
            100.0 * reward_nonzero / max(reward_total, 1), reward_sum,
            score_nonzero, batch_size,
            groups_with_var, num_groups,
            int(done_count), local_adv_nonzero, adv.numel(),
        )
        if self.rank == 0:
            logger.debug(
                "[GRPO diag GLOBAL] rewards: nonzero=%d/%d (%.1f%%) sum=%.4f | "
                "scores: nonzero=%d/%d (%.1f%%) | "
                "groups_with_var=%d/%d (%.1f%%) | "
                "dones=%d | adv_nonzero=%d",
                g_reward_nz, g_reward_total,
                100.0 * g_reward_nz / max(g_reward_total, 1), g_reward_sum,
                g_score_nz, g_envs,
                100.0 * g_score_nz / max(g_envs, 1),
                g_groups_var, g_groups,
                100.0 * g_groups_var / max(g_groups, 1),
                g_done_count, g_adv_nz,
            )

        return {
            "reward_nonzero_pct": 100.0 * g_reward_nz / max(g_reward_total, 1),
            "score_nonzero_pct": 100.0 * g_score_nz / max(g_envs, 1),
            "groups_with_var_pct": 100.0 * g_groups_var / max(g_groups, 1),
            "score_max": float(scores_flat.max().item()) if scores_flat.numel() else 0.0,
        }

    def _check_batch_staleness(self, flat: dict) -> dict[str, float]:
        """Staleness metrics from batch versions; hard-drop only on extremes.

        Normal staleness (0/1) is handled inside the loss via proximal
        correction; dropping costs a full rollout, so the hard threshold
        should only catch pathological pipeline stalls. The drop decision is
        all-reduced (MAX) across actor ranks so FSDP stays in lockstep.
        """
        versions = flat.get("versions")
        if not isinstance(versions, torch.Tensor):
            return {}
        stale = (float(self.current_version) - versions.float()).clamp(min=0.0)
        metrics = {
            "average_version": float(versions.float().mean().item()),
            "current_version": float(self.current_version),
            "average_staleness": float(stale.mean().item()),
            "data_staleness_0_ratio": float((stale == 0).float().mean().item()),
            "data_staleness_1_ratio": float((stale == 1).float().mean().item()),
        }
        warn_thr = self.cfg.algorithm.get("staleness_threshold", None)
        if warn_thr is not None and metrics["average_staleness"] > float(warn_thr):
            logger.warning(
                "Actor rank %d: average batch staleness %.2f exceeds "
                "staleness_threshold %.1f (current_version=%d)",
                self.rank, metrics["average_staleness"], float(warn_thr),
                self.current_version,
            )
        hard_thr = self.cfg.algorithm.get("stale_hard_drop_threshold", None)
        if hard_thr is not None and float(hard_thr) > 0:
            max_stale = stale.max().to(self.device)
            if dist.is_initialized():
                dist.all_reduce(
                    max_stale, op=dist.ReduceOp.MAX, group=self.actor_group,
                )
            if float(max_stale.item()) > float(hard_thr):
                logger.error(
                    "Actor rank %d: dropping batch, max staleness %.0f > "
                    "stale_hard_drop_threshold %.0f (pipeline stall?)",
                    self.rank, float(max_stale.item()), float(hard_thr),
                )
                metrics["dropped_stale_batch"] = 1.0
        return metrics

    def run_training(self):
        def _sync_now():
            torch.cuda.synchronize(self.device)

        def _time_block(enabled: bool, acc: dict[str, float], key: str, fn):
            if not enabled:
                return fn()
            _sync_now()
            t0 = time.time()
            out = fn()
            _sync_now()
            acc[key] = acc.get(key, 0.0) + (time.time() - t0)
            return out

        timing_enabled = logger.isEnabledFor(logging.DEBUG)

        t_drain0 = time.time()
        if timing_enabled:
            _sync_now()
        drain_elapsed = time.time() - t_drain0
        if timing_enabled:
            logger.debug(
                "Actor rank %d drain CUDA before training: %.2fs",
                self.rank, drain_elapsed,
            )
        self._prev_model_fwd_total = 0.0
        self._prev_backward_total = 0.0
        # Wall-clock intervals for timeline (fwd / bwd / weight_update only).
        self._train_compute_timeline: list[dict[str, int | str]] = []

        timing = {
            "drain_cuda": drain_elapsed,
            "shuffle": 0.0,
            "move_to": 0.0,
            "model_forward": 0.0,
            "policy_loss": 0.0,
            "backward": 0.0,
            "align_grad": 0.0,
            "clip_grad": 0.0,
            "optimizer_step": 0.0,
            "scheduler_step": 0.0,
        }
        train_t0 = time.time()
        if self.enable_offload:
            self._onload_model()
            self._onload_optimizer()
        # enable_optimizer_offload: keep Adam on CPU through fwd/bwd; onload
        # only inside _optimizer_step_with_optional_onload.
        self.fsdp_model.train()
        self.gpu_phase_gate.enter_training()
        _log_actor_vram(f"actor_rank{self.rank}_pre_compute", self.device)

        try:
            return self._run_training_body(timing, timing_enabled, train_t0)
        finally:
            self.gpu_phase_gate.end_training_granularity_release()
            self.gpu_phase_gate.exit_training()

    def pop_train_compute_timeline(self) -> list[dict[str, int | str]]:
        """Return and clear fwd/bwd/weight_update intervals from the last train."""
        out = list(getattr(self, "_train_compute_timeline", []) or [])
        self._train_compute_timeline = []
        return out

    def _run_training_body(self, timing, timing_enabled, train_t0):
        def _sync_now():
            torch.cuda.synchronize(self.device)

        def _time_block(enabled: bool, acc: dict[str, float], key: str, fn):
            if not enabled:
                return fn()
            _sync_now()
            t0 = time.time()
            out = fn()
            _sync_now()
            acc[key] = acc.get(key, 0.0) + (time.time() - t0)
            return out

        def _record_compute(name: str, fn):
            """Record a timeline interval; does not include gate/recv waits."""
            start_ns = time.time_ns()
            out = fn()
            end_ns = time.time_ns()
            self._train_compute_timeline.append({
                "name": name,
                "start_wall_ns": start_ns,
                "end_wall_ns": end_ns,
            })
            return out

        algo = self.cfg.algorithm
        adv_type = str(algo.get("adv_type", "grpo"))
        loss_type = str(algo.get("loss_type", "actor"))
        compute_values = adv_type == "gae"

        layout = getattr(self, "_batch_layout", None)
        if layout is None:
            layout = layout_from_batch(
                self.batch,
                reward_type=algo.get("reward_type", "action_level"),
                logprob_type=algo.get("logprob_type", "chunk_level"),
                action_dim=int(self.cfg.actor.model.get("action_dim", 7)),
            )
        flat = flatten_for_training(
            self.batch,
            layout,
            compute_loss_mask_fn=compute_loss_mask,
        )
        total = flat["prev_logprobs"].shape[0]

        openpi_cfg = self.cfg.actor.model.get("openpi", {}) or {}
        if bool(openpi_cfg.get("reuse_rollout_prefix_kv", False)):
            if "prefix_kv" not in flat or "prefix_pad_masks" not in flat:
                raise RuntimeError(
                    "reuse_rollout_prefix_kv=true but batch is missing "
                    "prefix_kv / prefix_pad_masks; rollout must pack KV into "
                    "forward_inputs (enable the same flag on the OpenPI model)."
                )
        gr00t_cfg = self.cfg.actor.model.get("gr00t", {}) or {}
        if bool(gr00t_cfg.get("reuse_rollout_backbone_features", False)):
            if "prefix_kv" not in flat:
                raise RuntimeError(
                    "reuse_rollout_backbone_features=true but batch is missing "
                    "cached GR00T features in prefix_kv"
                )

        staleness_metrics = self._check_batch_staleness(flat)
        if staleness_metrics.get("dropped_stale_batch"):
            return staleness_metrics
        # Index-only shuffle: do NOT materialize flat[k] = v[perm].
        # Advanced indexing copies the full tensor; for SharedHost prefix_kv
        # that is ~37 GiB/actor and defeats zero-copy IPC. Gather only the
        # micro-batch rows below (mbs << total).
        perm = _time_block(
            timing_enabled, timing, "shuffle",
            lambda: torch.randperm(total),
        )
        ga = self.ga_steps
        epochs = self.cfg.algorithm.get("update_epoch", 4)
        all_m = {}
        micro_steps = 0
        opt_steps = 0
        micro_per_epoch = (total + self.mbs - 1) // self.mbs
        total_micro = epochs * micro_per_epoch
        collect_hist = bool(getattr(self, "_collect_histograms", False))
        for _ in range(epochs):
            for i in range(0, total, self.mbs):
                micro_steps += 1
                if timing_enabled and micro_steps == 1:
                    t_pre_first = time.time()
                    _sync_now()
                    t_pre_first_sync = time.time() - t_pre_first
                    logger.debug(
                        "Actor rank %d pre-first-micro sync: %.2fs",
                        self.rank, t_pre_first_sync,
                    )
                step_in_ga = (i // self.mbs) % ga
                if step_in_ga == 0:
                    self.optimizer.zero_grad()
                end = min(i + self.mbs, total)
                mb = _gather_microbatch(flat, perm, i, end)
                # Wait until paired predict is idle, then claim the GPU for this MB
                # before H2D (avoids predict activations + train inputs peaking).
                self.gpu_phase_gate.before_microbatch()
                try:
                    mb = _time_block(
                        timing_enabled, timing, "move_to",
                        lambda: _move_to(mb, self.device),
                    )
                    if self.train_autocast_dtype is not None:
                        autocast_ctx = torch.autocast(
                            device_type="cuda", dtype=self.train_autocast_dtype
                        )
                    else:
                        autocast_ctx = torch.autocast(
                            device_type="cuda", enabled=False
                        )
                    def _model_forward():
                        with record_function("joyrlinf/actor/train/forward"):
                            with autocast_ctx:
                                return self.fsdp_model(
                                    forward_type="default_forward",
                                    data=mb,
                                    compute_values=compute_values,
                                )
                    out = _time_block(
                        timing_enabled, timing, "model_forward",
                        lambda: _record_compute("forward", _model_forward),
                    )
                    _log_actor_vram(
                        f"actor_rank{self.rank}_post_forward "
                        f"micro={micro_steps}/{total_micro}",
                        self.device,
                    )
                    use_model_prev_logprobs = bool(
                        self.cfg.algorithm.get("use_model_prev_logprobs", True)
                    )
                    env_override = os.getenv("JOYRLINF_USE_MODEL_PREV_LOGPROBS")
                    if env_override is not None:
                        use_model_prev_logprobs = env_override.lower() in (
                            "1", "true", "yes", "on",
                        )
                    old_logprobs = mb["prev_logprobs"]
                    model_prev_logprobs = out.get("prev_logprobs")
                    old_logprobs_from_model = float(
                        use_model_prev_logprobs and model_prev_logprobs is not None
                    )
                    if old_logprobs_from_model:
                        old_logprobs = model_prev_logprobs

                    extra_kwargs = {}
                    if loss_type != "actor":
                        extra_kwargs["loss_type"] = loss_type
                    # Proximal stale correction (async): pass versions/current_version
                    # for decoupled paths, regardless of whether a critic is present.
                    if loss_type in ("decoupled_actor", "decoupled_actor_critic"):
                        extra_kwargs.update(
                            versions=mb.get("versions"),
                            current_version=self.current_version,
                            behave_weight_threshold=self.cfg.algorithm.get(
                                "behave_weight_threshold", None),
                        )
                    if loss_type in ("actor_critic", "decoupled_actor_critic"):
                        extra_kwargs.update(
                            values=out.get("values"),
                            returns=mb.get("returns"),
                            prev_values=mb.get("prev_values"),
                            value_clip=float(self.cfg.algorithm.get("value_clip", 0.2)),
                            huber_delta=float(self.cfg.algorithm.get("huber_delta", 10.0)),
                        )
                    def _compute_loss():
                        return policy_loss(
                            logprobs=out["logprobs"],
                            old_logprobs=old_logprobs,
                            advantages=mb["advantages"],
                            clip_ratio_high=self.cfg.algorithm.clip_ratio_high,
                            clip_ratio_low=self.cfg.algorithm.clip_ratio_low,
                            clip_ratio_c=self.cfg.algorithm.get("clip_ratio_c", None),
                            clip_log_ratio_min=self.cfg.algorithm.get(
                                "clip_log_ratio_min", None),
                            clip_log_ratio_max=self.cfg.algorithm.get(
                                "clip_log_ratio_max", None),
                            nan_to_num=bool(
                                self.cfg.algorithm.get("nan_to_num", False)),
                            loss_mask=mb.get("loss_mask"),
                            loss_mask_sum=mb.get("loss_mask_sum"),
                            max_episode_steps=self.cfg.env.train.get(
                                "max_episode_steps"),
                            entropy=out.get("entropy"),
                            entropy_bonus=self.cfg.algorithm.get(
                                "entropy_bonus", 0.005),
                            logprob_type=self.cfg.algorithm.get(
                                "logprob_type", "chunk_level"),
                            reward_type=self.cfg.algorithm.get(
                                "reward_type", "action_level"),
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            layout=layout,
                            **extra_kwargs,
                        )
                    loss, met = _time_block(
                        timing_enabled, timing, "policy_loss", _compute_loss
                    )
                    met["old_logprobs_from_model"] = old_logprobs_from_model
                    if collect_hist and micro_steps == total_micro:
                        with torch.no_grad():
                            lp = out["logprobs"].float()
                            old_lp = old_logprobs.float()
                            ratio = torch.exp((lp - old_lp).clamp(-20.0, 20.0)).reshape(-1)
                            adv_s = mb["advantages"].reshape(-1).float()
                            lm = mb.get("loss_mask")
                            if isinstance(lm, torch.Tensor):
                                mask = lm.reshape(-1).bool()
                                if mask.shape == ratio.shape:
                                    ratio = ratio[mask]
                                    adv_s = adv_s[mask]
                            n = min(512, ratio.numel())
                            if n > 0:
                                self._histogram_samples = {
                                    "train/ratio": ratio[:n].detach().cpu().tolist(),
                                    "adv/advantages": adv_s[:n].detach().cpu().tolist(),
                                }
                    _prev_fwd = getattr(self, '_prev_model_fwd_total', 0.0)
                    _prev_bwd = getattr(self, '_prev_backward_total', 0.0)
                    _this_fwd = timing["model_forward"] - _prev_fwd
                    _this_bwd = timing["backward"] - _prev_bwd
                    self._prev_model_fwd_total = timing["model_forward"]
                    self._prev_backward_total = timing["backward"]
                    if timing_enabled and (
                        micro_steps <= 4 or micro_steps % 4 == 0
                    ):
                        logger.debug(
                            "Actor rank %d micro_step %d/%d: "
                            "fwd=%.3fs bwd=%.3fs (this step actual)",
                            self.rank, micro_steps, total_micro,
                            _this_fwd, _this_bwd,
                        )
                    scaled = loss / ga
                    is_last_micro_batch = step_in_ga == ga - 1 or end == total
                    backward_ctx = self._before_micro_batch(
                        is_last_micro_batch=is_last_micro_batch,
                    )
                    with backward_ctx:
                        def _backward():
                            with record_function("joyrlinf/actor/train/backward"):
                                scaled.backward()

                        _time_block(
                            timing_enabled, timing, "backward",
                            lambda: _record_compute("backward", _backward),
                        )
                    if is_last_micro_batch:
                        opt_steps += 1
                        self._maybe_log_grad_norm_diagnostics(
                            opt_steps, float(scaled.detach().item()),
                        )

                        def _weight_update():
                            gn_local = _time_block(
                                timing_enabled, timing, "clip_grad",
                                lambda: self._clip_grad_norm(),
                            )
                            if bool(self.cfg.actor.optim.get(
                                "align_grad_dtype_for_optimizer", False,
                            )):
                                _time_block(
                                    timing_enabled, timing, "align_grad",
                                    lambda: self._align_grad_dtype_for_optimizer(),
                                )
                            with record_function("joyrlinf/actor/train/optimizer"):
                                _time_block(
                                    timing_enabled, timing, "optimizer_step",
                                    lambda: self._optimizer_step_with_optional_onload(),
                                )
                            return gn_local

                        gn = _record_compute("weight_update", _weight_update)
                        met["grad_norm"] = (
                            float(gn.detach().cpu().item())
                            if torch.is_tensor(gn)
                            else float(gn)
                        )
                    for k, v in met.items():
                        all_m.setdefault(k, []).append(v)
                finally:
                    self.gpu_phase_gate.after_microbatch()
        _time_block(
            timing_enabled, timing, "scheduler_step",
            lambda: self.scheduler.step(),
        )
        if self.enable_offload:
            self._offload_all()
        elif self.enable_optimizer_offload:
            self._offload_optimizer()
        total_elapsed = time.time() - train_t0
        if timing_enabled:
            logger.debug(
                "Actor rank %d run_training breakdown: total=%.2fs "
                "micro_steps=%d opt_steps=%d ga=%d epochs=%d | "
                "drain_cuda=%.2fs shuffle=%.2fs move_to=%.2fs "
                "model_fwd=%.2fs policy_loss=%.2fs "
                "backward=%.2fs align_grad=%.2fs clip_grad=%.2fs optimizer_step=%.2fs "
                "scheduler_step=%.2fs",
                self.rank, total_elapsed,
                micro_steps, opt_steps, ga, epochs,
                timing["drain_cuda"],
                timing["shuffle"], timing["move_to"],
                timing["model_forward"], timing["policy_loss"],
                timing["backward"],
                timing["align_grad"], timing["clip_grad"], timing["optimizer_step"],
                timing["scheduler_step"],
            )
        else:
            logger.debug(
                "Actor rank %d run_training: total=%.2fs micro_steps=%d "
                "opt_steps=%d ga=%d epochs=%d "
                "(set JOYRLINF_DEBUG=1 for CUDA-sync breakdown)",
                self.rank, total_elapsed, micro_steps, opt_steps, ga, epochs,
            )
        local_metrics = {k: np.mean(v) for k, v in all_m.items()}
        local_metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        local_metrics.update(staleness_metrics)
        metrics = self._allreduce_train_metrics(local_metrics)
        train_keys = [
            "policy_loss", "policy_loss_abs", "clip_fraction", "grad_norm",
            "ratio_mean", "ratio_max", "ratio_min", "ratio_abs",
            "entropy", "total_loss", "approx_kl", "proximal_approx_kl",
            "advantages_mean", "lr",
            "value_loss", "values_mean", "returns_mean",
            "behav_clip_fraction", "average_staleness",
            "logprobs_finite_frac", "old_logprobs_finite_frac",
            "ratio_finite_frac", "old_logprobs_from_model",
        ]
        if self.rank == 0:
            logger.info(
                "[TRAIN] %s",
                format_metrics_compact(metrics, keys=train_keys),
            )
        return metrics

    def enable_histogram_collection(self, enabled: bool = True) -> None:
        self._collect_histograms = enabled
        if enabled:
            self._histogram_samples = {}

    def pop_histogram_samples(self) -> dict[str, list[float]]:
        return dict(getattr(self, "_histogram_samples", {}) or {})

    @torch.no_grad()
    def _allreduce_train_metrics(self, metrics: dict) -> dict:
        """Average training metrics across actor ranks for global step logs."""
        if not dist.is_initialized():
            return metrics
        keys = [
            "policy_loss", "policy_loss_abs", "clip_fraction", "grad_norm",
            "ratio_mean", "ratio_max", "ratio_min", "ratio_abs",
            "entropy", "total_loss", "advantages_mean", "lr",
            "value_loss", "values_mean", "returns_mean", "approx_kl",
            "proximal_approx_kl", "behav_clip_fraction", "average_staleness",
            "logprobs_finite_frac", "old_logprobs_finite_frac", "ratio_finite_frac",
            "old_logprobs_from_model",
        ]
        vals = torch.tensor(
            [metrics.get(k, 0.0) for k in keys],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=self.actor_group)
        ws = self.actor_ws
        for i, k in enumerate(keys):
            if k in metrics:
                metrics[k] = (vals[i] / ws).item()
        return metrics

    def save_checkpoint(self, save_dir: str, global_step: int) -> None:
        """Save FSDP actor state (rank 0 of actor group writes files)."""
        if self.enable_offload:
            self._onload_model()
            self._onload_optimizer()
        elif self.enable_optimizer_offload:
            self._onload_optimizer()
        torch.cuda.synchronize(self.device)
        save_actor_checkpoint(
            self.fsdp_model,
            self.optimizer,
            self.scheduler,
            save_dir,
            global_step,
            trainable_param_names=self._pre_fsdp_trainable_param_names,
            process_group=self.actor_group,
        )
        self.model_wrapper.set_global_step(global_step)
        if self.enable_optimizer_offload and not self.enable_offload:
            self._offload_optimizer()

    def set_global_step(self, global_step: int) -> None:
        self._ensure_built()
        self.model_wrapper.set_global_step(global_step)

    def load_checkpoint(self, load_dir: str) -> int:
        """Load FSDP actor state; returns global_step from meta.json."""
        self._ensure_built()
        if self.enable_offload or self._parked:
            self._onload_model()
            self._onload_optimizer()
            self._parked = False
        elif self.enable_optimizer_offload:
            self._onload_optimizer()
        global_step = load_actor_checkpoint(
            self.fsdp_model,
            self.optimizer,
            self.scheduler,
            load_dir,
            process_group=self.actor_group,
        )
        self.model_wrapper.set_global_step(global_step)
        if self.enable_optimizer_offload and not self.enable_offload:
            self._offload_optimizer()
        return global_step

    def load_model_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a CPU state dict into the FSDP model (colocated sync phase handoff)."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        self._ensure_built()
        if self.enable_offload or self._parked:
            self._onload_model()
            self._onload_optimizer()
            self._parked = False
        torch.cuda.synchronize(self.device)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(
            self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg,
        ):
            self.fsdp_model.load_state_dict(state_dict, strict=False)
        _refresh_orig_param_views(self.fsdp_model)
        torch.cuda.synchronize(self.device)
        logger.info(
            "Actor rank %d loaded model state dict (%d keys) from staging",
            self.rank, len(state_dict),
        )

    def teardown_actor(
        self, *, release_cuda_cache: bool = False, keep_alive: bool = False,
    ) -> None:
        """Release FSDP model and optimizer (colocated sync phase switch).

        When *keep_alive* is True, park on CPU for reuse on the next phase B
        (avoids ~60s OpenPI+FSDP rebuild; weights/optimizer state persist).
        """
        if keep_alive:
            self.park_actor()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                if release_cuda_cache:
                    torch.cuda.empty_cache()
            logger.info(
                "ActorWorker teardown complete: rank=%d release_cuda_cache=%s "
                "keep_alive=True",
                self.rank, release_cuda_cache,
            )
            return
        self.batch = None
        self.optimizer = None
        self.scheduler = None
        self.fsdp_model = None
        self.model_wrapper = None
        self._flat_mapping = None
        self._built = False
        self._parked = False
        self.is_weight_offloaded = False
        self.is_optimizer_offloaded = False
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            if release_cuda_cache:
                torch.cuda.empty_cache()
        gc.collect()
        logger.info(
            "ActorWorker teardown complete: rank=%d release_cuda_cache=%s",
            self.rank, release_cuda_cache,
        )
