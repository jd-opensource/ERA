"""Embodied RL runner: orchestrates rollout-train loop.

Deployment modes (from ``component_placement``):

Co-located spatial split (env == rollout != actor), e.g. rollout 0-5 + actor 6-7:
  Async only: CombinedEnvRolloutWorker + ActorWorker with NCCL batch/weight comm.

Fully disaggregated (env != rollout):
  Async only: EnvWorker + RolloutOnlyWorker + ActorWorker.

Co-located sync (env == rollout == actor), e.g. ranks 0-7:
  ``pipeline_mode: sync`` — temporal phase switch: all GPUs rollout, then all
  GPUs train; rank-local batch, CPU weight staging, no batch/weight broadcast.

Pipeline modes (``algorithm.pipeline_mode``):
  ``sync``:  colocated full-GPU temporal reuse (strict on-policy).
  ``async``: spatial split or disaggregated (overlap, optional weight staleness).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from joyrlinf.core.config_validation import validate_embodied_training_config
from joyrlinf.core.launcher import (
    Placement, Role, get_role,
    create_process_groups, init_distributed, parse_placement)
from joyrlinf.core.direct_pipe import NCCLPipe, all_to_all_batch, warmup_all2all
from joyrlinf.workers.rollout_worker import CombinedEnvRolloutWorker, RolloutOnlyWorker
from joyrlinf.workers.env_worker import EnvWorker
from joyrlinf.workers.actor_worker import ActorWorker
from joyrlinf.training.checkpoint import load_actor_model_state_dict_cpu
from joyrlinf.observability import MetricsEmitter, PerfSummaryTracker, strip_runner_metrics
from joyrlinf.profiler import ProfilerManager
from joyrlinf.observability.scalar_transport import (
    recv_rollout_scalars,
    send_rollout_scalars,
)
from joyrlinf.utils.logging import format_metrics_compact
from joyrlinf.utils.logging import TrainingLogger
from joyrlinf.utils.logging_setup import GLOBAL_LOG_LEADER, log_at_rank, is_verbose_comm
from joyrlinf.utils.runner_utils import check_progress
from joyrlinf.utils.run_paths import resolve_run_output_dir

logger = logging.getLogger(__name__)

# Long-wait heartbeat: one log line every N seconds while blocked on comms.
HEARTBEAT_LOG_INTERVAL_SEC = 60.0

_ADV_METRIC_KEYS = [
    "adv_mean", "adv_std", "adv_mean_masked", "adv_std_masked",
    "values_mean", "returns_mean",
]
_DATA_METRIC_KEYS = [
    "reward_nonzero_pct", "score_nonzero_pct", "groups_with_var_pct",
]
_TRAIN_METRIC_KEYS = [
    "policy_loss", "policy_loss_abs", "clip_fraction", "grad_norm",
    "ratio_mean", "ratio_max", "ratio_min", "ratio_abs",
    "entropy", "total_loss", "proximal_approx_kl", "approx_kl",
    "advantages_mean", "value_loss", "lr",
    "logprobs_finite_frac", "old_logprobs_finite_frac", "ratio_finite_frac",
    "old_logprobs_from_model",
]


def _flush_log_handlers() -> None:
    for handler in logging.getLogger().handlers:
        flush = getattr(handler, "flush", None)
        if flush is not None:
            try:
                flush()
            except Exception:
                pass


def _parse_global_step_from_resume_dir(resume_dir: str) -> int:
    """Parse ``N`` from ``.../global_step_{N}`` (resume_dir path)."""
    suffix = str(resume_dir).rstrip("/").split("global_step_")[-1]
    try:
        return int(suffix)
    except ValueError as e:
        raise ValueError(
            f"Cannot parse global step from resume_dir={resume_dir!r}; "
            f"expected path ending in global_step_<N>"
        ) from e


def _tensor_payload_bytes(data):
    """Recursively sum tensor payload bytes for logging."""
    if isinstance(data, torch.Tensor):
        return data.numel() * data.element_size()
    if isinstance(data, dict):
        return sum(_tensor_payload_bytes(v) for v in data.values())
    if isinstance(data, (list, tuple)):
        return sum(_tensor_payload_bytes(v) for v in data)
    return 0


def _pin_batch_cpu(batch: dict) -> dict:
    """Move rollout batch tensors to pinned CPU memory for phase handoff."""
    out = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            cpu = val.detach().cpu()
            if cpu.device.type == "cpu" and not cpu.is_pinned():
                cpu = cpu.pin_memory()
            out[key] = cpu
        else:
            out[key] = val
    return out


class EmbodiedRunner:
    """Main training orchestrator."""

    def __init__(self, cfg: DictConfig, gpu_guard=None):
        self.cfg = cfg
        self.gpu_guard = gpu_guard
        self.rank, self.world_size, self.device = init_distributed()
        self.placement = parse_placement(cfg.cluster)
        self.role = get_role(self.rank, self.placement)
        self.comm_mode = cfg.algorithm.get("comm_mode", "v1")
        self.pipeline_mode = cfg.algorithm.get("pipeline_mode", "sync")
        self.is_sync_colocated = (
            self.pipeline_mode == "sync" and self.placement.sync_colocated
        )
        self.groups = create_process_groups(
            self.placement,
            comm_mode=self.comm_mode,
            pipeline_mode=self.pipeline_mode,
        )
        self.is_colocated = self.placement.colocated
        self._leader_rank = self.placement.rollout_ranks[0]

        # commv2: per-node all-to-all group metadata (async spatial split only)
        self._all2all_group = None
        self._a2a_n_rollouts = 0
        self._a2a_n_actors = 0
        self._a2a_my_group_rank = -1
        if not self.is_sync_colocated:
            a2a_node_groups = self.groups.get("_a2a_node_groups", {})
            if self.comm_mode == "v2" and a2a_node_groups:
                from joyrlinf.core.launcher import _get_nproc_per_node
                nproc = _get_nproc_per_node()
                my_node = self.rank // nproc
                info = a2a_node_groups.get(my_node)
                if info is not None:
                    self._all2all_group = info["group"]
                    self._a2a_n_rollouts = info["n_rollouts"]
                    self._a2a_n_actors = info["n_actors"]
                    self._a2a_my_group_rank = info["ranks"].index(self.rank)
        # per-node weight broadcast metadata (async only)
        if self.is_sync_colocated:
            self._wb_group = None
            self._wb_src_actor = self._leader_rank
        else:
            from joyrlinf.core.launcher import _get_nproc_per_node as _nproc
            _nproc_val = _nproc()
            _my_node = self.rank // _nproc_val
            wb_node_groups = self.groups.get("_wb_node_groups", {})
            wb_info = wb_node_groups.get(_my_node)
            if wb_info is not None:
                self._wb_group = wb_info["group"]
                self._wb_src_actor = wb_info["src_actor"]
            else:
                self._wb_group = self.groups.get("weight_broadcast")
                self._wb_src_actor = self.placement.actor_ranks[0]

        self._metrics_rank = (
            self._leader_rank if self.is_sync_colocated else self._wb_src_actor
        )
        # Gloo group for CPU scalar metrics (NCCL default PG cannot send CPU tensors).
        self._metrics_scalar_group = self.groups.get("shutdown")

        log_at_rank(
            logger, self.rank, GLOBAL_LOG_LEADER, logging.INFO,
            "Runner init: role=%s colocated=%s sync_colocated=%s pipeline=%s comm_mode=%s "
            "world=%d device=%s env_ranks=%s rollout_ranks=%s actor_ranks=%s "
            "leader_rank=%d metrics_rank=%d",
            self.role.name, self.is_colocated, self.is_sync_colocated,
            self.pipeline_mode,
            self.comm_mode, self.world_size, self.device,
            self.placement.env_ranks,
            self.placement.rollout_ranks, self.placement.actor_ranks,
            self._leader_rank, self._metrics_rank,
        )

        if str(cfg.runner.get("task_type", "embodied")) == "embodied":
            validate_embodied_training_config(cfg)

        self.worker = None
        self._rollout_worker: CombinedEnvRolloutWorker | None = None
        self._actor_worker: ActorWorker | None = None
        self._weight_staging: dict[str, torch.Tensor] | None = None
        self._cached_inference_model = None
        self._batch_staging: dict | None = None
        self._staging_rollout_scalars: dict[str, float] | None = None
        self._resume_actor_ckpt: str | None = None
        self._sync_resume_loaded = False

        if self.is_sync_colocated:
            log_at_rank(
                logger, self.rank, self._leader_rank, logging.INFO,
                "Sync colocated mode: rollout/actor workers persist across steps "
                "in _run_colocated_sync_loop",
            )
        elif self.role == Role.ENV:
            log_at_rank(
                logger, self.rank, self.placement.env_ranks[0], logging.INFO,
                "Creating EnvWorker (disaggregated)",
            )
            self.worker = EnvWorker(self.rank, cfg, self.device)
        elif self.role == Role.ROLLOUT:
            if self.is_colocated:
                log_at_rank(
                    logger, self.rank, self.placement.rollout_ranks[0], logging.INFO,
                    "Creating CombinedEnvRolloutWorker (co-located)",
                )
                self.worker = CombinedEnvRolloutWorker(
                    self.rank, cfg, self.device, gpu_guard=gpu_guard)
            else:
                log_at_rank(
                    logger, self.rank, self.placement.rollout_ranks[0], logging.INFO,
                    "Creating RolloutOnlyWorker (disaggregated)",
                )
                self.worker = RolloutOnlyWorker(
                    self.rank, cfg, self.device)
        elif self.role == Role.ACTOR:
            log_at_rank(
                logger, self.rank, self.placement.actor_ranks[0], logging.INFO,
                "Creating ActorWorker",
            )
            self.worker = ActorWorker(
                self.rank, cfg, self.device,
                self.groups["actor"])

        self.max_steps = cfg.runner.get("max_steps", 1000)
        self.max_epochs = cfg.runner.get("max_epochs", 10)
        self.val_interval = cfg.runner.get("val_check_interval", 10)
        self.save_interval = cfg.runner.get("save_interval", 50)
        self.global_step = 0
        self.output_dir = resolve_run_output_dir(cfg)
        self._profiler = ProfilerManager.from_cfg(
            cfg,
            rank=self.rank,
            role=self._profiler_role_name(),
            output_dir=self.output_dir,
        )
        self.first_rollout = self.placement.rollout_ranks[0]
        self.metric_logger = None
        self._metrics_emitter: MetricsEmitter | None = None

        log_at_rank(
            logger, self.rank, self._metrics_rank, logging.INFO,
            "Run output directory: %s (resume=%s)",
            self.output_dir,
            bool(cfg.runner.get("resume_dir")),
        )

        if self.rank == self._metrics_rank:
            logger_cfg = cfg.runner.get("logger", {})
            cfg_path = os.path.join(self.output_dir, "config.yaml")
            self.metric_logger = TrainingLogger(
                logger_cfg,
                rank=0,
                output_dir=self.output_dir,
                experiment_cfg=cfg,
                config_path=cfg_path if os.path.isfile(cfg_path) else None,
            )
            perf_tracker = PerfSummaryTracker(
                pipeline_mode=str(self.pipeline_mode),
                comm_mode=str(self.comm_mode),
                output_dir=self.output_dir,
            )
            self._metrics_emitter = MetricsEmitter(
                self.metric_logger,
                perf_tracker,
                gpu_guard=gpu_guard if self.is_sync_colocated else None,
            )
            self._save_experiment_config()

        resume_dir = cfg.runner.get("resume_dir", None)
        if resume_dir:
            actor_ckpt = os.path.join(resume_dir, "actor")
            if not os.path.isdir(actor_ckpt):
                raise FileNotFoundError(
                    f"resume_dir actor checkpoint not found: {actor_ckpt}"
                )
            parsed_step = _parse_global_step_from_resume_dir(resume_dir)
            if self.is_sync_colocated:
                sd = load_actor_model_state_dict_cpu(actor_ckpt)
                self._weight_staging = {
                    k: v.detach().cpu().clone() for k, v in sd.items()
                }
                self._resume_actor_ckpt = actor_ckpt
                self.global_step = parsed_step
            elif self.role == Role.ACTOR:
                loaded_step = self.worker.load_checkpoint(actor_ckpt)
                self.global_step = loaded_step if loaded_step else parsed_step
                if loaded_step and loaded_step != parsed_step:
                    logger.warning(
                        "resume_dir path implies global_step=%d but meta.json has %d; "
                        "using meta.json",
                        parsed_step, loaded_step,
                    )
            else:
                # Rollout/env must share the same step as actor for loop bounds,
                # weight thread iteration counts, eval/save scheduling, and TB step.
                self.global_step = parsed_step
            log_at_rank(
                logger, self.rank, self._metrics_rank, logging.INFO,
                "Resumed %s rank %d from %s (global_step=%d, next_step=%d)",
                "sync" if self.is_sync_colocated else self.role.name,
                self.rank, resume_dir,
                self.global_step, self.global_step + 1,
            )

        total_num_envs = cfg.env.train.total_num_envs
        rollout_epoch = cfg.algorithm.get("rollout_epoch", 1)
        # Total env transitions per env per rollout_epoch() is max_steps_per_rollout_epoch
        # (rollout divides by num_action_chunks only to schedule chunk_step; do not multiply
        # num_action_chunks again here — that would overcount).
        env_steps_per_rollout = cfg.env.train.max_steps_per_rollout_epoch
        self.env_frames_per_step = (
            total_num_envs * rollout_epoch * env_steps_per_rollout
        )
        log_at_rank(
            logger, self.rank, self._metrics_rank, logging.INFO,
            "Env-frames per step (throughput numerator): %d envs × %d rollout_epoch × "
            "%d env_steps/rollout = %d",
            total_num_envs, rollout_epoch, env_steps_per_rollout,
            self.env_frames_per_step,
        )

        if (
            self.role == Role.ROLLOUT
            and self.comm_mode != "v2"
            and not self.is_sync_colocated
        ):
            self._setup_data_pipes()

        if not self.is_colocated and self.role in (Role.ENV, Role.ROLLOUT):
            self._setup_env_rollout_pipe()
            logger.info("Env<->Rollout pipe ready")

    def _profiler_role_name(self) -> str:
        if self.is_sync_colocated:
            return "sync"
        if self.role == Role.ENV:
            return "env"
        if self.role == Role.ROLLOUT:
            return "rollout"
        if self.role == Role.ACTOR:
            return "actor"
        return self.role.name.lower()

    def _step_throughput(self, step_elapsed_s: float) -> tuple[float, int, float]:
        """Per-step env-frames/s based on the current step elapsed time."""
        elapsed = max(float(step_elapsed_s), 1e-6)
        step_frames = int(self.env_frames_per_step)
        return step_frames / elapsed, step_frames, elapsed

    def _rollout_train_success_denominator(self) -> int:
        """Upper bound for summed success_once counts: one per env per rollout_epoch."""
        re = self.cfg.algorithm.get("rollout_epoch", 1)
        return int(re * self.cfg.env.train.total_num_envs)

    def _reduce_rollout_success(self, local_total: int) -> int:
        """Sum ``local_success_count`` across all rollout ranks (training interaction)."""
        t = torch.tensor([local_total], dtype=torch.int64, device=self.device)
        grp = self.groups.get("rollout")
        if grp is not None:
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=grp)
        return int(t.item())

    def _reduce_rollout_env_metric_sums(
        self, local_sums: dict[str, float]
    ) -> dict[str, float]:
        keys = (
            "num_trajectories",
            "success_once_sum",
            "return_sum",
            "episode_len_sum",
            "reward_sum",
        )
        values = [float(local_sums.get(k, 0.0)) for k in keys]
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        grp = self.groups.get("rollout")
        if grp is not None:
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=grp)
        return {k: float(v) for k, v in zip(keys, t.cpu().tolist())}

    @staticmethod
    def _mean_from_sum(total: float, count: float) -> float:
        return float(total) / max(float(count), 1.0)

    def _save_experiment_config(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        cfg_path = os.path.join(self.output_dir, "config.yaml")
        OmegaConf.save(self.cfg, cfg_path)
        logger.info("Saved experiment config to %s", cfg_path)

    def _checkpoint_actor_dir(self, global_step: int) -> str:
        return os.path.join(
            self.output_dir, "checkpoints", f"global_step_{global_step}", "actor",
        )

    def _save_checkpoint(self, global_step: int) -> None:
        if self.rank != self._metrics_rank:
            return
        save_dir = self._checkpoint_actor_dir(global_step)
        logger.info("Saving checkpoint at global_step=%d -> %s", global_step, save_dir)
        actor = self._actor_worker if self.is_sync_colocated else self.worker
        actor.save_checkpoint(save_dir, global_step)
    
    def _save_timeline_jsonl(self, timeline_obj: dict) -> None:
        save_dir = os.path.join(self.output_dir, "timeline")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"timeline_rank{self.rank}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(timeline_obj, separators=(",", ":")) + "\n")

    @staticmethod
    def _attach_metric_tensors(
        batch: dict, metrics: dict[str, float], env_dim: int, device: torch.device,
    ) -> None:
        """Pack scalar metrics as [1, env_dim] tensors for commv2 all2all."""
        for key, val in metrics.items():
            safe = key.replace("/", "__")
            batch[f"_runner_metric_{safe}"] = torch.full(
                (1, env_dim), float(val), dtype=torch.float32, device=device,
            )

    def _strip_runner_metrics(self, batch: dict) -> dict[str, float]:
        return strip_runner_metrics(batch)

    def _attach_rollout_time_metrics(
        self,
        batch: dict,
        timing: dict[str, float],
    ) -> None:
        """Pack rollout-side wall times for async actor metrics merge."""
        if not timing:
            return
        ref = next(
            (
                t for k, t in batch.items()
                if isinstance(t, torch.Tensor) and not k.startswith("_runner_metric_")
            ),
            None,
        )
        if ref is None or ref.dim() < 2:
            return
        env_dim = ref.shape[1]
        metrics = {f"time/{k}": float(v) for k, v in timing.items()}
        self._attach_metric_tensors(batch, metrics, env_dim, ref.device)

    def _histogram_interval(self) -> int:
        if self.metric_logger is None:
            return 0
        return int(getattr(self.metric_logger, "histogram_interval", 0))

    def _prepare_histogram_collection(self, actor: ActorWorker, step: int) -> None:
        interval = self._histogram_interval()
        if interval > 0 and step % interval == 0:
            actor.enable_histogram_collection(True)

    def _pop_actor_histograms(self, actor: ActorWorker) -> dict[str, list[float]]:
        if hasattr(actor, "pop_histogram_samples"):
            return actor.pop_histogram_samples()
        return {}

    def _emit_step_metrics(
        self,
        global_step: int,
        *,
        train_metrics: dict | None = None,
        adv_metrics: dict | None = None,
        stripped_rollout: dict | None = None,
        time_metrics: dict | None = None,
        eval_metrics: dict | None = None,
        histograms: dict[str, list[float]] | None = None,
    ) -> None:
        if self._metrics_emitter is None:
            return
        self._metrics_emitter.emit_step(
            global_step,
            train=train_metrics or {},
            adv=adv_metrics or {},
            stripped_rollout=stripped_rollout or {},
            time_metrics=time_metrics or {},
            eval_metrics=eval_metrics,
            histograms=histograms,
        )

    def _reduce_eval_metrics(self, local: dict[str, float]) -> dict[str, float]:
        """Mean-aggregate eval scalars across rollout ranks."""
        keys = ["eval/success_rate", "eval/mean_return"]
        out = {}
        grp = self.groups.get("rollout")
        for key in keys:
            val = float(local.get(key, 0.0))
            t = torch.tensor([val], dtype=torch.float64, device=self.device)
            if grp is not None:
                dist.all_reduce(t, op=dist.ReduceOp.SUM, group=grp)
                n = dist.get_world_size(grp)
                val = float(t.item()) / n
            out[key] = val
        return out

    def _run_rollout_eval_if_needed(
        self,
        step: int,
        weight_lock: threading.Lock | None = None,
        pending_weights: list | None = None,
    ) -> dict[str, float]:
        """Run held-out eval on rollout ranks when ``val_check_interval`` hits.

        In async mode, never call ``_sync_weights_from_actor()`` here: the
        background ``weight_recv_loop`` is the sole consumer of gloo weight
        broadcasts.  Use the latest ``pending_weights`` snapshot instead.
        """
        run_val, _, _ = check_progress(
            step,
            self.max_steps,
            self.val_interval,
            self.save_interval,
        )
        if not run_val or not self.is_colocated:
            return {}
        if not isinstance(self.worker, CombinedEnvRolloutWorker):
            return {}

        if self.pipeline_mode == "async":
            if weight_lock is not None and pending_weights is not None:
                with weight_lock:
                    if pending_weights[0] is not None:
                        self.worker.load_weights(pending_weights[0])
            else:
                logger.warning(
                    "Async eval at step %d without pending_weights; "
                    "using weights from rollout start (may be stale).",
                    step,
                )
        else:
            if self.pipeline_mode == "sync":
                pass  # weights already loaded in rollout model from staging
            else:
                self._sync_weights_from_actor()

        eval_local = self.worker.rollout_eval(
            use_train_env=self.is_sync_colocated,
        )
        eval_global = self._reduce_eval_metrics(eval_local)
        if self.rank == self.first_rollout:
            eval_msg = (
                f"[EVAL] step {step}/{self.max_steps} (interval={self.val_interval}): "
                + ", ".join(f"{k}={v:.4f}" for k, v in eval_global.items())
            )
            # Print via tqdm.write directly (not logger): the openvla/prismatic
            # model import reconfigures the root logger, so logger records can be
            # swallowed. tqdm.write is hijack-proof and coexists with the bar.
            tqdm.write(eval_msg)
            logger.debug(eval_msg)
        return eval_global

    def _build_rollout_scalar_metrics(
        self,
        global_success: int,
        rollout_epochs: int,
        eval_metrics: dict[str, float] | None = None,
        timing: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Build rollout-side scalars for the dedicated transport channel (P1)."""
        if not isinstance(self.worker, CombinedEnvRolloutWorker):
            return {}
        metrics = CombinedEnvRolloutWorker.make_train_runner_metrics(
            global_success,
            self._rollout_train_success_denominator(),
        )
        if eval_metrics:
            metrics.update(eval_metrics)
        if timing:
            metrics.update({f"time/{k}": float(v) for k, v in timing.items()})
        return metrics

    def _enrich_batch_with_metrics(
        self,
        batch: dict,
        global_success: int,
        rollout_epochs: int,
        step: int,
        eval_metrics: dict[str, float] | None = None,
    ) -> dict:
        """Deprecated: batch tensor piggyback removed in P1; returns batch unchanged."""
        return batch

    def _canonical_time_metrics(self, **kwargs: float) -> dict[str, float]:
        """Benchmark-friendly aliases alongside fine-grained timers."""
        out = dict(kwargs)
        rollout_total = out.get("rollout_total", out.get("phase_a_rollout"))
        if rollout_total is not None:
            out.setdefault("rollout_total", rollout_total)
            out.setdefault("rollout", rollout_total)
        actor_train = out.get("actor_train", out.get("phase_b_train", out.get("train")))
        if actor_train is not None:
            out.setdefault("actor_train", actor_train)
            out.setdefault("train", actor_train)
        tp = out.get("throughput")
        if tp is not None:
            out.setdefault("throughput_env_s", tp)
        return out

    def _emit_step_perf(
        self,
        msg: str,
        *,
        pbar: tqdm | None = None,
        level: int = logging.INFO,
    ) -> None:
        """Print step timing and metrics above tqdm; also write to log file."""
        logger.log(level, msg)
        if pbar is not None and not pbar.disable:
            pbar.write(f"[rank {self.rank}] {msg}", file=sys.stdout)
        _flush_log_handlers()

    def _format_step_metrics(
        self,
        adv_metrics: dict,
        train_metrics: dict,
        data_metrics: dict | None = None,
    ) -> str:
        from joyrlinf.observability.metrics import split_adv_data_metrics

        adv_part, data_part = split_adv_data_metrics(adv_metrics)
        if data_metrics:
            data_part = {**data_part, **data_metrics}
        adv_s = format_metrics_compact(adv_part, keys=_ADV_METRIC_KEYS)
        train_s = format_metrics_compact(train_metrics, keys=_TRAIN_METRIC_KEYS)
        data_s = format_metrics_compact(data_part, keys=_DATA_METRIC_KEYS)
        parts = [f"adv: {adv_s}", f"train: {train_s}"]
        if data_s != "(empty)":
            parts.append(f"data: {data_s}")
        return " | ".join(parts)

    def _start_wait_heartbeat(
        self, phase: str, interval_s: float = HEARTBEAT_LOG_INTERVAL_SEC,
    ):
        """Emit one INFO heartbeat every *interval_s* while blocking (default 60s)."""
        stop_evt = threading.Event()
        start_t = time.time()

        def _beat():
            while not stop_evt.wait(interval_s):
                logger.info(
                    "Heartbeat: phase=%s role=%s rank=%d waiting=%.1fs",
                    phase, self.role.name, self.rank, time.time() - start_t
                )

        th = threading.Thread(
            target=_beat, daemon=True, name=f"hb-{self.rank}-{phase[:24]}"
        )
        th.start()
        return stop_evt, th

    def _barrier_with_log(self, name: str, group: dist.ProcessGroup | None = None):
        """Barrier with begin/end logging for hang diagnosis."""
        t0 = time.time()
        backend = dist.get_backend(group) if group is not None else dist.get_backend()
        comm_level = logging.INFO if is_verbose_comm() else logging.DEBUG
        logger.log(
            comm_level,
            "Barrier wait start: name=%s role=%s rank=%d group=%s backend=%s",
            name, self.role.name, self.rank,
            "default" if group is None else name,
            backend,
        )
        hb_evt, hb_th = self._start_wait_heartbeat(f"barrier:{name}")
        barrier_kwargs = {}
        if backend == "nccl":
            barrier_kwargs["device_ids"] = [int(torch.cuda.current_device())]
        if group is None:
            try:
                dist.barrier(**barrier_kwargs)
            finally:
                hb_evt.set()
                hb_th.join(timeout=0.2)
        else:
            try:
                dist.barrier(group=group, **barrier_kwargs)
            finally:
                hb_evt.set()
                hb_th.join(timeout=0.2)
        elapsed = time.time() - t0
        done_level = logging.WARNING if elapsed >= 1.0 else comm_level
        logger.log(
            done_level,
            "Barrier wait done: name=%s role=%s rank=%d elapsed=%.2fs",
            name, self.role.name, self.rank, elapsed,
        )

    # ------------------------------------------------------------------ #
    #  Pipe setup
    # ------------------------------------------------------------------ #

    def _setup_data_pipes(self):
        """Setup NCCLPipes for rollout->actor batch transfer.

        One pipe per paired actor. With M actors and N rollouts (M >= N),
        each rollout may feed multiple actors.
        """
        local_rank = int(torch.cuda.current_device())
        self.data_pipes = []
        for a_rank in self.placement.get_all_paired_actors(self.rank):
            pair_key = f"pair_{self.rank}_{a_rank}"
            group = self.groups.get(pair_key)
            pipe = NCCLPipe(
                src_rank=self.rank, dst_rank=a_rank,
                local_rank=local_rank, group=group)
            self.data_pipes.append(pipe)
        logger.debug(
            "Rollout rank %d: %d data pipes to actors %s",
            self.rank, len(self.data_pipes),
            self.placement.get_all_paired_actors(self.rank),
        )

    def _setup_env_rollout_pipe(self):
        """Setup NCCLPipe for env<->rollout communication (disaggregated mode)."""
        local_rank = int(torch.cuda.current_device())
        if self.role == Role.ENV:
            r_rank = self.placement.get_paired_rollout_for_env(self.rank)
            pair_key = f"env_rollout_pair_{self.rank}_{r_rank}"
            group = self.groups.get(pair_key)
            self.env_rollout_send_pipe = NCCLPipe(
                src_rank=self.rank, dst_rank=r_rank,
                local_rank=local_rank, group=group)
            self.env_rollout_recv_pipe = NCCLPipe(
                src_rank=r_rank, dst_rank=self.rank,
                local_rank=local_rank, group=group)
            self.paired_rollout_rank = r_rank
            self.env_rollout_group = group
        elif self.role == Role.ROLLOUT:
            e_rank = self.placement.get_paired_env_for_rollout(self.rank)
            pair_key = f"env_rollout_pair_{e_rank}_{self.rank}"
            group = self.groups.get(pair_key)
            self.env_rollout_recv_pipe = NCCLPipe(
                src_rank=e_rank, dst_rank=self.rank,
                local_rank=local_rank, group=group)
            self.env_rollout_send_pipe = NCCLPipe(
                src_rank=self.rank, dst_rank=e_rank,
                local_rank=local_rank, group=group)
            self.paired_env_rank = e_rank
            self.env_rollout_group = group

    # ------------------------------------------------------------------ #
    #  Dispatch
    # ------------------------------------------------------------------ #

    def run(self):
        """Main training loop dispatched by role and pipeline mode."""
        shutdown_group = self.groups.get("shutdown")
        self._barrier_with_log("shutdown_pre_warmup", shutdown_group)
        self._warmup_nccl_pairs()
        self._barrier_with_log("shutdown_post_warmup", shutdown_group)
        log_at_rank(
            logger, self.rank, self._leader_rank, logging.INFO,
            "NCCL pair communicators warmed up, entering main loop",
        )

        if self.is_sync_colocated:
            self._run_colocated_sync_loop()
        elif self.role == Role.ENV:
            self._run_env_loop_async()
        elif self.role == Role.ROLLOUT:
            self._run_rollout_loop_async()
        elif self.role == Role.ACTOR:
            self._run_actor_loop_async()

        logger.info("Loop finished for role=%s, waiting at exit barrier (gloo)", self.role.name)
        self._profiler.finalize()
        if self._metrics_emitter is not None:
            self._metrics_emitter.close()
        if self.metric_logger is not None:
            self.metric_logger.close()
        shutdown_group = self.groups.get("shutdown")
        if shutdown_group is not None:
            self._barrier_with_log("shutdown_exit", shutdown_group)
        else:
            self._barrier_with_log("shutdown_exit_default")
        logger.info("Exit barrier passed, safe to tear down")

    def _warmup_nccl_pairs(self):
        """Pre-warm NCCL communicators for all pair groups.

        NCCL communicators are created lazily on first use. Both members of a
        pair group must participate in the initialization handshake.  Without
        this warmup, async mode can deadlock because one side triggers init
        before the other side has reached its first NCCL call on that group.
        """
        comm_level = logging.INFO if is_verbose_comm() else logging.DEBUG
        dummy = {"_w": torch.zeros(1, device=self.device)}
        logger.log(
            comm_level,
            "Comm warmup start: role=%s colocated=%s sync_colocated=%s pipeline=%s",
            self.role.name, self.is_colocated, self.is_sync_colocated, self.pipeline_mode,
        )

        if self.is_sync_colocated:
            actor_group = self.groups.get("actor")
            if actor_group is not None:
                self._barrier_with_log("warmup_actor_group", actor_group)
            log_at_rank(
                logger, self.rank, self._leader_rank, comm_level,
                "Comm warmup done (sync colocated): role=%s", self.role.name,
            )
            return

        if self.comm_mode == "v2" and self._all2all_group is not None:
            if self.role in (Role.ROLLOUT, Role.ACTOR):
                logger.log(
                    comm_level,
                    "Warmup all2all group: rank=%d group_rank=%d",
                    self.rank, self._a2a_my_group_rank,
                )
                warmup_all2all(
                    self._a2a_n_rollouts, self._a2a_n_actors,
                    self._a2a_my_group_rank, self._all2all_group, self.device,
                )
            if self.role == Role.ACTOR:
                actor_group = self.groups.get("actor")
                if actor_group is not None:
                    self._barrier_with_log("warmup_actor_group", actor_group)
        elif self.role == Role.ROLLOUT:
            logger.log(
                comm_level,
                "Warmup rollout->actor NCCL pairs: rollout_rank=%d actors=%s",
                self.rank, self.placement.get_all_paired_actors(self.rank),
            )
            for pipe in self.data_pipes:
                logger.log(
                    comm_level,
                    "Warmup send start: pair rollout=%d -> actor=%d",
                    pipe.src_rank, pipe.dst_rank,
                )
                pipe.send_dict(dummy)
                logger.log(
                    comm_level,
                    "Warmup send done: pair rollout=%d -> actor=%d",
                    pipe.src_rank, pipe.dst_rank,
                )
        elif self.role == Role.ACTOR:
            actor_group = self.groups.get("actor")
            if actor_group is not None:
                logger.log(
                    comm_level,
                    "Warmup actor_group NCCL collective start: actor_rank=%d",
                    self.rank,
                )
                self._barrier_with_log("warmup_actor_group", actor_group)
                logger.log(
                    comm_level,
                    "Warmup actor_group NCCL collective done: actor_rank=%d",
                    self.rank,
                )
            r_rank = self.placement.get_paired_rollout(self.rank)
            local_rank = int(torch.cuda.current_device())
            pair_key = f"pair_{r_rank}_{self.rank}"
            group = self.groups.get(pair_key)
            if group is not None:
                logger.log(
                    comm_level,
                    "Warmup recv start: pair rollout=%d -> actor=%d",
                    r_rank, self.rank,
                )
                pipe = NCCLPipe(src_rank=r_rank, dst_rank=self.rank,
                                local_rank=local_rank, group=group)
                pipe.recv_dict()
                logger.log(
                    comm_level,
                    "Warmup recv done: pair rollout=%d -> actor=%d",
                    r_rank, self.rank,
                )

        if not self.is_colocated and self.role in (Role.ENV, Role.ROLLOUT):
            if self.role == Role.ENV:
                logger.log(
                    comm_level,
                    "Warmup env<->rollout start: env=%d rollout=%d",
                    self.rank, self.paired_rollout_rank,
                )
                self.env_rollout_send_pipe.send_dict(dummy)
                self.env_rollout_recv_pipe.recv_dict()
                logger.log(
                    comm_level,
                    "Warmup env<->rollout done: env=%d rollout=%d",
                    self.rank, self.paired_rollout_rank,
                )
            elif self.role == Role.ROLLOUT:
                logger.log(
                    comm_level,
                    "Warmup env<->rollout start: env=%d rollout=%d",
                    self.paired_env_rank, self.rank,
                )
                self.env_rollout_recv_pipe.recv_dict()
                self.env_rollout_send_pipe.send_dict(dummy)
                logger.log(
                    comm_level,
                    "Warmup env<->rollout done: env=%d rollout=%d",
                    self.paired_env_rank, self.rank,
                )
        log_at_rank(
            logger, self.rank, self._leader_rank, comm_level,
            "Comm warmup done: role=%s", self.role.name,
        )

    # ================================================================== #
    #  Colocated sync loop (env == rollout == actor, temporal phases)
    # ================================================================== #

    def _prepare_initial_sync_weight_staging(self, actor_group) -> None:
        """Stage actor weights before the first sync rollout.

        RLinf's patch weight syncer performs an initial actor -> rollout sync
        before step 1.  The colocated CPU-staging path must do the same;
        otherwise step 1 rolls out with a freshly built inference model instead
        of the actor/FSDP state used for training.
        """
        if self._weight_staging is not None:
            return

        t0 = time.time()
        if self.rank == self._leader_rank:
            logger.warning("[SYNC] Preparing initial rollout weights from actor state")
        if self._actor_worker is None:
            self._actor_worker = ActorWorker(
                self.rank, self.cfg, self.device, actor_group,
                defer_build=True,
            )
        build_timings = self._actor_worker.build_actor()
        if self._resume_actor_ckpt and not self._sync_resume_loaded:
            loaded = self._actor_worker.load_checkpoint(self._resume_actor_ckpt)
            self._sync_resume_loaded = True
            logger.info(
                "Sync resume: loaded full actor checkpoint from %s (step=%d)",
                self._resume_actor_ckpt, loaded,
            )

        t_export0 = time.time()
        self._weight_staging = self._actor_worker.get_state_dict()
        t_export = time.time() - t_export0
        self._actor_worker.teardown_actor(
            release_cuda_cache=True, keep_alive=True,
        )
        self.worker = None
        if self.rank == self._leader_rank:
            build_detail = " ".join(
                f"{k}={v:.2f}s" for k, v in sorted(build_timings.items())
            )
            logger.warning(
                "[SYNC] Initial rollout weights staged: %s export_w=%.2fs total=%.2fs",
                build_detail or "actor_reused",
                t_export,
                time.time() - t0,
            )

    def _run_colocated_sync_loop(self):
        """All ranks: Phase A rollout -> Phase B actor each global step."""
        shutdown_group = self.groups.get("shutdown")
        actor_group = self.groups.get("actor")
        rollout_epochs = self.cfg.algorithm.get("rollout_epoch", 1)
        pbar = tqdm(
            total=self.max_steps,
            initial=self.global_step,
            desc="Sync",
            disable=(self.rank != self._leader_rank),
            dynamic_ncols=True,
        )
        log_at_rank(
            logger, self.rank, self._leader_rank, logging.INFO,
            "[SYNC] Entering colocated sync loop: max_steps=%d start_step=%d world=%d",
            self.max_steps, self.global_step + 1, self.world_size,
        )
        self._rollout_worker = CombinedEnvRolloutWorker(
            self.rank, self.cfg, self.device,
            gpu_guard=self.gpu_guard, defer_build=True,
        )
        self._prepare_initial_sync_weight_staging(actor_group)

        for step in range(self.global_step + 1, self.max_steps + 1):
            step_start_ns = time.time_ns()
            t0 = time.time()
            self._profiler.on_step_begin(step)
            if self.rank == self._leader_rank:
                logger.info("[SYNC] Step %d/%d start (phase A: rollout)", step, self.max_steps)

            # ---- Phase A: Rollout ----
            t_a0 = time.time()
            with self._profiler.range("phase_a_rollout", step):
                build_timings = self._rollout_worker.build_rollout(
                    weight_staging=self._weight_staging,
                    cached_model=self._cached_inference_model,
                )
                self._cached_inference_model = None
                # Sync mode: rollout always uses last step's weights (staleness 0).
                self._rollout_worker.weight_version = step - 1
                self._rollout_worker.set_global_step(step - 1)
                self.worker = self._rollout_worker
                t_build = time.time() - t_a0

                all_batches = []
                total_local_success = 0
                local_env_metric_sums = {
                    "num_trajectories": 0.0,
                    "success_once_sum": 0.0,
                    "return_sum": 0.0,
                    "episode_len_sum": 0.0,
                    "reward_sum": 0.0,
                }
                t_rollout_core0 = time.time()

                rollout_epochs_timeline = []

                for rollout_epoch_idx in range(rollout_epochs):
                    # call rollout_epoch() and record the time taken for each epoch
                    rollout_epoch_start_ns = time.time_ns()
                    batch, succ_st, env_rollout_timeline = (
                        self._rollout_worker.rollout_epoch(
                            mode="train", return_timeline=True,
                        )
                    )
                    rollout_epoch_end_ns = time.time_ns()
                    rollout_epochs_timeline.append({
                        "rollout_epoch_idx": rollout_epoch_idx,
                        "epoch_timeline" : {
                            "start_wall_ns": rollout_epoch_start_ns,
                            "end_wall_ns": rollout_epoch_end_ns,
                        },
                        "env_reset": env_rollout_timeline["env_reset"],
                        "interactions": env_rollout_timeline["interactions"],
                        "bootstrap_infer": env_rollout_timeline["bootstrap_infer"],
                    })

                    total_local_success += succ_st["local_success_count"]
                    for key in local_env_metric_sums:
                        local_env_metric_sums[key] += float(succ_st.get(key, 0.0))
                    all_batches.append(batch)
                merged = self._merge_batches(all_batches)
                global_success = self._reduce_rollout_success(total_local_success)
                env_metric_sums = self._reduce_rollout_env_metric_sums(
                    local_env_metric_sums
                )
                env_traj = env_metric_sums["num_trajectories"]
                env_success = self._mean_from_sum(
                    env_metric_sums["success_once_sum"], env_traj
                )
                t_rollout_core = time.time() - t_rollout_core0

                t_eval0 = time.time()
                eval_metrics = self._run_rollout_eval_if_needed(step)
                t_eval = time.time() - t_eval0

                self._staging_rollout_scalars = self._build_rollout_scalar_metrics(
                    global_success, rollout_epochs, eval_metrics=eval_metrics or None,
                )
                self._staging_rollout_scalars.update({
                    "env/success_once": env_success,
                    "env/num_trajectories": env_traj,
                    "env/mean_return": self._mean_from_sum(
                        env_metric_sums["return_sum"], env_traj),
                    "env/mean_episode_len": self._mean_from_sum(
                        env_metric_sums["episode_len_sum"], env_traj),
                    "env/mean_reward": self._mean_from_sum(
                        env_metric_sums["reward_sum"], env_traj),
                })
                t_rollout = time.time() - t_a0

                t_stash0 = time.time()
                self._batch_staging = _pin_batch_cpu(merged)
                del merged
                t_stash = time.time() - t_stash0

                t_teardown_a0 = time.time()
                self._cached_inference_model = self._rollout_worker.teardown_rollout(
                    release_cuda_cache=True, keep_model=True, keep_env=True,
                )
                self.worker = None
                t_teardown_a = time.time() - t_teardown_a0

            if self.rank == self._leader_rank:
                build_detail = " ".join(
                    f"{k}={v:.2f}s" for k, v in sorted(build_timings.items())
                )
                succ_denom = self._rollout_train_success_denominator()
                succ_pct = 100.0 * global_success / max(succ_denom, 1)
                env_traj = env_metric_sums["num_trajectories"]
                env_success = self._mean_from_sum(
                    env_metric_sums["success_once_sum"], env_traj
                )
                logger.info(
                    "[SYNC] Rollout step %d/%d: build=%.2fs rollout=%.2fs eval=%.2fs "
                    "total=%.2fs | train/success_once=%d/%d (%.2f%%) | "
                    "env/success_once=%.6f env/num_trajectories=%.0f",
                    step, self.max_steps, t_build, t_rollout_core, t_eval, t_rollout,
                    global_success, succ_denom, succ_pct, env_success, env_traj,
                )
                logger.info(
                    "[SYNC] Step %d/%d phase A done: build=%.2fs (%s) "
                    "rollout_core=%.2fs total_rollout=%.2fs stash=%.2fs teardown=%.2fs",
                    step, self.max_steps, t_build, build_detail or "n/a",
                    t_rollout_core, t_rollout, t_stash, t_teardown_a,
                )

            self._barrier_with_log(f"sync_phase_a_step_{step}", shutdown_group)

            # ---- Phase B: Actor ----
            actor_phase_start_ns = time.time_ns()
            t_b0 = time.time()
            with self._profiler.range("phase_b_train", step):
                actor_build_timings: dict[str, float] = {}
                if self._actor_worker is None:
                    self._actor_worker = ActorWorker(
                        self.rank, self.cfg, self.device, actor_group,
                        defer_build=True,
                    )
                    actor_build_timings = self._actor_worker.build_actor()
                    if self._resume_actor_ckpt and not self._sync_resume_loaded:
                        t_load0 = time.time()
                        loaded = self._actor_worker.load_checkpoint(self._resume_actor_ckpt)
                        actor_build_timings["load_checkpoint"] = time.time() - t_load0
                        self._sync_resume_loaded = True
                        logger.info(
                            "Sync resume: loaded full actor checkpoint from %s (step=%d)",
                            self._resume_actor_ckpt, loaded,
                        )
                    elif self._weight_staging is not None:
                        t_load0 = time.time()
                        self._actor_worker.load_model_state_dict(self._weight_staging)
                        actor_build_timings["load_weights"] = time.time() - t_load0
                else:
                    t_wake0 = time.time()
                    self._actor_worker.wake_actor()
                    actor_build_timings["wake_actor"] = time.time() - t_wake0
                t_build_b = time.time() - t_b0
                self._actor_worker.current_version = step - 1
                self._actor_worker.set_global_step(step - 1)
                self.worker = self._actor_worker
                rollout_metrics = dict(self._staging_rollout_scalars or {})
                self._staging_rollout_scalars = None
                rollout_metrics.update(self._strip_runner_metrics(self._batch_staging))
                self._actor_worker.receive_rollout_batch(self._batch_staging)
                self._batch_staging = None

                t_adv0 = time.time()
                adv_metrics = self._actor_worker.compute_advantages()
                t_adv = time.time() - t_adv0
                if actor_group is not None:
                    self._barrier_with_log(f"sync_actor_group_step_{step}", actor_group)
                t_train0 = time.time()
                self._prepare_histogram_collection(self._actor_worker, step)
                actor_train_start_ns = time.time_ns()
                train_metrics = self._actor_worker.run_training()
                actor_train_end_ns = time.time_ns()
                histograms = self._pop_actor_histograms(self._actor_worker)
                t_train = time.time() - t_train0

                _, save_model, _ = check_progress(
                    step, self.max_steps, self.val_interval, self.save_interval,
                )
                if save_model:
                    self._save_checkpoint(step)

                t_export0 = time.time()
                self._weight_staging = self._actor_worker.get_state_dict()
                t_export = time.time() - t_export0

                t_teardown_b0 = time.time()
                self._actor_worker.teardown_actor(
                    release_cuda_cache=True, keep_alive=True,
                )
                self.worker = None
                t_teardown_b = time.time() - t_teardown_b0
            t_phase_b = time.time() - t_b0
            actor_phase_end_ns = time.time_ns()

            self._barrier_with_log(f"sync_phase_b_step_{step}", shutdown_group)
            step_end_ns = time.time_ns()
            self._save_timeline_jsonl({
                "step": step,
                "rank": self.rank,
                "step_timeline": {
                    "start_wall_ns": step_start_ns,
                    "end_wall_ns": step_end_ns,
                },
                "env_rollout_timeline": rollout_epochs_timeline,
                "actor_phase_timeline": {
                    "start_wall_ns": actor_phase_start_ns,
                    "end_wall_ns": actor_phase_end_ns,
                },
                "actor_train_timeline": {
                    "start_wall_ns": actor_train_start_ns,
                    "end_wall_ns": actor_train_end_ns,
                },
            })
            t_total = time.time() - t0

            if self.rank == self._metrics_rank:
                actor_build_detail = " ".join(
                    f"{k}={v:.2f}s" for k, v in sorted(actor_build_timings.items())
                )
                logger.info(
                    "[SYNC] Step %d/%d phase B prep: build=%.2fs (%s)",
                    step, self.max_steps, t_build_b, actor_build_detail or "n/a",
                )
                tput, step_f, step_elapsed = self._step_throughput(t_total)
                succ_denom = self._rollout_train_success_denominator()
                succ_pct = 100.0 * global_success / max(succ_denom, 1)
                self._emit_step_metrics(
                    step,
                    train_metrics=train_metrics,
                    adv_metrics=adv_metrics,
                    stripped_rollout=rollout_metrics,
                    histograms=histograms or None,
                    time_metrics=self._canonical_time_metrics(
                        phase_a_build=t_build,
                        phase_a_rollout_core=t_rollout_core,
                        phase_a_rollout=t_rollout,
                        phase_a_eval=t_eval,
                        phase_a_stash=t_stash,
                        phase_a_teardown=t_teardown_a,
                        rollout_total=t_rollout,
                        rollout_core=t_rollout_core,
                        phase_b_build=t_build_b,
                        phase_b_advantages=t_adv,
                        phase_b_train=t_train,
                        actor_train=t_train,
                        actor_adv=t_adv,
                        phase_b_export=t_export,
                        phase_b_teardown=t_teardown_b,
                        phase_b_total=t_phase_b,
                        total=t_total,
                        throughput=tput,
                    ),
                )
                metrics_line = self._format_step_metrics(adv_metrics, train_metrics)
                self._emit_step_perf(
                    f"[SYNC] Step {step}/{self.max_steps}: build_a={t_build:.2f}s "
                    f"rollout={t_rollout:.2f}s (core={t_rollout_core:.2f}s) "
                    f"stash={t_stash:.2f}s teardown_a={t_teardown_a:.2f}s "
                    f"build_b={t_build_b:.2f}s advantages={t_adv:.2f}s "
                    f"train={t_train:.2f}s export_w={t_export:.2f}s "
                    f"teardown_b={t_teardown_b:.2f}s total={t_total:.2f}s | "
                    f"throughput={tput:.0f} env-frames/s "
                    f"(step {step_f} / {step_elapsed:.2f}s) | "
                    f"train/success_once={global_success}/{succ_denom} "
                    f"({succ_pct:.2f}%) | env/success_once={env_success:.6f} "
                    f"env/num_trajectories={env_traj:.0f} | staleness=0 | "
                    f"{metrics_line}",
                    pbar=pbar,
                )
                pbar.set_description(f"Sync {step}/{self.max_steps}")
                pbar.update(1)
                pbar.set_postfix_str(
                    f"{tput:.0f} env-f/s total={t_total:.0f}s", refresh=True,
                )

            self._profiler.on_step_end(step)

        pbar.close()
        if self._rollout_worker is not None:
            self._rollout_worker.teardown_rollout(release_cuda_cache=True)
            self._rollout_worker = None
            self._cached_inference_model = None
        if self._actor_worker is not None:
            self._actor_worker.teardown_actor(release_cuda_cache=True)
            self._actor_worker = None
        log_at_rank(
            logger, self.rank, self._leader_rank, logging.INFO,
            "[SYNC] Colocated sync loop done",
        )

    # ================================================================== #
    #  ASYNC loops
    # ================================================================== #
    #
    # Architecture overview (threading model):
    #
    #   Rollout rank                        Actor rank (src only)
    #   ----------                          ----------------------
    #   main thread:                        main thread:
    #     rollout -> send_batch               recv_batch -> train
    #   weight-recv thread:                 weight-bcast thread:
    #     recv broadcast (weight_broadcast    get_state_dict -> broadcast
    #     group)                              (weight_broadcast group)
    #
    # The two thread pairs use *different* process groups, so they
    # can run concurrently without conflict.  The weight thread on actor
    # extracts the state-dict AFTER training completes (guarded by events)
    # to prevent races on model parameters.
    #
    # Overlap: while actor trains on step N, rollout generates step N+1
    # data using (potentially stale) weights from step N-1.
    # Steady-state step time = max(rollout+send, recv+train).
    # ================================================================== #

    def _run_env_loop_async(self):
        """Async env loop: no global barrier, driven by rollout NCCL P2P."""
        log_at_rank(
            logger, self.rank, self._leader_rank, logging.INFO,
            "[ASYNC] Entering env loop: max_steps=%d", self.max_steps,
        )
        rollout_epochs = self.cfg.algorithm.get("rollout_epoch", 1)
        for step in range(self.global_step + 1, self.max_steps + 1):
            step_start_ns = time.time_ns()
            t0 = time.time()
            rollout_epochs_timeline = []
            self._profiler.on_step_begin(step)
            with self._profiler.range("env", step):
                for rollout_epoch_idx in range(rollout_epochs):
                    rollout_epoch_start_ns = time.time_ns()
                    env_rollout_timeline = self.worker.run_epoch(
                        send_pipe=self.env_rollout_send_pipe,
                        recv_pipe=self.env_rollout_recv_pipe,
                        env_rollout_group=self.env_rollout_group,
                        return_timeline=True,
                    )
                    rollout_epoch_end_ns = time.time_ns()
                    rollout_epochs_timeline.append({
                        "rollout_epoch_idx": rollout_epoch_idx,
                        "epoch_timeline": {
                            "start_wall_ns": rollout_epoch_start_ns,
                            "end_wall_ns": rollout_epoch_end_ns,
                        },
                        "env_reset": env_rollout_timeline["env_reset"],
                        "interactions": env_rollout_timeline["interactions"],
                        "bootstrap_infer": env_rollout_timeline["bootstrap_infer"],
                    })
            step_end_ns = time.time_ns()
            self._save_timeline_jsonl({
                "step": step,
                "rank": self.rank,
                "role": "env",
                "step_timeline": {
                    "start_wall_ns": step_start_ns,
                    "end_wall_ns": step_end_ns,
                },
                "env_rollout_timeline": rollout_epochs_timeline,
            })
            t_total = time.time() - t0
            tput, step_f, step_elapsed = self._step_throughput(t_total)
            logger.info(
                "[ASYNC] Env step %d/%d: total=%.1fs | "
                "throughput=%.0f env-frames/s (step %d / %.2fs)",
                step, self.max_steps, t_total,
                tput, step_f, step_elapsed,
            )
            self._profiler.on_step_end(step)

    def _run_rollout_loop_async(self):
        """Async rollout: overlaps rollout with actor training.

        A background thread receives weight broadcasts from the actor.
        The main thread loads the latest available weights (non-blocking)
        before each rollout, tolerating 1-step staleness.

        Note: No tqdm here — sync mode advances the bar after ``dist.barrier``
        (i.e. after actor trains). Async would mislead if we ticked after send:
        rollout can finish before actor's ``run_training`` completes. The main
        step progress bar lives in ``_run_actor_loop_async`` instead.
        """
        first_rollout = self.placement.rollout_ranks[0]
        rollout_epochs = self.cfg.algorithm.get("rollout_epoch", 1)
        log_at_rank(
            logger, self.rank, first_rollout, logging.INFO,
            "[ASYNC] Entering rollout loop: max_steps=%d", self.max_steps,
        )

        # ---- shared state for weight updates ----
        weight_lock = threading.Lock()
        pending_weights = [None]
        weight_ver = [0]

        def weight_recv_loop():
            logger.debug("weight_recv_loop thread started")
            # Initial weights are loaded synchronously.  Only steps that have a
            # subsequent rollout need a post-train update; receiving the final
            # actor update leaves this daemon thread racing process-group
            # teardown when the actor tail is longer than the join timeout.
            n_updates = max(self.max_steps - self.global_step - 1, 0)
            for i in range(n_updates):
                logger.debug("weight_recv_loop iter %d: calling _recv_weights_raw()", i)
                w = self._recv_weights_raw()
                logger.debug("weight_recv_loop iter %d: received weights, %d keys", i, len(w))
                with weight_lock:
                    pending_weights[0] = w
                    weight_ver[0] += 1
            logger.debug("weight_recv_loop thread exiting")

        # Synchronous initial weight load (step 0)
        logger.debug("initial _sync_weights_from_actor() starting")
        self._sync_weights_from_actor()
        logger.debug("initial _sync_weights_from_actor() done")
        loaded_ver = 0
        # Init to the resumed global step so the displayed staleness starts at
        # 0 (initial weights were just loaded synchronously above), instead of
        # showing a bogus value equal to the resumed step count.
        last_weight_apply_step = self.global_step
        # Weight-version metadata for async staleness (GRPO now uses it too).
        base_version = self.global_step
        self.worker.weight_version = base_version

        # Start background weight receiver (uses weight_broadcast group)
        wt = threading.Thread(target=weight_recv_loop, daemon=True,
                              name="weight-recv")
        wt.start()
        logger.debug("weight_recv_loop thread launched")
        for step in range(self.global_step + 1, self.max_steps + 1):
            step_start_ns = time.time_ns()
            t0 = time.time()
            self._profiler.on_step_begin(step)
            logger.debug("rollout_async step %d starting", step)

            with self._profiler.range("rollout", step):
                # Non-blocking weight update
                t_ws = time.time()
                with weight_lock:
                    if pending_weights[0] is not None:
                        logger.debug("loading pending weights (ver=%d)", weight_ver[0])
                        self.worker.load_weights(pending_weights[0])
                        loaded_ver = weight_ver[0]
                        self.worker.weight_version = base_version + loaded_ver
                        pending_weights[0] = None
                        last_weight_apply_step = step
                    else:
                        logger.debug("no pending weights available")
                staleness = step - last_weight_apply_step
                t_sync = time.time() - t_ws

                # Rollout
                t1 = time.time()
                logger.debug("step %d rollout_epoch starting", step)
                all_batches = []
                total_local_success = 0
                rollout_epochs_timeline = []
                for rollout_epoch_idx in range(rollout_epochs):
                    rollout_epoch_start_ns = time.time_ns()
                    if self.is_colocated:
                        batch, succ_st, env_rollout_timeline = (
                            self.worker.rollout_epoch(
                                mode="train", return_timeline=True,
                            )
                        )
                    else:
                        batch, succ_st, env_rollout_timeline = self.worker.rollout_epoch(
                            recv_pipe=self.env_rollout_recv_pipe,
                            send_pipe=self.env_rollout_send_pipe,
                            env_rollout_group=self.env_rollout_group,
                            env_rank=self.paired_env_rank,
                            mode="train",
                            return_timeline=True,
                        )
                    rollout_epoch_end_ns = time.time_ns()
                    rollout_epochs_timeline.append({
                        "rollout_epoch_idx": rollout_epoch_idx,
                        "epoch_timeline": {
                            "start_wall_ns": rollout_epoch_start_ns,
                            "end_wall_ns": rollout_epoch_end_ns,
                        },
                        # Reset runs on EnvWorker and is recorded in that rank's
                        # timeline, not returned by RolloutOnlyWorker.
                        "env_reset": env_rollout_timeline.get("env_reset", {}),
                        "interactions": env_rollout_timeline["interactions"],
                        "bootstrap_infer": env_rollout_timeline["bootstrap_infer"],
                    })
                    total_local_success += succ_st["local_success_count"]
                    all_batches.append(batch)
                merged = self._merge_batches(all_batches)
                global_success = self._reduce_rollout_success(total_local_success)
                t_rollout = time.time() - t1

                t_eval0 = time.time()
                eval_metrics = self._run_rollout_eval_if_needed(
                    step, weight_lock=weight_lock, pending_weights=pending_weights,
                )
                t_eval = time.time() - t_eval0

                rollout_scalars = self._build_rollout_scalar_metrics(
                    global_success, rollout_epochs, eval_metrics=eval_metrics or None,
                    timing={
                        "rollout_weight_load": t_sync,
                        "rollout_core": t_rollout,
                        "rollout_eval": t_eval,
                        "rollout_total": t_sync + t_rollout + t_eval,
                    },
                )
                logger.debug("step %d rollout_epoch done (%.2fs)", step, t_rollout)

                t2 = time.time()
                if self.comm_mode == "v2":
                    logger.debug("step %d _send_batch_v2 starting", step)
                    self._send_batch_v2(merged)
                else:
                    logger.debug("step %d _send_batch_to_actor starting", step)
                    self._send_batch_to_actor(merged)
                t_send = time.time() - t2
                rollout_scalars["time/rollout_send"] = float(t_send)
                if self.rank == first_rollout:
                    send_rollout_scalars(
                        self._metrics_rank,
                        rollout_scalars,
                        group=self._metrics_scalar_group,
                    )
                logger.debug("step %d send done (%.2fs)", step, t_send)
                del merged
            step_end_ns = time.time_ns()
            self._save_timeline_jsonl({
                "step": step,
                "rank": self.rank,
                "role": "rollout",
                "step_timeline": {
                    "start_wall_ns": step_start_ns,
                    "end_wall_ns": step_end_ns,
                },
                "env_rollout_timeline": rollout_epochs_timeline,
            })
            t_total = time.time() - t0
            if self.rank == first_rollout:
                tput, step_f, step_elapsed = self._step_throughput(t_total)
                succ_denom = self._rollout_train_success_denominator()
                succ_pct = 100.0 * global_success / max(succ_denom, 1)
                rollout_msg = (
                    f"[ASYNC] Rollout step {step}/{self.max_steps}: weight_load={t_sync:.2f}s "
                    f"rollout={t_rollout:.2f}s send={t_send:.2f}s eval={t_eval:.2f}s "
                    f"total={t_total:.2f}s | staleness={staleness} | "
                    f"throughput={tput:.0f} env-frames/s (step {step_f} / {step_elapsed:.2f}s) | "
                    f"train/success_once={global_success}/{succ_denom} ({succ_pct:.2f}%)"
                )
                # tqdm.write is hijack-proof (see eval note): the model import
                # can reconfigure the root logger, so print directly here and
                # keep the logger record at debug for the file only.
                tqdm.write(rollout_msg)
                logger.debug(rollout_msg)
            self._profiler.on_step_end(step)

        wt.join(timeout=120)
        logger.info("Rollout async loop done")

    def _run_actor_loop_async(self):
        """Async actor: overlaps weight broadcast with next batch receive.

        Only the source actor (rank 0 in actor group) runs a background
        thread for weight extraction + broadcast.  All actor ranks sync
        via actor_group barrier before FSDP training to ensure collective
        all-reduce in backward does not timeout.

        Key ordering: recv_batch is done BEFORE waiting for weight
        extraction so that rollout's NCCL send is consumed promptly,
        preventing cascading timeouts.
        """
        is_src = self.rank == self._wb_src_actor
        actor_group = self.groups.get("actor")
        pbar = tqdm(
            total=self.max_steps,
            initial=self.global_step,
            desc=f"Actor[async] rank {self.rank}",
            disable=(self.rank != self.placement.actor_ranks[0]),
        )

        # ---- threading primitives (src actor only) ----
        training_done_evt = threading.Event()
        weights_extracted_evt = threading.Event()
        weights_extracted_evt.set()  # no wait needed before first training

        def weight_broadcast_loop():
            logger.debug("weight_broadcast_loop thread started")
            # The final trained weights have no consumer: all rollout epochs
            # are already complete.  Skipping that broadcast also guarantees
            # the background Gloo thread exits before process-group teardown.
            n_updates = max(self.max_steps - self.global_step - 1, 0)
            for i in range(n_updates):
                logger.debug("weight_bcast iter %d: waiting training_done_evt", i)
                training_done_evt.wait()
                training_done_evt.clear()
                logger.debug("weight_bcast iter %d: extracting state_dict", i)
                torch.cuda.synchronize()
                sd = self.worker.get_state_dict()
                logger.debug("weight_bcast iter %d: state_dict extracted (%d keys)", i, len(sd))
                weights_extracted_evt.set()
                logger.debug("weight_bcast iter %d: broadcasting weights (gloo/CPU)", i)
                self._broadcast_weights_from_sd(sd)
                logger.debug("weight_bcast iter %d: broadcast done", i)
            logger.debug("weight_broadcast_loop thread exiting")

        # Synchronous initial broadcast (all ranks participate)
        logger.debug("actor_async is_src=%s, initial broadcast starting", is_src)
        if is_src:
            self._broadcast_weights()
        logger.debug("initial broadcast phase done")

        wbt = None
        if is_src:
            wbt = threading.Thread(target=weight_broadcast_loop, daemon=True,
                                   name="weight-bcast")
            wbt.start()
            logger.debug("weight_broadcast_loop thread launched")

        for step in range(self.global_step + 1, self.max_steps + 1):
            step_start_ns = time.time_ns()
            t0 = time.time()
            self._profiler.on_step_begin(step)
            logger.debug("actor_async step %d starting", step)
            # Policy version = completed updates (async PPO staleness metadata).
            self.worker.current_version = step - 1

            with self._profiler.range("actor_train", step):
                # 1. Receive batch FIRST so rollout's send is consumed promptly.
                t1 = time.time()
                if self.comm_mode == "v2":
                    logger.debug("step %d _recv_batch_v2 starting", step)
                    batch = self._recv_batch_v2()
                else:
                    logger.debug("step %d _recv_batch_from_rollout starting", step)
                    batch = self._recv_batch_from_rollout()
                t_recv = time.time() - t1
                logger.debug("step %d recv done (%.2fs, %d keys)", step, t_recv, len(batch))
                rollout_metrics: dict[str, float] = {}
                if is_src:
                    rollout_metrics = recv_rollout_scalars(
                        self.first_rollout,
                        group=self._metrics_scalar_group,
                    )
                rollout_metrics.update(self._strip_runner_metrics(batch))
                self.worker.receive_rollout_batch(batch)
                del batch

                # 2. Wait for weight extraction to finish (prevents model access
                #    race with the BG thread that reads state_dict).
                t_we = time.time()
                if is_src:
                    logger.debug("step %d waiting weights_extracted_evt", step)
                    weights_extracted_evt.wait()
                    weights_extracted_evt.clear()
                    logger.debug("step %d weights_extracted_evt received", step)
                t_wait_extract = time.time() - t_we

                # 3. Compute advantages (no FSDP dependency)
                t2 = time.time()
                adv_metrics = self.worker.compute_advantages()
                t_adv = time.time() - t2

                # 4. Sync all actor ranks before FSDP training (backward all-reduce)
                logger.debug("step %d actor_group barrier", step)
                if actor_group is not None:
                    self._barrier_with_log(f"async_actor_group_step_{step}", actor_group)
                logger.debug("step %d actor_group barrier done", step)

                # 5. Train
                t3 = time.time()
                logger.debug("step %d training starting", step)
                self._prepare_histogram_collection(self.worker, step)
                actor_train_start_ns = time.time_ns()
                train_metrics = self.worker.run_training()
                actor_train_end_ns = time.time_ns()
                histograms = self._pop_actor_histograms(self.worker)
                t_train = time.time() - t3
                logger.debug("step %d training done (%.2fs)", step, t_train)

            t_total = time.time() - t0
            _, save_model, _ = check_progress(
                step, self.max_steps, self.val_interval, self.save_interval,
            )
            # Save before weight thread reads state_dict; no post-save barrier so
            # rank-7 can enter the next NCCL all2all recv promptly.
            if save_model and is_src:
                self._save_checkpoint(step)

            # Signal weight thread: training complete, safe to extract state_dict
            if is_src and step < self.max_steps:
                torch.cuda.synchronize()
                training_done_evt.set()
                logger.debug("step %d training_done_evt set", step)
            step_end_ns = time.time_ns()
            self._save_timeline_jsonl({
                "step": step,
                "rank": self.rank,
                "role": "actor",
                "step_timeline": {
                    "start_wall_ns": step_start_ns,
                    "end_wall_ns": step_end_ns,
                },
                "actor_phase_timeline": {
                    "start_wall_ns": step_start_ns,
                    "end_wall_ns": step_end_ns,
                },
                "actor_train_timeline": {
                    "start_wall_ns": actor_train_start_ns,
                    "end_wall_ns": actor_train_end_ns,
                },
            })
            if is_src:
                tput, step_f, step_elapsed = self._step_throughput(t_total)
                self._emit_step_metrics(
                    step,
                    train_metrics=train_metrics,
                    adv_metrics=adv_metrics,
                    stripped_rollout=rollout_metrics,
                    histograms=histograms or None,
                    time_metrics=self._canonical_time_metrics(
                        recv=t_recv,
                        wait_extract=t_wait_extract,
                        actor_adv=t_adv,
                        actor_train=t_train,
                        total=t_total,
                        throughput=tput,
                    ),
                )
                metrics_line = self._format_step_metrics(adv_metrics, train_metrics)
                actor_msg = (
                    f"[ASYNC] Actor step {step}/{self.max_steps}: recv={t_recv:.2f}s "
                    f"wait_extract={t_wait_extract:.2f}s advantages={t_adv:.2f}s "
                    f"train={t_train:.2f}s total={t_total:.2f}s | "
                    f"throughput={tput:.0f} env-frames/s (step {step_f} / {step_elapsed:.2f}s) | "
                    f"{metrics_line}"
                )
                self._emit_step_perf(actor_msg, pbar=pbar)
                pbar.set_postfix_str(
                    f"{tput:.0f} env-f/s train={t_train:.1f}s total={t_total:.1f}s",
                    refresh=True,
                )
            pbar.update(1)
            self._profiler.on_step_end(step)

        pbar.close()
        if wbt is not None:
            wbt.join(timeout=120)
        logger.info("Actor async loop done")

    # ================================================================== #
    #  Weight transfer helpers
    # ================================================================== #
    #
    # All weight transfers use gloo (CPU tensors) to avoid CUDA stream
    # contention with ManiSkill's cudaDeviceSynchronize() in env.reset().
    #
    # Optimization: all trainable params are flattened into ONE contiguous
    # tensor and broadcast in a single gloo collective (instead of
    # hundreds of per-tensor broadcasts), cutting round-trip overhead
    # dramatically.
    # ================================================================== #

    def _sync_weights_from_actor(self):
        """Receive model weights broadcast from source actor rank."""
        src = self._wb_src_actor
        t0 = time.time()
        comm_level = logging.INFO if is_verbose_comm() else logging.DEBUG
        logger.log(comm_level, "Rollout rank %d waiting weights from actor %d", self.rank, src)
        hb_evt, hb_th = self._start_wait_heartbeat(f"rollout_wait_weights:src{src}")
        try:
            received = self._recv_weights_raw()
        finally:
            hb_evt.set()
            hb_th.join(timeout=0.2)
        payload_mb = _tensor_payload_bytes(received) / (1024 ** 2)
        self.worker.load_weights(received)
        logger.log(
            comm_level,
            "Rollout rank %d loaded weights from actor %d: params=%d payload=%.1f MiB elapsed=%.2fs",
            self.rank, src, len(received), payload_mb, time.time() - t0,
        )

    def _recv_weights_raw(self):
        """Receive flattened weight broadcast from actor.

        Returns raw state dict (no model load). CPU->GPU after receive.
        """
        src = self._wb_src_actor
        group = self._wb_group

        info_list = [None]
        logger.debug("_recv_weights_raw: waiting meta (src=%d)", src)
        dist.broadcast_object_list(info_list, src=src, group=group)
        info = info_list[0]
        total_numel = info["total_numel"]
        flat_dtype = info["flat_dtype"]
        logger.debug("_recv_weights_raw: meta received, total_numel=%d", total_numel)

        flat_buf = torch.empty(total_numel, dtype=flat_dtype, device="cpu")
        dist.broadcast(flat_buf, src=src, group=group)

        received = {}
        offset = 0
        for name, shape, dtype_name in info["params"]:
            dtype = getattr(torch, dtype_name)
            numel = 1
            for s in shape:
                numel *= s
            t = flat_buf[offset:offset + numel].view(shape)
            if dtype != flat_dtype:
                t = t.to(dtype=dtype)
            received[name] = t.to(self.device, non_blocking=True)
            offset += numel
        logger.debug("_recv_weights_raw: done, %d params", len(received))
        return received

    def _broadcast_weights(self):
        """Actor extracts state dict and broadcasts to all rollout workers."""
        comm_level = logging.INFO if is_verbose_comm() else logging.DEBUG
        targets = self.placement.rollout_ranks
        logger.log(
            comm_level,
            "Actor rank %d preparing weight broadcast to rollout ranks %s",
            self.rank, targets,
        )
        sd = self.worker.get_state_dict()
        self._broadcast_weights_from_sd(sd)
        logger.log(
            comm_level,
            "Actor rank %d weight broadcast completed to %d rollout ranks",
            self.rank, len(targets),
        )

    def _broadcast_weights_from_sd(self, sd):
        """Broadcast a pre-extracted state dict to rollout workers.

        Flattens all tensors into a single contiguous CPU tensor and performs
        ONE gloo broadcast (+ 1 meta broadcast) instead of hundreds of
        per-tensor broadcasts.
        """
        src = self._wb_src_actor
        group = self._wb_group
        keys = sorted(sd.keys())

        flat_dtype = torch.float32
        if keys:
            flat_dtype = sd[keys[0]].dtype

        total_numel = sum(sd[k].numel() for k in keys)
        flat_buf = torch.empty(total_numel, dtype=flat_dtype, device="cpu")
        offset = 0
        params_meta = []
        for k in keys:
            t = sd[k].detach().cpu().contiguous()
            numel = t.numel()
            if t.dtype != flat_dtype:
                flat_buf[offset:offset + numel].copy_(t.view(-1).to(flat_dtype))
            else:
                flat_buf[offset:offset + numel].copy_(t.view(-1))
            params_meta.append(
                (k, tuple(sd[k].shape), str(sd[k].dtype).replace("torch.", ""))
            )
            offset += numel

        info = {
            "params": params_meta,
            "total_numel": total_numel,
            "flat_dtype": flat_dtype,
        }
        comm_level = logging.INFO if is_verbose_comm() else logging.DEBUG
        mb = total_numel * flat_buf.element_size() / 1e6
        logger.log(
            comm_level,
            "Weight broadcast start: src_actor=%d params=%d flat=%.1f MB dtype=%s",
            src, len(keys), mb, str(flat_dtype).replace("torch.", ""),
        )
        dist.broadcast_object_list([info], src=src, group=group)
        dist.broadcast(flat_buf, src=src, group=group)
        logger.log(
            comm_level,
            "Weight broadcast done: src_actor=%d params=%d flat=%.1f MB",
            src, len(keys), mb,
        )
        del sd

    # ================================================================== #
    #  Data transfer helpers
    # ================================================================== #

    def _send_batch_to_actor(self, batch):
        """Send rollout batch to all paired actors via NCCL."""
        comm_level = logging.INFO if is_verbose_comm() else logging.DEBUG
        tensors_only = {
            k: v for k, v in batch.items()
            if isinstance(v, torch.Tensor)}
        payload_mb = _tensor_payload_bytes(tensors_only) / (1024 ** 2)
        logger.log(
            comm_level,
            "Rollout rank %d sending batch: tensor_keys=%d payload=%.1f MiB actors=%s",
            self.rank, len(tensors_only), payload_mb,
            [pipe.dst_rank for pipe in self.data_pipes],
        )
        for pipe in self.data_pipes:
            t0 = time.time()
            logger.log(
                comm_level,
                "Rollout rank %d -> actor %d send start",
                self.rank, pipe.dst_rank,
            )
            pipe.send_dict(tensors_only)
            logger.log(
                comm_level,
                "Rollout rank %d -> actor %d send done (%.2fs)",
                self.rank, pipe.dst_rank, time.time() - t0,
            )
        # Ensure NCCL kernel completion is attributed to send, not the next
        # rollout_epoch (async mode otherwise surfaces this as pre-reset stall).
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _recv_batch_from_rollout(self):
        """Receive rollout batch from paired rollout worker."""
        comm_level = logging.INFO if is_verbose_comm() else logging.DEBUG
        r_rank = self.placement.get_paired_rollout(self.rank)
        local_rank = int(torch.cuda.current_device())
        pair_key = f"pair_{r_rank}_{self.rank}"
        group = self.groups.get(pair_key)
        pipe = NCCLPipe(
            src_rank=r_rank,
            dst_rank=self.rank,
            local_rank=local_rank,
            group=group)
        logger.log(
            comm_level,
            "Actor rank %d waiting batch from rollout rank %d",
            self.rank, r_rank,
        )
        t0 = time.time()
        hb_evt, hb_th = self._start_wait_heartbeat(f"actor_wait_batch:src{r_rank}")
        try:
            batch = pipe.recv_dict()
        finally:
            hb_evt.set()
            hb_th.join(timeout=0.2)
        payload_mb = _tensor_payload_bytes(batch) / (1024 ** 2)
        logger.log(
            comm_level,
            "Actor rank %d received batch from rollout rank %d: tensor_keys=%d payload=%.1f MiB elapsed=%.2fs",
            self.rank, r_rank, len(batch), payload_mb, time.time() - t0,
        )
        return batch

    # ---- commv2 all-to-all helpers ----

    def _send_batch_v2(self, batch):
        """commv2: rollout sends batch fragments to all actors via all_to_all."""
        comm_level = logging.INFO if is_verbose_comm() else logging.DEBUG
        tensors_only = {
            k: v for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        first_t = next(iter(tensors_only.values()))
        env_dim = first_t.shape[1] if first_t.dim() >= 2 else first_t.shape[0]
        assert env_dim % self._a2a_n_actors == 0, (
            f"commv2: batch env-dim ({env_dim}) must be divisible by "
            f"n_actors ({self._a2a_n_actors})"
        )
        payload_mb = _tensor_payload_bytes(tensors_only) / (1024 ** 2)
        logger.log(
            comm_level,
            "Rollout rank %d commv2 all2all send: keys=%d payload=%.1f MiB "
            "env_dim=%d chunk_per_actor=%d",
            self.rank, len(tensors_only), payload_mb,
            env_dim, env_dim // self._a2a_n_actors,
        )
        t0 = time.time()
        all_to_all_batch(
            tensors_only, self._a2a_n_rollouts, self._a2a_n_actors,
            self._a2a_my_group_rank, self._all2all_group, self.device,
        )
        # Without this, CUDA/NCCL may finish during the next rollout's first GPU
        # work; ``t_send`` stays ~0 and ``rollout`` absorbs the wait (misleading).
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        logger.log(
            comm_level,
            "Rollout rank %d commv2 all2all send done (%.2fs)",
            self.rank, time.time() - t0,
        )

    def _recv_batch_v2(self):
        """commv2: actor receives batch fragments from all rollouts via all_to_all."""
        comm_level = logging.INFO if is_verbose_comm() else logging.DEBUG
        logger.log(
            comm_level,
            "Actor rank %d commv2 all2all recv: waiting for %d rollouts",
            self.rank, self._a2a_n_rollouts,
        )
        t0 = time.time()
        hb_evt, hb_th = self._start_wait_heartbeat("actor_all2all_recv")
        try:
            batch = all_to_all_batch(
                None, self._a2a_n_rollouts, self._a2a_n_actors,
                self._a2a_my_group_rank, self._all2all_group, self.device,
            )
        finally:
            hb_evt.set()
            hb_th.join(timeout=0.2)
        payload_mb = _tensor_payload_bytes(batch) / (1024 ** 2)
        elapsed = time.time() - t0
        recv_level = logging.WARNING if elapsed >= 30.0 else comm_level
        logger.log(
            recv_level,
            "Actor rank %d commv2 all2all recv done: keys=%d "
            "payload=%.1f MiB elapsed=%.2fs",
            self.rank, len(batch), payload_mb, elapsed,
        )
        return batch

    def _merge_batches(self, batches):
        """Merge multiple epoch batches."""
        if len(batches) == 1:
            return batches[0]
        merged = {}
        for k in batches[0]:
            vals = [b[k] for b in batches if k in b]
            if isinstance(vals[0], torch.Tensor):
                merged[k] = torch.cat(vals, dim=0)
            else:
                merged[k] = vals[0]
        return merged
