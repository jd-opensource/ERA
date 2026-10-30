"""Dedicated async MPS runner for the Pi0.5 ManiSkill experiment.

This file intentionally supports one topology only::

    P physical GPUs across one or more homogeneous nodes
      EnvRollout process  (global ranks 0..P-1)
      Actor process       (global ranks P..2P-1)

The two processes on each GPU are NVIDIA MPS clients.  Actor ranks use NCCL
for FSDP collectives; a same-GPU EnvRollout/Actor pair uses CPU/Gloo for the
rollout batch because NCCL does not support duplicate GPU ranks.  The first
Actor broadcasts policy weights to every EnvRollout through CPU/Gloo.

Keeping this runner topology-specific is deliberate: unsupported modes fail
during initialization instead of leaving inactive sync/disaggregated/v1/v2
branches in the code that make profiling and deadlock diagnosis ambiguous.
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
from joyrlinf.core.gpu_memory_guard import (
    read_cuda_peak_metrics,
    reset_cuda_peak_stats,
)
from joyrlinf.core.gpu_phase_gate import build_gpu_phase_gate
from joyrlinf.core.shared_host_kv import (
    KV_IPC_KEYS,
    SharedHostKvChannel,
    SharedHostKvMeta,
    resolve_prefix_kv_ipc,
)
from joyrlinf.core.launcher import (
    Placement,
    Role,
    get_role,
    init_distributed,
    parse_placement,
)
from joyrlinf.observability import MetricsEmitter, PerfSummaryTracker
from joyrlinf.observability.scalar_transport import (
    recv_rollout_scalars,
    send_rollout_scalars,
)
from joyrlinf.profiler import ProfilerManager
from joyrlinf.utils.logging import TrainingLogger, format_metrics_compact
from joyrlinf.utils.logging_setup import (
    GLOBAL_LOG_LEADER,
    is_verbose_comm,
    log_at_rank,
)
from joyrlinf.utils.run_paths import resolve_run_output_dir
from joyrlinf.workers.actor_worker import ActorWorker
from joyrlinf.workers.rollout_worker import CombinedEnvRolloutWorker

logger = logging.getLogger(__name__)

HEARTBEAT_LOG_INTERVAL_SEC = 60.0
THREAD_JOIN_TIMEOUT_SEC = 120.0

_ADV_METRIC_KEYS = [
    "adv_mean", "adv_std", "values_mean", "returns_mean",
]
_DATA_METRIC_KEYS = [
    "reward_nonzero_pct", "score_nonzero_pct", "groups_with_var_pct",
]
_TRAIN_METRIC_KEYS = [
    "policy_loss", "policy_loss_abs", "clip_fraction", "grad_norm",
    "ratio_mean", "ratio_max", "ratio_min", "ratio_abs",
    "entropy", "total_loss", "proximal_approx_kl", "approx_kl",
    "advantages_mean", "value_loss", "lr",
]


def _create_mps_process_groups(placement: Placement) -> dict:
    """Create only the process groups used by this MPS topology."""
    all_ranks = sorted(set(
        placement.rollout_ranks + placement.actor_ranks
    ))
    groups: dict = {
        # Actor ranks live on distinct GPUs, so NCCL is valid here.
        "actor": dist.new_group(ranks=placement.actor_ranks, backend="nccl"),
        # Rollout success aggregation is scalar-only and stays on CPU.
        "rollout": dist.new_group(ranks=placement.rollout_ranks, backend="gloo"),
    }

    # Every rank must call new_group in the same order, including non-members.
    for rollout_rank, actor_rank in zip(
        placement.rollout_ranks, placement.actor_ranks, strict=True,
    ):
        groups[f"batch_{rollout_rank}_{actor_rank}"] = dist.new_group(
            ranks=[rollout_rank, actor_rank], backend="gloo",
        )
    # Dedicated Gloo pairs for predict⊥train gate (never share batch pipe).
    for rollout_rank, actor_rank in zip(
        placement.rollout_ranks, placement.actor_ranks, strict=True,
    ):
        groups[f"gate_{rollout_rank}_{actor_rank}"] = dist.new_group(
            ranks=[rollout_rank, actor_rank], backend="gloo",
        )

    source_actor = placement.actor_ranks[0]
    weight_ranks = sorted(placement.rollout_ranks + [source_actor])
    groups["weight"] = dist.new_group(ranks=weight_ranks, backend="gloo")
    groups["shutdown"] = dist.new_group(ranks=all_ranks, backend="gloo")
    return groups


class _GlooTensorPipe:
    """Point-to-point CPU tensor-dict transport for one same-GPU pair."""

    def __init__(self, src_rank: int, dst_rank: int, group):
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.group = group

    def send_dict(self, data: dict[str, torch.Tensor]) -> None:
        tensors = {
            key: value.detach().cpu().contiguous()
            for key, value in data.items()
            if isinstance(value, torch.Tensor)
        }
        metadata = [
            (key, tuple(value.shape), value.dtype)
            for key, value in sorted(tensors.items())
        ]
        dist.send_object_list([metadata], dst=self.dst_rank, group=self.group)
        for key, _, _ in metadata:
            dist.send(tensors[key], dst=self.dst_rank, group=self.group)

    def recv_dict(self) -> dict[str, torch.Tensor]:
        metadata_obj = [None]
        dist.recv_object_list(
            metadata_obj, src=self.src_rank, group=self.group,
        )
        result: dict[str, torch.Tensor] = {}
        for key, shape, dtype in metadata_obj[0]:
            value = torch.empty(shape, dtype=dtype, device="cpu")
            dist.recv(value, src=self.src_rank, group=self.group)
            result[key] = value
        return result


def _tensor_payload_bytes(data) -> int:
    if isinstance(data, torch.Tensor):
        return data.numel() * data.element_size()
    if isinstance(data, dict):
        return sum(_tensor_payload_bytes(value) for value in data.values())
    if isinstance(data, (list, tuple)):
        return sum(_tensor_payload_bytes(value) for value in data)
    return 0


def _flush_log_handlers() -> None:
    for handler in logging.getLogger().handlers:
        flush = getattr(handler, "flush", None)
        if flush is not None:
            try:
                flush()
            except Exception:
                pass


def _validate_mps_only_config(
    cfg: DictConfig,
    placement: Placement,
    world_size: int,
) -> None:
    """Reject options whose execution paths were intentionally removed."""
    errors: list[str] = []

    if str(cfg.algorithm.get("pipeline_mode", "")) != "async":
        errors.append("algorithm.pipeline_mode must be 'async'")
    if str(cfg.algorithm.get("comm_mode", "")) != "mps":
        errors.append("algorithm.comm_mode must be 'mps'")
    adv_type = str(cfg.algorithm.get("adv_type", "")).lower()
    if adv_type not in ("grpo", "gae"):
        errors.append("algorithm.adv_type must be 'grpo' or 'gae'")
    if int(cfg.algorithm.get("rollout_epoch", 1)) != 1:
        errors.append("algorithm.rollout_epoch must be 1")
    rollout_ranks = placement.rollout_ranks
    actor_ranks = placement.actor_ranks
    if placement.env_ranks != rollout_ranks:
        errors.append("env and rollout ranks must be identical")
    if set(rollout_ranks) & set(actor_ranks):
        errors.append("EnvRollout and Actor ranks must be disjoint processes")
    if len(rollout_ranks) != len(actor_ranks):
        errors.append("EnvRollout and Actor must have the same rank count")
    expected_rollout = list(range(len(rollout_ranks)))
    expected_actor = list(range(len(rollout_ranks), 2 * len(rollout_ranks)))
    if rollout_ranks != expected_rollout or actor_ranks != expected_actor:
        errors.append(
            f"ranks must be EnvRollout={expected_rollout}, Actor={expected_actor}"
        )
    gpus_per_node = int(
        cfg.cluster.get("gpus_per_node", len(rollout_ranks))
    )
    num_nodes = int(cfg.cluster.get("num_nodes", 1))
    if len(rollout_ranks) != num_nodes * gpus_per_node:
        errors.append(
            f"rollout rank count {len(rollout_ranks)} must equal "
            f"num_nodes*gpus_per_node={num_nodes * gpus_per_node}"
        )
    physical_world = len(rollout_ranks)
    for rollout_rank, actor_rank in zip(
        rollout_ranks, actor_ranks, strict=True
    ):
        actor_physical_rank = actor_rank - physical_world
        if (
            actor_physical_rank != rollout_rank
            or actor_physical_rank // gpus_per_node
            != rollout_rank // gpus_per_node
        ):
            errors.append(
                "each MPS actor must pair with the same physical GPU/node: "
                f"rollout={rollout_rank}, actor={actor_rank}"
            )
            break
    if world_size != len(rollout_ranks) + len(actor_ranks):
        errors.append(
            f"world_size must be {len(rollout_ranks) + len(actor_ranks)}, "
            f"got {world_size}"
        )
    if dist.get_backend() != "gloo":
        errors.append("the default process group must use Gloo")

    if cfg.runner.get("resume_dir", None):
        errors.append("runner.resume_dir must be null (resume path was removed)")
    if int(cfg.runner.get("val_check_interval", -1)) > 0:
        errors.append(
            "runner.val_check_interval must be <= 0 (eval path was removed)"
        )
    if int(cfg.runner.get("save_interval", -1)) > 0:
        errors.append(
            "runner.save_interval must be <= 0 (checkpoint path was removed)"
        )

    sharding = str(
        cfg.actor.get("fsdp_config", {}).get("sharding_strategy", "")
    ).lower()
    if sharding != "no_shard":
        errors.append(
            "actor.fsdp_config.sharding_strategy must be 'no_shard'; only the "
            "source Actor extracts weights"
        )

    if errors:
        formatted = "\n  - ".join(errors)
        raise ValueError(f"Unsupported async MPS configuration:\n  - {formatted}")


class EmbodiedRunner:
    """Async EnvRollout/Actor orchestrator specialized for NVIDIA MPS."""

    def __init__(self, cfg: DictConfig, gpu_guard=None):
        self.cfg = cfg
        self.gpu_guard = gpu_guard
        self.rank, self.world_size, self.device = init_distributed()
        self.placement = parse_placement(cfg.cluster)

        validate_embodied_training_config(cfg)
        _validate_mps_only_config(cfg, self.placement, self.world_size)

        self.role = get_role(self.rank, self.placement)
        if self.role not in (Role.ROLLOUT, Role.ACTOR):
            raise ValueError(
                f"MPS runner rank {self.rank} has unsupported role {self.role.name}"
            )

        self.groups = _create_mps_process_groups(self.placement)
        self.actor_group = self.groups["actor"]
        self.rollout_group = self.groups["rollout"]
        self.weight_group = self.groups["weight"]
        self.shutdown_group = self.groups["shutdown"]
        self.first_rollout = self.placement.rollout_ranks[0]
        self.source_actor = self.placement.actor_ranks[0]
        self.metrics_rank = self.source_actor

        self.prefix_kv_ipc = resolve_prefix_kv_ipc(cfg)
        self.kv_shm: SharedHostKvChannel | None = None
        self._pending_kv_ack_slot: int | None = None

        if self.role == Role.ROLLOUT:
            paired_rank = self.placement.get_paired_actor(self.rank)
            batch_group = self.groups[f"batch_{self.rank}_{paired_rank}"]
            gate_group = self.groups[f"gate_{self.rank}_{paired_rank}"]
            self.batch_pipe = _GlooTensorPipe(self.rank, paired_rank, batch_group)
            self.gpu_phase_gate = build_gpu_phase_gate(
                cfg,
                role="rollout",
                group=gate_group,
                my_rank=self.rank,
                peer_rank=paired_rank,
            )
            self.worker = CombinedEnvRolloutWorker(
                self.rank, cfg, self.device, gpu_guard=gpu_guard,
                gpu_phase_gate=self.gpu_phase_gate,
            )
            if self.prefix_kv_ipc == "host_shm":
                self.kv_shm = SharedHostKvChannel(
                    name_prefix=f"joyrl_kv_r{self.rank}_a{paired_rank}",
                    n_slots=2,
                    is_owner=True,
                    src_rank=self.rank,
                    dst_rank=paired_rank,
                    group=batch_group,
                )
                self.worker.set_kv_shm_channel(self.kv_shm)
                logger.info(
                    "SharedHost KV IPC enabled (owner): rollout=%d actor=%d",
                    self.rank, paired_rank,
                )
        else:
            paired_rank = self.placement.get_paired_rollout(self.rank)
            batch_group = self.groups[f"batch_{paired_rank}_{self.rank}"]
            gate_group = self.groups[f"gate_{paired_rank}_{self.rank}"]
            self.batch_pipe = _GlooTensorPipe(paired_rank, self.rank, batch_group)
            self.gpu_phase_gate = build_gpu_phase_gate(
                cfg,
                role="actor",
                group=gate_group,
                my_rank=self.rank,
                peer_rank=paired_rank,
            )
            self.worker = ActorWorker(
                self.rank, cfg, self.device, self.actor_group,
                gpu_phase_gate=self.gpu_phase_gate,
            )
            if self.prefix_kv_ipc == "host_shm":
                self.kv_shm = SharedHostKvChannel(
                    name_prefix=f"joyrl_kv_r{paired_rank}_a{self.rank}",
                    n_slots=2,
                    is_owner=False,
                    src_rank=paired_rank,
                    dst_rank=self.rank,
                    group=batch_group,
                )
                logger.info(
                    "SharedHost KV IPC enabled (attach): actor=%d rollout=%d",
                    self.rank, paired_rank,
                )

        self.max_steps = int(cfg.runner.get("max_steps", 1000))
        self.global_step = 0
        self.output_dir = resolve_run_output_dir(cfg)
        self.env_frames_per_step = int(
            cfg.env.train.total_num_envs
            * cfg.env.train.max_steps_per_rollout_epoch
        )
        self._profiler = ProfilerManager.from_cfg(
            cfg,
            rank=self.rank,
            role="rollout" if self.role == Role.ROLLOUT else "actor",
            output_dir=self.output_dir,
        )

        self.metric_logger: TrainingLogger | None = None
        self.metrics_emitter: MetricsEmitter | None = None
        if self.rank == self.metrics_rank:
            logger_cfg = cfg.runner.get("logger", {})
            cfg_path = os.path.join(self.output_dir, "config.yaml")
            self.metric_logger = TrainingLogger(
                logger_cfg,
                rank=0,
                output_dir=self.output_dir,
                experiment_cfg=cfg,
                config_path=cfg_path if os.path.isfile(cfg_path) else None,
            )
            self.metrics_emitter = MetricsEmitter(
                self.metric_logger,
                PerfSummaryTracker(
                    pipeline_mode="async",
                    comm_mode="mps",
                    output_dir=self.output_dir,
                ),
                gpu_guard=self.gpu_guard,
            )
            self._save_experiment_config()

        log_at_rank(
            logger, self.rank, GLOBAL_LOG_LEADER, logging.INFO,
            "Async MPS runner: rank=%d/%d role=%s device=%s paired_rank=%d "
            "rollout_ranks=%s actor_ranks=%s output=%s",
            self.rank, self.world_size, self.role.name, self.device, paired_rank,
            self.placement.rollout_ranks, self.placement.actor_ranks,
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._barrier("mps_start", self.shutdown_group)
            if self.role == Role.ACTOR:
                self._barrier("actor_nccl_warmup", self.actor_group)
            self._barrier("mps_post_warmup", self.shutdown_group)

            log_at_rank(
                logger, self.rank, self.first_rollout, logging.INFO,
                "MPS communication ready: same-GPU batch=Gloo, "
                "weight/scalars=Gloo, Actor FSDP=NCCL",
            )
            if self.role == Role.ROLLOUT:
                self._run_rollout_loop()
            else:
                self._run_actor_loop()

            self._profiler.finalize()
            if self.metrics_emitter is not None:
                self.metrics_emitter.close()
                self.metrics_emitter = None
            if self.metric_logger is not None:
                self.metric_logger.close()
                self.metric_logger = None
            self._barrier("mps_exit", self.shutdown_group)
            logger.info(
                "Async MPS loop complete: role=%s rank=%d",
                self.role.name,
                self.rank,
            )
        finally:
            # SharedMemory objects otherwise survive until interpreter teardown.
            # Explicit close after the exit barrier lets the owner unlink only
            # after its paired consumer has finished using every slot.
            if self.kv_shm is not None:
                self.kv_shm.close()
                self.kv_shm = None
            if self.metrics_emitter is not None:
                self.metrics_emitter.close()
                self.metrics_emitter = None
            if self.metric_logger is not None:
                self.metric_logger.close()
                self.metric_logger = None

    # ------------------------------------------------------------------
    # Rollout process
    # ------------------------------------------------------------------

    def _run_rollout_loop(self) -> None:
        """Generate rollout N+1 while Actor trains on rollout N."""
        weight_lock = threading.Lock()
        pending_weights: list[dict[str, torch.Tensor] | None] = [None]
        received_version = [0]
        # ProcessGroupGloo is not safe for concurrent collectives across threads.
        # Do not enter weight_recv until after batch_send (no post-train weights
        # exist before that), and do not start the next batch Gloo until the
        # weight thread has left Gloo.
        # Note: do not emit weight_recv into weight_sync_timeline — its wall
        # clock spans the previous step's wait and misleads combined plots.
        start_weight_recv = threading.Event()
        weight_gloo_idle = threading.Event()
        weight_gloo_idle.set()

        self._load_initial_actor_weights()
        self.worker.weight_version = 0

        def receive_weights() -> None:
            # The actor's final update has no subsequent rollout consumer.
            # Waiting for it kept this daemon thread inside Gloo during
            # process-group teardown and could turn an otherwise complete run
            # into SIGABRT/Connection-closed errors.
            for iteration in range(max(self.max_steps - 1, 0)):
                logger.debug(
                    "Weight receiver iteration %d waiting for post-send permit",
                    iteration,
                )
                start_weight_recv.wait()
                start_weight_recv.clear()
                weight_gloo_idle.clear()
                try:
                    weights = self._recv_weights_raw()
                    with weight_lock:
                        pending_weights[0] = weights
                        received_version[0] += 1
                finally:
                    weight_gloo_idle.set()

        weight_thread = threading.Thread(
            target=receive_weights,
            daemon=True,
            name=f"mps-weight-recv-rank{self.rank}",
        )
        weight_thread.start()

        last_weight_apply_step = 0
        for step in range(1, self.max_steps + 1):
            step_start_ns = time.time_ns()
            step_start = time.time()
            self._profiler.on_step_begin(step)
            reset_cuda_peak_stats(self.device)

            weight_sync_timeline: list[dict[str, int | str]] = []
            weight_load_start = time.time()
            load_start_ns = time.time_ns()
            with self._profiler.range("mps_weight_load", step):
                with weight_lock:
                    if pending_weights[0] is not None:
                        self.worker.load_weights(pending_weights[0])
                        self.worker.weight_version = received_version[0]
                        pending_weights[0] = None
                        last_weight_apply_step = step
                        did_load = True
                    else:
                        did_load = False
            load_end_ns = time.time_ns()
            if did_load:
                weight_sync_timeline.append({
                    "name": "weight_load",
                    "start_wall_ns": load_start_ns,
                    "end_wall_ns": load_end_ns,
                })
            weight_load_time = time.time() - weight_load_start
            staleness = step - last_weight_apply_step

            rollout_start_ns = time.time_ns()
            rollout_start = time.time()
            # Tag this epoch's gate messages so the paired actor credits them
            # to the right training step (epoch N feeds actor step N - 1).
            self.gpu_phase_gate.begin_epoch(step)
            if self.kv_shm is not None:
                self.worker.begin_kv_shm_epoch()
            with self._profiler.range("mps_rollout", step):
                batch, success_stats, rollout_timeline = self.worker.rollout_epoch(
                    mode="train", return_timeline=True,
                )
            # Serialize main-thread Gloo against the weight thread before reduce/send.
            weight_gloo_idle.wait()
            global_success = self._reduce_rollout_success(
                success_stats["local_success_count"]
            )
            rollout_time = time.time() - rollout_start
            rollout_end_ns = time.time_ns()

            mem_metrics = read_cuda_peak_metrics(self.device, prefix="rollout_")
            rollout_scalars = self._build_rollout_scalars(
                global_success,
                weight_load_time=weight_load_time,
                rollout_time=rollout_time,
            )
            for key, value in mem_metrics.items():
                rollout_scalars[f"memory/{key}"] = value

            send_start = time.time()
            with self._profiler.range("mps_rollout_send", step):
                self._send_batch(batch)
            send_time = time.time() - send_start
            rollout_scalars["time/rollout_send"] = send_time
            if self.rank == self.first_rollout:
                send_rollout_scalars(
                    self.metrics_rank,
                    rollout_scalars,
                    group=self.shutdown_group,
                )
            del batch
            # Post-train weights are useful only if another rollout follows.
            if step < self.max_steps:
                start_weight_recv.set()

            step_end_ns = time.time_ns()
            step_time = time.time() - step_start
            # Keep full rollout step span (incl. weight apply) for env alignment.
            self._save_timeline({
                "step": step,
                "rank": self.rank,
                "role": "rollout",
                "step_timeline": {
                    "start_wall_ns": step_start_ns,
                    "end_wall_ns": step_end_ns,
                },
                "weight_sync_timeline": weight_sync_timeline,
                "env_rollout_timeline": [{
                    "rollout_epoch_idx": 0,
                    "epoch_timeline": {
                        "start_wall_ns": rollout_start_ns,
                        "end_wall_ns": rollout_end_ns,
                    },
                    "env_reset": rollout_timeline["env_reset"],
                    "interactions": rollout_timeline["interactions"],
                    "bootstrap_infer": rollout_timeline["bootstrap_infer"],
                }],
                "memory": mem_metrics,
            })

            if self.rank == self.first_rollout:
                throughput, step_frames, elapsed = self._step_throughput(step_time)
                self._save_benchmark_timing({
                    "step": step,
                    "role": "rollout",
                    "rank": self.rank,
                    "total_time_s": step_time,
                    "rollout_time_s": rollout_time,
                    "weight_load_time_s": weight_load_time,
                    "send_time_s": send_time,
                    "throughput_env_s": throughput,
                    "env_frames": step_frames,
                })
                success_denom = self._success_denominator()
                success_pct = 100.0 * global_success / max(success_denom, 1)
                peak_alloc = mem_metrics.get("rollout_pt_peak_alloc_mib", 0.0)
                peak_rsv = mem_metrics.get("rollout_pt_peak_reserved_mib", 0.0)
                free_mib = mem_metrics.get("rollout_free_mib", 0.0)
                message = (
                    f"[MPS] Rollout step {step}/{self.max_steps}: "
                    f"weight_load={weight_load_time:.2f}s "
                    f"rollout={rollout_time:.2f}s send={send_time:.2f}s "
                    f"total={step_time:.2f}s | staleness={staleness} | "
                    f"throughput={throughput:.0f} env-frames/s "
                    f"(step {step_frames} / {elapsed:.2f}s) | "
                    f"train/success_once={global_success}/{success_denom} "
                    f"({success_pct:.2f}%) | "
                    f"mem peak_alloc={peak_alloc:.0f}MiB "
                    f"peak_rsv={peak_rsv:.0f}MiB free={free_mib:.0f}MiB"
                )
                tqdm.write(message)
                logger.info(message)

            self._profiler.on_step_end(step)

        weight_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
        if weight_thread.is_alive():
            logger.warning("Weight receiver thread did not exit within %.0fs", THREAD_JOIN_TIMEOUT_SEC)

    # ------------------------------------------------------------------
    # Actor process
    # ------------------------------------------------------------------

    def _run_actor_loop(self) -> None:
        """Train Actor N while EnvRollout starts collecting N+1."""
        is_source = self.rank == self.source_actor
        # Gate service must be live before EnvRollout's first predict request
        # (step 1 collect overlaps actor's initial recv_batch wait).
        self.gpu_phase_gate.start_actor_service()
        progress = tqdm(
            total=self.max_steps,
            desc=f"Actor[MPS] rank {self.rank}",
            disable=not is_source,
        )

        training_done = threading.Event()
        weights_extracted = threading.Event()
        weights_extracted.set()
        # Keep weight Gloo broadcast off the batch/scalar Gloo path. Extract can
        # still overlap the next step; only the Gloo portion is serialized.
        weight_broadcast_idle = threading.Event()
        weight_broadcast_idle.set()
        weight_sync_lock = threading.Lock()
        # step -> intervals; filled by broadcast thread, consumed when saving timeline.
        weight_sync_by_step: dict[int, list[dict[str, int | str]]] = {}

        def broadcast_weights() -> None:
            # No rollout consumes weights produced by the final actor step.
            for iteration in range(max(self.max_steps - 1, 0)):
                step = iteration + 1
                logger.debug("Weight broadcaster iteration %d waiting", iteration)
                training_done.wait()
                training_done.clear()
                intervals: list[dict[str, int | str]] = []

                extract_start_ns = time.time_ns()
                # May run after on_step_end; still emit torch annotation when active.
                with self._profiler.range("mps_weight_extract", step):
                    torch.cuda.synchronize()
                    state_dict = self.worker.get_state_dict()
                extract_end_ns = time.time_ns()
                intervals.append({
                    "name": "weight_extract",
                    "start_wall_ns": extract_start_ns,
                    "end_wall_ns": extract_end_ns,
                })
                # Unblock actor adv path ASAP; Gloo broadcast itself must not
                # overlap the next batch_recv (ProcessGroupGloo is not thread-safe).
                weights_extracted.set()

                weight_broadcast_idle.clear()
                try:
                    bcast_start_ns = time.time_ns()
                    with self._profiler.range("mps_weight_broadcast", step):
                        self._broadcast_weights_from_state_dict(state_dict)
                    bcast_end_ns = time.time_ns()
                    intervals.append({
                        "name": "weight_broadcast",
                        "start_wall_ns": bcast_start_ns,
                        "end_wall_ns": bcast_end_ns,
                    })
                    with weight_sync_lock:
                        weight_sync_by_step[step] = intervals
                finally:
                    weight_broadcast_idle.set()

        # Rollout cannot start until it has the Actor's initial policy.
        if is_source:
            self._broadcast_current_weights()
        weight_thread = None
        if is_source:
            weight_thread = threading.Thread(
                target=broadcast_weights,
                daemon=True,
                name=f"mps-weight-bcast-rank{self.rank}",
            )
            weight_thread.start()

        try:
            for step in range(1, self.max_steps + 1):
                step_start_ns = time.time_ns()
                step_start = time.time()
                self._profiler.on_step_begin(step)
                self.worker.current_version = step - 1
                reset_cuda_peak_stats(self.device)

                if is_source:
                    weight_broadcast_idle.wait()

                recv_start = time.time()
                with self._profiler.range("mps_actor_recv", step):
                    batch = self._recv_batch()
                recv_time = time.time() - recv_start
                rollout_metrics = (
                    recv_rollout_scalars(
                        self.first_rollout,
                        group=self.shutdown_group,
                    )
                    if is_source else {}
                )
                self.worker.receive_rollout_batch(batch)
                del batch

                extract_wait_start = time.time()
                if is_source:
                    weights_extracted.wait()
                    weights_extracted.clear()
                extract_wait_time = time.time() - extract_wait_start

                adv_start = time.time()
                with self._profiler.range("mps_actor_adv", step):
                    adv_metrics = self.worker.compute_advantages()
                adv_time = time.time() - adv_start

                self._barrier(f"actor_step_{step}", self.actor_group)
                self._prepare_histograms(step)
                # Training step N consumes gate credits of rollout epoch N + 1.
                self.gpu_phase_gate.set_training_step(step)
                if step == self.max_steps:
                    # No rollout epoch overlaps the final training step, so no
                    # predict credits will arrive; disarm the phase gate or the
                    # first microbatch waits forever (shutdown deadlock).
                    self.gpu_phase_gate.mark_no_more_predicts()
                train_start_ns = time.time_ns()
                train_start = time.time()
                with self._profiler.range("mps_actor_train", step):
                    train_metrics = self.worker.run_training()
                train_time = time.time() - train_start
                train_end_ns = time.time_ns()
                # Release SharedHost slot so rollout may reuse it for epoch+2.
                if (
                    self.kv_shm is not None
                    and self._pending_kv_ack_slot is not None
                ):
                    self.kv_shm.send_ack(self._pending_kv_ack_slot)
                    self._pending_kv_ack_slot = None
                histograms = self._pop_histograms()

                if is_source and step < self.max_steps:
                    torch.cuda.synchronize()
                    training_done.set()

                # Attach weight sync from the *previous* step if the async
                # broadcaster has finished it (step 1 has none yet).
                with weight_sync_lock:
                    weight_sync_timeline = list(
                        weight_sync_by_step.pop(step - 1, [])
                    )

                mem_metrics = read_cuda_peak_metrics(self.device, prefix="actor_")
                memory_metrics = self._merge_step_memory_metrics(
                    rollout_metrics, mem_metrics,
                )
                compute_timeline = self.worker.pop_train_compute_timeline()
                actor_compute_time = sum(
                    max(
                        int(event.get("end_wall_ns", 0))
                        - int(event.get("start_wall_ns", 0)),
                        0,
                    )
                    for event in compute_timeline
                ) / 1_000_000_000
                actor_gate_wait_time = (
                    self.gpu_phase_gate.training_credit_wait_seconds()
                )
                actor_other_time = max(
                    train_time - actor_compute_time - actor_gate_wait_time,
                    0.0,
                )
                span_events = list(compute_timeline) + list(weight_sync_timeline)
                if span_events:
                    compute_start_ns = min(
                        int(ev["start_wall_ns"]) for ev in span_events
                    )
                    compute_end_ns = max(
                        int(ev["end_wall_ns"]) for ev in span_events
                    )
                else:
                    compute_start_ns = train_start_ns
                    compute_end_ns = train_end_ns

                step_end_ns = time.time_ns()
                step_time = time.time() - step_start
                self._save_timeline({
                    "step": step,
                    "rank": self.rank,
                    "role": "actor",
                    "step_timeline": {
                        "start_wall_ns": compute_start_ns,
                        "end_wall_ns": compute_end_ns,
                    },
                    "actor_compute_timeline": compute_timeline,
                    "weight_sync_timeline": weight_sync_timeline,
                    "memory": mem_metrics,
                })

                if is_source:
                    throughput, step_frames, elapsed = self._step_throughput(step_time)
                    self._save_benchmark_timing({
                        "step": step,
                        "role": "actor",
                        "rank": self.rank,
                        "total_time_s": step_time,
                        "recv_time_s": recv_time,
                        "adv_time_s": adv_time,
                        "train_time_s": train_time,
                        "throughput_env_s": throughput,
                        "env_frames": step_frames,
                    })
                    self._emit_metrics(
                        step,
                        train_metrics=train_metrics,
                        adv_metrics=adv_metrics,
                        rollout_metrics=rollout_metrics,
                        histograms=histograms or None,
                        time_metrics=self._canonical_time_metrics(
                            recv=recv_time,
                            wait_extract=extract_wait_time,
                            actor_adv=adv_time,
                            actor_train=train_time,
                            actor_train_wall=train_time,
                            actor_compute=actor_compute_time,
                            actor_gate_wait=actor_gate_wait_time,
                            actor_train_other=actor_other_time,
                            total=step_time,
                            throughput=throughput,
                        ),
                        memory_metrics=memory_metrics,
                    )
                    metrics_text = self._format_step_metrics(
                        adv_metrics, train_metrics,
                    )
                    peak_alloc = mem_metrics.get("actor_pt_peak_alloc_mib", 0.0)
                    peak_rsv = mem_metrics.get("actor_pt_peak_reserved_mib", 0.0)
                    free_mib = mem_metrics.get("actor_free_mib", 0.0)
                    sum_peak = memory_metrics.get("pt_peak_alloc_sum_mib")
                    sum_txt = (
                        f" sum_peak={sum_peak:.0f}MiB" if sum_peak is not None else ""
                    )
                    message = (
                        f"[MPS] Actor step {step}/{self.max_steps}: "
                        f"recv={recv_time:.2f}s wait_extract={extract_wait_time:.2f}s "
                        f"advantages={adv_time:.2f}s train={train_time:.2f}s "
                        f"(compute={actor_compute_time:.2f}s "
                        f"gate_wait={actor_gate_wait_time:.2f}s "
                        f"other={actor_other_time:.2f}s) "
                        f"total={step_time:.2f}s | throughput={throughput:.0f} "
                        f"env-frames/s (step {step_frames} / {elapsed:.2f}s) | "
                        f"{metrics_text} | "
                        f"mem peak_alloc={peak_alloc:.0f}MiB "
                        f"peak_rsv={peak_rsv:.0f}MiB free={free_mib:.0f}MiB"
                        f"{sum_txt}"
                    )
                    self._emit_step_perf(message, progress)
                    progress.set_postfix_str(
                        f"{throughput:.0f} env-f/s train={train_time:.1f}s "
                        f"peak={peak_alloc:.0f}M total={step_time:.1f}s",
                        refresh=True,
                    )
                progress.update(1)
                self._profiler.on_step_end(step)

            # Flush the last step's async weight-sync timeline (never attached above).
            if is_source and weight_thread is not None:
                weight_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
                with weight_sync_lock:
                    last_sync = list(weight_sync_by_step.pop(self.max_steps, []))
                if last_sync:
                    self._save_timeline({
                        "step": self.max_steps,
                        "rank": self.rank,
                        "role": "actor",
                        "step_timeline": {
                            "start_wall_ns": int(last_sync[0]["start_wall_ns"]),
                            "end_wall_ns": int(last_sync[-1]["end_wall_ns"]),
                        },
                        "actor_compute_timeline": [],
                        "weight_sync_timeline": last_sync,
                        "memory": {},
                    })
                weight_thread = None
        finally:
            self.gpu_phase_gate.stop_actor_service()

        progress.close()
        if weight_thread is not None:
            weight_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
            if weight_thread.is_alive():
                logger.warning(
                    "Weight broadcaster thread did not exit within %.0fs",
                    THREAD_JOIN_TIMEOUT_SEC,
                )

    # ------------------------------------------------------------------
    # Same-GPU rollout batch transfer
    # ------------------------------------------------------------------

    def _send_batch(self, batch: dict) -> None:
        tensors = {
            key: value for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        kv_ipc_meta: dict | None = None
        if self.kv_shm is not None and "prefix_kv" in tensors:
            # KV already resides in the published write slot (pool bound to shm).
            n_rows = int(tensors["prefix_kv"].shape[0])
            kv_ipc_meta = self.kv_shm.publish(n_rows).to_dict()
            for key in KV_IPC_KEYS:
                tensors.pop(key, None)

        payload_mib = _tensor_payload_bytes(tensors) / (1024 ** 2)
        kv_mib = 0.0
        if kv_ipc_meta is not None:
            # Report logical KV size for logs (not copied over Gloo).
            kv = batch.get("prefix_kv")
            if isinstance(kv, torch.Tensor):
                kv_mib = kv.numel() * kv.element_size() / (1024 ** 2)
        start = time.time()
        heartbeat_stop, heartbeat = self._start_wait_heartbeat("batch_send")
        try:
            self._send_batch_tensors(tensors, kv_ipc_meta=kv_ipc_meta)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=0.2)
        logger.info(
            "MPS batch send: rollout=%d actor=%d payload=%.1f MiB "
            "kv_ipc=%.1f MiB elapsed=%.2fs",
            self.rank, self.placement.get_paired_actor(self.rank),
            payload_mib, kv_mib, time.time() - start,
        )

    def _recv_batch(self) -> dict[str, torch.Tensor]:
        rollout_rank = self.placement.get_paired_rollout(self.rank)
        start = time.time()
        heartbeat_stop, heartbeat = self._start_wait_heartbeat("batch_recv")
        try:
            batch, kv_ipc_meta = self._recv_batch_tensors()
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=0.2)

        kv_mib = 0.0
        if kv_ipc_meta is not None:
            if self.kv_shm is None:
                raise RuntimeError("Received kv_ipc meta but SharedHost is off")
            meta = SharedHostKvMeta.from_dict(kv_ipc_meta)
            kv_views = self.kv_shm.view(meta)
            batch.update(kv_views)
            self._pending_kv_ack_slot = meta.slot
            kv = kv_views.get("prefix_kv")
            if isinstance(kv, torch.Tensor):
                kv_mib = kv.numel() * kv.element_size() / (1024 ** 2)

        payload_mib = _tensor_payload_bytes(batch) / (1024 ** 2)
        logger.info(
            "MPS batch recv: actor=%d rollout=%d payload=%.1f MiB "
            "kv_ipc=%.1f MiB elapsed=%.2fs",
            self.rank, rollout_rank, payload_mib - kv_mib, kv_mib,
            time.time() - start,
        )
        return batch

    def _send_batch_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        kv_ipc_meta: dict | None,
    ) -> None:
        """Gloo-send small tensors (+ optional SharedHost meta). Never sends KV."""
        cpu = {
            key: value.detach().cpu().contiguous()
            for key, value in tensors.items()
        }
        metadata = [
            (key, tuple(value.shape), value.dtype)
            for key, value in sorted(cpu.items())
        ]
        dist.send_object_list(
            [{"fields": metadata, "kv_ipc": kv_ipc_meta}],
            dst=self.batch_pipe.dst_rank,
            group=self.batch_pipe.group,
        )
        for key, _, _ in metadata:
            dist.send(cpu[key], dst=self.batch_pipe.dst_rank, group=self.batch_pipe.group)

    def _recv_batch_tensors(self) -> tuple[dict[str, torch.Tensor], dict | None]:
        header_obj: list = [None]
        dist.recv_object_list(
            header_obj, src=self.batch_pipe.src_rank, group=self.batch_pipe.group,
        )
        header = header_obj[0]
        # Backward-compatible: old path sent bare metadata list.
        if isinstance(header, list):
            metadata = header
            kv_ipc_meta = None
        else:
            metadata = header["fields"]
            kv_ipc_meta = header.get("kv_ipc")
        result: dict[str, torch.Tensor] = {}
        for key, shape, dtype in metadata:
            value = torch.empty(shape, dtype=dtype, device="cpu")
            dist.recv(
                value, src=self.batch_pipe.src_rank, group=self.batch_pipe.group,
            )
            result[key] = value
        return result, kv_ipc_meta

    # ------------------------------------------------------------------
    # Actor -> EnvRollout weight broadcast
    # ------------------------------------------------------------------

    def _load_initial_actor_weights(self) -> None:
        start = time.time()
        weights = self._recv_weights_raw()
        self.worker.load_weights(weights)
        logger.info(
            "Initial Actor weights loaded: rank=%d params=%d payload=%.1f MiB "
            "elapsed=%.2fs",
            self.rank, len(weights), _tensor_payload_bytes(weights) / (1024 ** 2),
            time.time() - start,
        )

    def _recv_weights_raw(self) -> dict[str, torch.Tensor]:
        heartbeat_stop, heartbeat = self._start_wait_heartbeat("weight_recv")
        try:
            metadata_obj = [None]
            dist.broadcast_object_list(
                metadata_obj,
                src=self.source_actor,
                group=self.weight_group,
            )
            metadata = metadata_obj[0]
            flat = torch.empty(
                metadata["total_numel"],
                dtype=metadata["flat_dtype"],
                device="cpu",
            )
            dist.broadcast(flat, src=self.source_actor, group=self.weight_group)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=0.2)

        result: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape, dtype_name in metadata["params"]:
            dtype = getattr(torch, dtype_name)
            numel = 1
            for size in shape:
                numel *= size
            value = flat[offset:offset + numel].view(shape)
            if value.dtype != dtype:
                value = value.to(dtype=dtype)
            result[name] = value.to(self.device, non_blocking=True)
            offset += numel
        return result

    def _broadcast_current_weights(self) -> None:
        state_dict = self.worker.get_state_dict()
        self._broadcast_weights_from_state_dict(state_dict)

    def _broadcast_weights_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        keys = sorted(state_dict)
        flat_dtype = state_dict[keys[0]].dtype if keys else torch.float32
        total_numel = sum(state_dict[key].numel() for key in keys)
        flat = torch.empty(total_numel, dtype=flat_dtype, device="cpu")

        params = []
        offset = 0
        for key in keys:
            value = state_dict[key].detach().cpu().contiguous().view(-1)
            numel = value.numel()
            flat[offset:offset + numel].copy_(
                value if value.dtype == flat_dtype else value.to(flat_dtype)
            )
            params.append((
                key,
                tuple(state_dict[key].shape),
                str(state_dict[key].dtype).removeprefix("torch."),
            ))
            offset += numel

        metadata = {
            "params": params,
            "total_numel": total_numel,
            "flat_dtype": flat_dtype,
        }
        start = time.time()
        dist.broadcast_object_list(
            [metadata], src=self.source_actor, group=self.weight_group,
        )
        dist.broadcast(flat, src=self.source_actor, group=self.weight_group)
        logger.info(
            "Actor weight broadcast: params=%d payload=%.1f MiB elapsed=%.2fs",
            len(keys), flat.numel() * flat.element_size() / (1024 ** 2),
            time.time() - start,
        )

    # ------------------------------------------------------------------
    # Metrics, timeline, and diagnostics
    # ------------------------------------------------------------------

    def _reduce_rollout_success(self, local_success: int) -> int:
        # rollout_group is Gloo, so the reduction tensor must stay on CPU.
        value = torch.tensor([local_success], dtype=torch.int64, device="cpu")
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=self.rollout_group)
        return int(value.item())

    def _build_rollout_scalars(
        self,
        global_success: int,
        *,
        weight_load_time: float,
        rollout_time: float,
    ) -> dict[str, float]:
        metrics = CombinedEnvRolloutWorker.make_train_runner_metrics(
            global_success, self._success_denominator(),
        )
        metrics.update({
            "time/rollout_weight_load": weight_load_time,
            "time/rollout_core": rollout_time,
            "time/rollout_eval": 0.0,
            "time/rollout_total": weight_load_time + rollout_time,
        })
        return metrics

    def _success_denominator(self) -> int:
        return int(self.cfg.env.train.total_num_envs)

    def _step_throughput(self, elapsed: float) -> tuple[float, int, float]:
        elapsed = max(float(elapsed), 1e-6)
        return self.env_frames_per_step / elapsed, self.env_frames_per_step, elapsed

    def _save_experiment_config(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "config.yaml")
        OmegaConf.save(self.cfg, path)
        logger.info("Saved experiment config to %s", path)

    def _save_timeline(self, item: dict) -> None:
        directory = os.path.join(self.output_dir, "timeline")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"timeline_rank{self.rank}.jsonl")
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")

    def _save_benchmark_timing(self, item: dict) -> None:
        if not bool(self.cfg.runner.get("benchmark_timing", False)):
            return
        directory = os.path.join(self.output_dir, "benchmark")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{item['role']}_timings.jsonl")
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
            stream.flush()

    def _prepare_histograms(self, step: int) -> None:
        if self.metric_logger is None:
            return
        interval = int(getattr(self.metric_logger, "histogram_interval", 0))
        if interval > 0 and step % interval == 0:
            self.worker.enable_histogram_collection(True)

    def _pop_histograms(self) -> dict[str, list[float]]:
        if hasattr(self.worker, "pop_histogram_samples"):
            return self.worker.pop_histogram_samples()
        return {}

    def _merge_step_memory_metrics(
        self,
        rollout_metrics: dict[str, float],
        actor_mem: dict[str, float],
    ) -> dict[str, float]:
        """Combine rollout (Gloo) + actor process peaks for TensorBoard."""
        from joyrlinf.observability.metrics import (
            memory_snapshot_to_metrics,
            partition_stripped_metrics,
        )

        memory: dict[str, float] = {}
        parts = partition_stripped_metrics(rollout_metrics)
        memory.update(parts.get("memory", {}))
        memory.update(actor_mem)

        rollout_peak = memory.get("rollout_pt_peak_alloc_mib")
        actor_peak = memory.get("actor_pt_peak_alloc_mib")
        if rollout_peak is not None and actor_peak is not None:
            # Conservative same-GPU estimate (peaks may be non-overlapping under
            # predict⊥train; still useful as an upper-bound process sum).
            memory["pt_peak_alloc_sum_mib"] = float(rollout_peak) + float(actor_peak)
            memory["pt_peak_reserved_sum_mib"] = float(
                memory.get("rollout_pt_peak_reserved_mib", 0.0)
            ) + float(memory.get("actor_pt_peak_reserved_mib", 0.0))

        # Prefer actor-side device free at step end (after train); keep rollout
        # free as a separate series already in ``memory``.
        if self.gpu_guard is not None:
            snap = self.gpu_guard.snapshot()
            memory.update(memory_snapshot_to_metrics(snap, guard=self.gpu_guard))
        elif "actor_free_mib" in memory:
            memory.setdefault("free_mib", memory["actor_free_mib"])
            memory.setdefault("total_mib", memory.get("actor_total_mib", 0.0))
        return memory

    def _emit_metrics(
        self,
        step: int,
        *,
        train_metrics: dict,
        adv_metrics: dict,
        rollout_metrics: dict,
        time_metrics: dict,
        histograms: dict[str, list[float]] | None,
        memory_metrics: dict[str, float] | None = None,
    ) -> None:
        if self.metrics_emitter is None:
            return
        self.metrics_emitter.emit_step(
            step,
            train=train_metrics,
            adv=adv_metrics,
            stripped_rollout=rollout_metrics,
            time_metrics=time_metrics,
            histograms=histograms,
            memory_metrics=memory_metrics,
        )

    @staticmethod
    def _canonical_time_metrics(**values: float) -> dict[str, float]:
        result = dict(values)
        actor_train = result.get("actor_train")
        if actor_train is not None:
            result.setdefault("train", actor_train)
        throughput = result.get("throughput")
        if throughput is not None:
            result.setdefault("throughput_env_s", throughput)
        return result

    @staticmethod
    def _format_step_metrics(adv_metrics: dict, train_metrics: dict) -> str:
        from joyrlinf.observability.metrics import split_adv_data_metrics

        adv_part, data_part = split_adv_data_metrics(adv_metrics)
        adv_text = format_metrics_compact(adv_part, keys=_ADV_METRIC_KEYS)
        train_text = format_metrics_compact(train_metrics, keys=_TRAIN_METRIC_KEYS)
        data_text = format_metrics_compact(data_part, keys=_DATA_METRIC_KEYS)
        parts = [f"adv: {adv_text}", f"train: {train_text}"]
        if data_text != "(empty)":
            parts.append(f"data: {data_text}")
        return " | ".join(parts)

    def _emit_step_perf(self, message: str, progress: tqdm) -> None:
        logger.info(message)
        if not progress.disable:
            progress.write(f"[rank {self.rank}] {message}", file=sys.stdout)
        _flush_log_handlers()

    def _start_wait_heartbeat(
        self,
        phase: str,
        interval_s: float = HEARTBEAT_LOG_INTERVAL_SEC,
    ) -> tuple[threading.Event, threading.Thread]:
        stop = threading.Event()
        start = time.time()

        def emit() -> None:
            while not stop.wait(interval_s):
                logger.info(
                    "Heartbeat: phase=%s role=%s rank=%d waiting=%.1fs",
                    phase, self.role.name, self.rank, time.time() - start,
                )

        thread = threading.Thread(
            target=emit,
            daemon=True,
            name=f"heartbeat-{self.rank}-{phase[:24]}",
        )
        thread.start()
        return stop, thread

    def _barrier(self, name: str, group) -> None:
        backend = dist.get_backend(group)
        log_level = logging.INFO if is_verbose_comm() else logging.DEBUG
        logger.log(
            log_level,
            "Barrier start: name=%s role=%s rank=%d backend=%s",
            name, self.role.name, self.rank, backend,
        )
        stop, heartbeat = self._start_wait_heartbeat(f"barrier:{name}")
        start = time.time()
        kwargs = (
            {"device_ids": [int(torch.cuda.current_device())]}
            if backend == "nccl" else {}
        )
        try:
            dist.barrier(group=group, **kwargs)
        finally:
            stop.set()
            heartbeat.join(timeout=0.2)
        elapsed = time.time() - start
        logger.log(
            logging.WARNING if elapsed >= 1.0 else log_level,
            "Barrier done: name=%s role=%s rank=%d elapsed=%.2fs",
            name, self.role.name, self.rank, elapsed,
        )
