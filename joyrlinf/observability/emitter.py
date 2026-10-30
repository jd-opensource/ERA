"""Emit ``StepMetrics`` to TensorBoard / WandB and perf summary."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from joyrlinf.observability.metrics import (
    StepMetrics,
    partition_stripped_metrics,
    split_adv_data_metrics,
)
from joyrlinf.observability.perf_summary import PerfSummaryTracker
from joyrlinf.utils.logging import TrainingLogger

if TYPE_CHECKING:
    from joyrlinf.core.gpu_memory_guard import GPUMemoryGuard

logger = logging.getLogger(__name__)


class MetricsEmitter:
    """Single-writer metrics sink for embodied training (P0)."""

    def __init__(
        self,
        training_logger: TrainingLogger | None,
        perf_tracker: PerfSummaryTracker | None,
        *,
        gpu_guard: GPUMemoryGuard | None = None,
    ):
        self._logger = training_logger
        self._perf = perf_tracker
        self._gpu_guard = gpu_guard
        self._pending_eval: dict[int, dict[str, float]] = {}

    def ingest_eval_broadcast(self, step: int, metrics: dict[str, float]) -> None:
        """Store eval metrics tagged with the step they were collected on."""
        if step > 0 and metrics:
            self._pending_eval[int(step)] = dict(metrics)

    def emit_step(
        self,
        step: int,
        *,
        train: dict[str, float] | None = None,
        adv: dict[str, float] | None = None,
        stripped_rollout: dict[str, float] | None = None,
        time_metrics: dict[str, float] | None = None,
        eval_metrics: dict[str, float] | None = None,
        memory_metrics: dict[str, float] | None = None,
        histograms: dict[str, list[float]] | None = None,
    ) -> None:
        m = StepMetrics(step=step)
        if train:
            m.train.update(train)
        if adv:
            adv_part, data_part = split_adv_data_metrics(adv)
            m.adv.update(adv_part)
            if data_part:
                m.data.update(data_part)
        if stripped_rollout:
            parts = partition_stripped_metrics(stripped_rollout)
            for bucket, vals in parts.items():
                if vals:
                    getattr(m, bucket).update(vals)
        if time_metrics:
            m.time.update(time_metrics)
        if eval_metrics:
            m.eval.update(
                {k.removeprefix("eval/"): v for k, v in eval_metrics.items()}
                if any(k.startswith("eval/") for k in eval_metrics)
                else eval_metrics
            )
        pending = self._pending_eval.pop(int(step), None)
        if pending:
            m.eval.update(
                {k.removeprefix("eval/"): v for k, v in pending.items()}
                if any(k.startswith("eval/") for k in pending)
                else pending
            )

        if memory_metrics:
            m.memory.update(memory_metrics)
        elif self._gpu_guard is not None:
            from joyrlinf.observability.metrics import memory_snapshot_to_metrics

            snap = self._gpu_guard.snapshot()
            m.memory.update(memory_snapshot_to_metrics(snap, guard=self._gpu_guard))

        flat = m.flatten()
        if self._logger is not None:
            self._logger.log(flat, step)
            if histograms:
                self._logger.log_histograms(histograms, step)

        if self._perf is not None and time_metrics:
            tp = time_metrics.get("throughput_env_s", time_metrics.get("throughput"))
            self._perf.record_step(
                step,
                time_metrics,
                throughput_env_s=float(tp) if tp is not None else None,
            )

    def close(self) -> str | None:
        if self._perf is None:
            return None
        return self._perf.write_json()
