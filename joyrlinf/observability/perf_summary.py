"""Run-level performance summary export (``perf_summary.json``)."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_ROLLOUT_TIME_KEYS = (
    "rollout_total",
    "phase_a_rollout",
    "rollout",
)
_ACTOR_TIME_KEYS = (
    "actor_train",
    "phase_b_train",
    "train",
)


@dataclass
class PerfSummaryTracker:
    """Accumulate step timings; flush ``perf_summary.json`` after every step."""

    pipeline_mode: str = "sync"
    comm_mode: str = "v1"
    output_dir: str = "."
    flush_every_step: bool = True
    _steps: list[dict[str, Any]] = field(default_factory=list)
    _last_written_step: int = 0
    _completed: bool = False

    @property
    def json_path(self) -> str:
        return os.path.join(self.output_dir, "perf_summary.json")

    def record_step(
        self,
        step: int,
        time_metrics: dict[str, float],
        *,
        throughput_env_s: float | None = None,
    ) -> None:
        rollout_s = _pick(time_metrics, _ROLLOUT_TIME_KEYS)
        actor_s = _pick(time_metrics, _ACTOR_TIME_KEYS)
        total_s = time_metrics.get("total")
        if total_s is None:
            total_s = time_metrics.get("phase_b_total")
        other_s = None
        if total_s is not None and rollout_s is not None and actor_s is not None:
            other_s = max(float(total_s) - float(rollout_s) - float(actor_s), 0.0)

        entry: dict[str, Any] = {
            "step": int(step),
            "rollout_time_s": rollout_s,
            "actor_time_s": actor_s,
            "total_time_s": total_s,
            "other_time_s": other_s,
            "throughput_env_s": throughput_env_s,
        }
        entry.update(
            {f"time/{k}": float(v) for k, v in time_metrics.items()},
        )
        self._steps.append(entry)
        self._last_written_step = int(step)

        if self.flush_every_step:
            self.flush()

    def build_summary(self, *, final: bool = False) -> dict[str, Any]:
        tputs = [
            s["throughput_env_s"]
            for s in self._steps
            if s.get("throughput_env_s") is not None
        ]
        rollout_times = [
            s["rollout_time_s"] for s in self._steps if s.get("rollout_time_s") is not None
        ]
        actor_times = [
            s["actor_time_s"] for s in self._steps if s.get("actor_time_s") is not None
        ]
        total_times = [
            s["total_time_s"] for s in self._steps if s.get("total_time_s") is not None
        ]

        def _mean(vals: list[float]) -> float | None:
            return sum(vals) / len(vals) if vals else None

        def _mean_time_metric(key: str) -> float | None:
            values = [
                float(step[f"time/{key}"])
                for step in self._steps
                if step.get(f"time/{key}") is not None
            ]
            return _mean(values)

        peak_step = None
        peak_tp = None
        if tputs:
            peak_tp = max(tputs)
            for s in self._steps:
                if s.get("throughput_env_s") == peak_tp:
                    peak_step = s.get("step")
                    break

        return {
            "status": "completed" if (final or self._completed) else "in_progress",
            "pipeline_mode": self.pipeline_mode,
            "comm_mode": self.comm_mode,
            "steps_recorded": len(self._steps),
            "last_updated_step": self._last_written_step,
            "peak_throughput_env_s": peak_tp,
            "peak_throughput_step": peak_step,
            "mean_rollout_time_s": _mean(rollout_times),
            "mean_actor_time_s": _mean(actor_times),
            "mean_actor_compute_time_s": _mean_time_metric("actor_compute"),
            "mean_actor_gate_wait_time_s": _mean_time_metric("actor_gate_wait"),
            "mean_actor_other_time_s": _mean_time_metric("actor_train_other"),
            "mean_total_time_s": _mean(total_times),
            "per_step": self._steps,
        }

    def flush(self, path: str | None = None, *, final: bool = False) -> str:
        """Atomically write the current summary (safe if the process is killed)."""
        out_path = path or self.json_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        payload = self.build_summary(final=final)
        tmp_path = f"{out_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
        return out_path

    def write_json(self, path: str | None = None) -> str:
        """Mark the run complete and write the final summary."""
        self._completed = True
        out_path = self.flush(path, final=True)
        logger.info("Wrote perf summary to %s (status=completed)", out_path)
        return out_path


def _pick(metrics: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in metrics:
            return float(metrics[k])
    return None
