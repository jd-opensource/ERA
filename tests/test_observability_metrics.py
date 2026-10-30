"""Tests for P0 observability metrics."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from dataclasses import dataclass

from joyrlinf.observability.emitter import MetricsEmitter
from joyrlinf.observability.metrics import (
    StepMetrics,
    memory_snapshot_to_metrics,
    partition_stripped_metrics,
    split_adv_data_metrics,
    strip_runner_metrics,
)
from joyrlinf.observability.perf_summary import PerfSummaryTracker
from joyrlinf.observability.scalar_transport import (
    pack_rollout_scalars,
    unpack_rollout_scalars,
)


@dataclass
class _Snap:
    total_mib: float
    pt_allocated_mib: float
    pt_reserved_mib: float
    non_pt_used_mib: float
    free_mib: float

    @property
    def pt_fragmented_mib(self) -> float:
        return self.pt_reserved_mib - self.pt_allocated_mib


class TestStripRunnerMetrics(unittest.TestCase):
    def test_legacy_train_success_rate_normalized(self):
        batch = {
            "_runner_metric_train__success_rate": np.array([[0.25]]),
        }
        out = strip_runner_metrics(batch)
        self.assertEqual(out["rollout/success_rate"], 0.25)
        self.assertNotIn("_runner_metric_train__success_rate", batch)

    def test_time_metrics_partition(self):
        batch = {
            "_runner_metric_time__rollout_core": np.array([12.5]),
            "_runner_metric_eval__success_rate": np.array([0.9]),
        }
        stripped = strip_runner_metrics(batch)
        parts = partition_stripped_metrics(stripped)
        self.assertAlmostEqual(parts["time"]["rollout_core"], 12.5)
        self.assertAlmostEqual(parts["eval"]["success_rate"], 0.9)


class TestStepMetrics(unittest.TestCase):
    def test_flatten_buckets(self):
        m = StepMetrics(
            step=1,
            train={"policy_loss": 0.1},
            rollout={"success_rate": 0.5},
            time={"total": 100.0},
        )
        flat = m.flatten()
        self.assertEqual(flat["train/policy_loss"], 0.1)
        self.assertEqual(flat["rollout/success_rate"], 0.5)
        self.assertEqual(flat["time/total"], 100.0)


class TestPerfSummaryTracker(unittest.TestCase):
    def test_flush_after_each_record_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = PerfSummaryTracker(
                pipeline_mode="sync", comm_mode="v2", output_dir=tmp,
            )
            tracker.record_step(
                1,
                {"rollout_total": 10.0, "actor_train": 5.0, "total": 16.0, "throughput": 100.0},
                throughput_env_s=100.0,
            )
            path = tracker.json_path
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["status"], "in_progress")
            self.assertEqual(data["steps_recorded"], 1)
            self.assertEqual(data["last_updated_step"], 1)

    def test_write_json_peak(self):
        tracker = PerfSummaryTracker(pipeline_mode="sync", comm_mode="v2")
        tracker.record_step(
            1,
            {"rollout_total": 10.0, "actor_train": 5.0, "total": 16.0, "throughput": 100.0},
            throughput_env_s=100.0,
        )
        tracker.record_step(
            2,
            {"rollout_total": 12.0, "actor_train": 4.0, "total": 17.0, "throughput": 120.0},
            throughput_env_s=120.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tracker.output_dir = tmp
            path = tracker.write_json()
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["peak_throughput_env_s"], 120.0)
        self.assertEqual(data["peak_throughput_step"], 2)
        self.assertAlmostEqual(data["mean_rollout_time_s"], 11.0)

    def test_mps_actor_breakdown_is_preserved(self):
        tracker = PerfSummaryTracker(pipeline_mode="async", comm_mode="mps")
        tracker.record_step(
            2,
            {
                "actor_train": 40.0,
                "actor_train_wall": 40.0,
                "actor_compute": 15.0,
                "actor_gate_wait": 20.0,
                "actor_train_other": 5.0,
                "total": 48.0,
            },
        )
        summary = tracker.build_summary()
        entry = summary["per_step"][0]
        self.assertEqual(entry["actor_time_s"], 40.0)
        self.assertEqual(entry["time/actor_compute"], 15.0)
        self.assertEqual(entry["time/actor_gate_wait"], 20.0)
        self.assertEqual(entry["time/actor_train_other"], 5.0)
        self.assertEqual(summary["mean_actor_compute_time_s"], 15.0)
        self.assertEqual(summary["mean_actor_gate_wait_time_s"], 20.0)
        self.assertEqual(summary["mean_actor_other_time_s"], 5.0)


class TestMetricsEmitter(unittest.TestCase):
    def test_emit_writes_perf_summary(self):
        logged: list[tuple[dict, int]] = []

        class _FakeLogger:
            def log(self, metrics, step):
                logged.append((metrics, step))

        tracker = PerfSummaryTracker()
        emitter = MetricsEmitter(_FakeLogger(), tracker)
        emitter.emit_step(
            3,
            train={"policy_loss": 0.2},
            stripped_rollout={"rollout/success_rate": 0.4, "time/rollout_core": 9.0},
            time_metrics={"total": 20.0, "actor_train": 6.0, "throughput_env_s": 200.0},
            eval_metrics={"success_rate": 0.8},
        )
        self.assertTrue(any("train/policy_loss" in m for m, _ in logged))
        self.assertTrue(any("rollout/success_rate" in m for m, _ in logged))
        self.assertTrue(any("eval/success_rate" in m for m, _ in logged))
        with tempfile.TemporaryDirectory() as tmp:
            tracker.output_dir = tmp
            path = emitter.close()
            assert path is not None
            self.assertTrue(os.path.isfile(path))


class TestSplitAdvDataMetrics(unittest.TestCase):
    def test_grpo_diagnostics_to_data(self):
        adv, data = split_adv_data_metrics({
            "adv_mean": 0.1,
            "groups_with_var_pct": 42.0,
            "reward_nonzero_pct": 5.0,
        })
        self.assertIn("adv_mean", adv)
        self.assertNotIn("groups_with_var_pct", adv)
        self.assertEqual(data["groups_with_var_pct"], 42.0)


class TestScalarTransport(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        metrics = {
            "rollout/success_rate": 0.25,
            "eval/success_rate": 0.9,
            "time/rollout_send": 1.5,
        }
        payload = pack_rollout_scalars(metrics)
        out = unpack_rollout_scalars(payload)
        self.assertAlmostEqual(out["rollout/success_rate"], 0.25)
        self.assertAlmostEqual(out["eval/success_rate"], 0.9)
        self.assertAlmostEqual(out["time/rollout_send"], 1.5)


class TestMetricsEmitterDataBucket(unittest.TestCase):
    def test_emit_splits_data_bucket(self):
        logged: list[tuple[dict, int]] = []

        class _FakeLogger:
            def log(self, metrics, step):
                logged.append((metrics, step))

        emitter = MetricsEmitter(_FakeLogger(), None)
        emitter.emit_step(
            1,
            adv={"adv_mean": 0.0, "groups_with_var_pct": 50.0},
        )
        flat = logged[0][0]
        self.assertIn("adv/adv_mean", flat)
        self.assertIn("data/groups_with_var_pct", flat)
        self.assertNotIn("adv/groups_with_var_pct", flat)


class TestMemorySnapshotMetrics(unittest.TestCase):
    def test_to_metrics(self):
        snap = _Snap(
            total_mib=80000,
            pt_allocated_mib=1000,
            pt_reserved_mib=2000,
            non_pt_used_mib=3000,
            free_mib=4000,
        )

        class _Guard:
            critical_event_count = 2
            safety_warning_count = 5

        m = memory_snapshot_to_metrics(snap, guard=_Guard())
        self.assertEqual(m["free_mib"], 4000.0)
        self.assertEqual(m["critical_events"], 2.0)
        self.assertEqual(m["safety_warnings"], 5.0)


class TestRolloutScalarMemoryPeaks(unittest.TestCase):
    def test_pack_unpack_preserves_rollout_peaks(self):
        metrics = {
            "rollout/success_rate": 0.5,
            "time/rollout_send": 1.25,
            "memory/rollout_pt_peak_alloc_mib": 12345.0,
            "memory/rollout_pt_peak_reserved_mib": 13000.0,
            "memory/rollout_free_mib": 2000.0,
        }
        payload = pack_rollout_scalars(metrics)
        out = unpack_rollout_scalars(payload)
        self.assertAlmostEqual(out["memory/rollout_pt_peak_alloc_mib"], 12345.0)
        self.assertAlmostEqual(out["memory/rollout_pt_peak_reserved_mib"], 13000.0)
        self.assertAlmostEqual(out["memory/rollout_free_mib"], 2000.0)
        self.assertAlmostEqual(out["rollout/success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
