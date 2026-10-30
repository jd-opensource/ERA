import unittest

import numpy as np

from joyrlinf.observability.metrics import strip_runner_metrics


class TestStripRunnerMetricsRunner(unittest.TestCase):
    """Regression tests for batch metric stripping (formerly on EmbodiedRunner)."""

    def test_empty_batch(self):
        batch = {}
        result = strip_runner_metrics(batch)
        self.assertEqual(result, {})
        self.assertEqual(batch, {})

    def test_no_metrics(self):
        batch = {"other_key": "value", "another_key": 123}
        original_batch = batch.copy()
        result = strip_runner_metrics(batch)
        self.assertEqual(result, {})
        self.assertEqual(batch, original_batch)

    def test_runner_metric_prefix(self):
        batch = {
            "other_key": "value",
            "_runner_metric_reward": np.array([0.5]),
            "_runner_metric_loss": np.array([0.1]),
        }
        result = strip_runner_metrics(batch)
        self.assertEqual(result["reward"], 0.5)
        self.assertEqual(result["loss"], 0.1)
        self.assertNotIn("_runner_metric_reward", batch)

    def test_double_underscore_slash(self):
        batch = {
            "_runner_metric_train__loss": np.array([0.3]),
            "_runner_metric_eval__accuracy": np.array([0.95]),
        }
        result = strip_runner_metrics(batch)
        self.assertEqual(result["train/loss"], 0.3)
        self.assertEqual(result["eval/accuracy"], 0.95)

    def test_legacy_metrics_dict(self):
        batch = {
            "other_data": "value",
            "_runner_metrics": {"accuracy": 0.85, "f1_score": 0.78},
        }
        result = strip_runner_metrics(batch)
        self.assertEqual(result["accuracy"], 0.85)
        self.assertEqual(result["f1_score"], 0.78)
        self.assertNotIn("_runner_metrics", batch)


if __name__ == "__main__":
    unittest.main()
