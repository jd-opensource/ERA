"""LIBERO observation/action helpers and benchmark overrides."""

from __future__ import annotations

import math
import os
from typing import Union

import numpy as np

import libero.libero.benchmark as benchmark
from libero.libero.benchmark import Benchmark


def get_libero_type() -> str:
    return os.environ.get("LIBERO_TYPE", "standard").lower()


def get_libero_image(obs: dict[str, np.ndarray]) -> np.ndarray:
    img = obs["agentview_image"]
    return img[::-1, ::-1]


def get_libero_wrist_image(
    obs: dict[str, np.ndarray],
    resize_size: Union[int, tuple[int, int]] = 224,
) -> np.ndarray:
    del resize_size
    img = obs["robot0_eye_in_hand_image"]
    return img[::-1, ::-1]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_benchmark_overridden(benchmark_name: str) -> type[Benchmark]:
    """Return benchmark class; synthesize libero_130 when missing."""
    name = str(benchmark_name).lower()
    if name != "libero_130":
        return benchmark.get_benchmark(benchmark_name)

    libero_cls = benchmark.BENCHMARK_MAPPING.get("libero_130")
    if libero_cls is not None:
        return libero_cls

    aggregated_task_map: dict = {}
    suites = getattr(benchmark, "libero_suites", [])
    for suite_name in suites:
        suite_map = benchmark.task_maps.get(suite_name, {})
        for task_name, task in suite_map.items():
            if task_name not in aggregated_task_map:
                aggregated_task_map[task_name] = task

    class LIBERO_ALL(Benchmark):
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name = "libero_130"
            self._make_benchmark()

        def _make_benchmark(self):
            tasks = list(aggregated_task_map.values())
            self.tasks = tasks
            self.n_tasks = len(self.tasks)

    benchmark.BENCHMARK_MAPPING["libero_130"] = LIBERO_ALL
    return LIBERO_ALL
