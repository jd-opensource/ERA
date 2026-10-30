"""Tests for physical EGL binding used by LIBERO MPS workers."""

from __future__ import annotations

import os

import pytest

from joyrlinf.envs.libero.egl_env import (
    configure_libero_worker_egl,
    physical_egl_spawn_environment,
)


def test_non_mps_environment_is_unchanged(monkeypatch):
    monkeypatch.delenv("JOYRLINF_PHYSICAL_DEVICE_ID", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    configure_libero_worker_egl()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_spawn_environment_uses_physical_gpu_and_restores_parent(monkeypatch):
    monkeypatch.setenv("JOYRLINF_PHYSICAL_DEVICE_ID", "5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/mps-pipe")
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", "/tmp/mps-log")
    monkeypatch.setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "50")

    with physical_egl_spawn_environment():
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "5"
        assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
        assert "CUDA_MPS_LOG_DIRECTORY" not in os.environ
        assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in os.environ

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/mps-pipe"
    assert os.environ["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/mps-log"
    assert os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "50"


def test_physical_gpu_id_must_be_numeric(monkeypatch):
    monkeypatch.setenv("JOYRLINF_PHYSICAL_DEVICE_ID", "GPU-deadbeef")
    with pytest.raises(ValueError, match="numeric GPU index"):
        configure_libero_worker_egl()
