"""Environment helpers for binding spawned LIBERO renderers to a physical GPU."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator


_EGL_PIN_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "MUJOCO_EGL_DEVICE_ID",
    "CUDA_MPS_PIPE_DIRECTORY",
    "CUDA_MPS_LOG_DIRECTORY",
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
)


def configure_libero_worker_egl() -> None:
    """Bind a spawned renderer to its parent's physical GPU.

    MPS clients see their physical card as logical ``cuda:0``, but EGL selects
    from physical device indices.  The launcher preserves that index in
    ``JOYRLINF_PHYSICAL_DEVICE_ID`` before remapping the CUDA client.
    """
    physical = os.environ.get("JOYRLINF_PHYSICAL_DEVICE_ID")
    if not physical:
        return
    if not physical.isdecimal():
        raise ValueError(
            "JOYRLINF_PHYSICAL_DEVICE_ID must be a numeric GPU index, "
            f"got {physical!r}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = physical
    os.environ["MUJOCO_EGL_DEVICE_ID"] = physical
    # Renderer workers are not MPS compute clients.
    for key in _EGL_PIN_KEYS[2:]:
        os.environ.pop(key, None)


@contextlib.contextmanager
def physical_egl_spawn_environment() -> Iterator[None]:
    """Temporarily configure the parent environment inherited by ``spawn``."""
    physical = os.environ.get("JOYRLINF_PHYSICAL_DEVICE_ID")
    if not physical:
        yield
        return
    previous = {key: os.environ.get(key) for key in _EGL_PIN_KEYS}
    try:
        configure_libero_worker_egl()
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
