"""Suppress noisy LIBERO / robosuite / gym / sapien startup messages."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings
from typing import Iterator


def libero_warnings_enabled() -> bool:
    return os.environ.get("JOYRLINF_SUPPRESS_LIBERO_WARNINGS", "1") != "0"


def suppress_libero_startup_noise() -> None:
    """Idempotent: warnings filters + robosuite log level."""
    if not libero_warnings_enabled():
        return
    if getattr(suppress_libero_startup_noise, "_done", False):
        return
    suppress_libero_startup_noise._done = True  # type: ignore[attr-defined]

    # Must be "ignore", not "once" — each LIBERO subprocess would still print once.
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*pkg_resources.*")
    warnings.filterwarnings("ignore", message=r".*libvulkan.*")
    warnings.filterwarnings(
        "ignore",
        message=r".*Gym has been unmaintained.*",
        category=Warning,
    )
    for name in ("robosuite", "robosuite.macros"):
        logging.getLogger(name).setLevel(logging.ERROR)


@contextlib.contextmanager
def quiet_stderr() -> Iterator[None]:
    """Redirect stderr (gym prints its notice via print(..., file=sys.stderr))."""
    if not libero_warnings_enabled():
        yield
        return
    stderr = sys.stderr
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = stderr


def import_libero_env_stack() -> None:
    """Import gym/libero/robosuite stack once with noise suppressed."""
    suppress_libero_startup_noise()
    with quiet_stderr():
        import gym  # noqa: F401
        from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
