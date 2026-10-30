"""LIBERO env package. Heavy deps load lazily via ``get_env_cls`` / ``LiberoEnv``."""

from __future__ import annotations

__all__ = ["LiberoEnv"]


def __getattr__(name: str):
    if name == "LiberoEnv":
        from joyrlinf.envs.libero.libero_env import LiberoEnv

        return LiberoEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
