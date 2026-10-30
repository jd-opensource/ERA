from enum import Enum


class SupportedEnvType(Enum):
    MANISKILL = "maniskill"
    LIBERO = "libero"
    WANWM = "wan_wm"


def get_env_cls(env_type: str):
    """
    Get environment class based on environment type.

    Args:
        env_type: Type of environment (e.g., "maniskill", "libero", "wan_wm", etc.)

    Returns:
        Environment class corresponding to the environment type.
    """
    env_type = SupportedEnvType(env_type)

    if env_type == SupportedEnvType.MANISKILL:
        from .maniskill_env import ManiSkillEnv

        return ManiSkillEnv
    if env_type == SupportedEnvType.LIBERO:
        from .libero.libero_env import LiberoEnv

        return LiberoEnv
    if env_type == SupportedEnvType.WANWM:
        from .world_model.world_model_wan_env import WanEnv

        return WanEnv
    raise NotImplementedError(f"Environment type {env_type} not implemented")


def __getattr__(name: str):
    if name == "ManiSkillEnv":
        from .maniskill_env import ManiSkillEnv

        return ManiSkillEnv
    if name == "LiberoEnv":
        from .libero.libero_env import LiberoEnv

        return LiberoEnv
    if name == "build_env":
        from .build_env import build_env

        return build_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ManiSkillEnv",
    "LiberoEnv",
    "SupportedEnvType",
    "build_env",
    "get_env_cls",
]
