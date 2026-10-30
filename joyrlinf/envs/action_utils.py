"""Action format conversion between model output and environment input."""

from __future__ import annotations

import numpy as np
import torch

from joyrlinf.envs import SupportedEnvType


def prepare_actions_for_maniskill(
    raw_chunk_actions: np.ndarray | torch.Tensor,
    num_action_chunks: int,
    action_dim: int = 7,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
) -> torch.Tensor:
    if isinstance(raw_chunk_actions, np.ndarray):
        actions_tensor = torch.from_numpy(raw_chunk_actions).float()
    else:
        actions_tensor = raw_chunk_actions.float()

    if policy in ("panda-ee-dpos", "panda-ee-dpose", "panda-qpos"):
        return actions_tensor.cuda()

    if "panda" in policy and policy != "panda_wristcam":
        return actions_tensor.cuda()

    reshaped = actions_tensor.reshape(-1, action_dim).cpu().numpy()
    batch_size = reshaped.shape[0]

    world_vector = reshaped[:, :3] * action_scale
    rot_axangle = reshaped[:, 3:6] * action_scale
    if policy in ("widowx_bridge", "panda_wristcam"):
        gripper = 2.0 * (reshaped[:, 6:7] > 0.5) - 1.0
    else:
        gripper = reshaped[:, 6:7]

    actions = np.concatenate([world_vector, rot_axangle, gripper], axis=1)
    actions = torch.tensor(actions, dtype=torch.float32).cuda()
    return actions.reshape(-1, num_action_chunks, action_dim)


def prepare_actions_for_libero(
    raw_chunk_actions: np.ndarray | torch.Tensor,
    num_action_chunks: int,
    action_dim: int = 7,
    device: str | torch.device = "cuda",
    model_type: str | None = None,
) -> torch.Tensor:
    """LIBERO action conversion aligned with RLinf.

    GR00T/OpenPI already emit actions in LIBERO's expected convention.  Only
    OpenVLA-style policies use a [0, 1] gripper that needs conversion.
    """
    if isinstance(raw_chunk_actions, torch.Tensor):
        arr = raw_chunk_actions.detach().cpu().numpy()
    else:
        arr = np.asarray(raw_chunk_actions, dtype=np.float32)
    arr = arr.reshape(-1, num_action_chunks, action_dim)
    if str(model_type or "").lower() in ("openvla", "openvla_oft", "openvla-oft"):
        arr[..., -1] = 2 * arr[..., -1] - 1
        arr[..., -1] = np.sign(arr[..., -1]) * -1.0
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _resolve_env_type(env_type: str | SupportedEnvType) -> SupportedEnvType:
    if isinstance(env_type, SupportedEnvType):
        return env_type
    return SupportedEnvType(str(env_type).lower())


def prepare_actions(
    raw_chunk_actions,
    env_type: str | SupportedEnvType,
    num_action_chunks: int,
    action_dim: int = 7,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
    device: str | torch.device = "cuda",
    model_type: str | None = None,
    wm_env_type: str | None = None,
    env_ft_type: str | None = None,
    **kwargs,
) -> torch.Tensor:
    """Dispatch action preparation based on env type."""
    env_type = _resolve_env_type(env_type)
    if env_type == SupportedEnvType.MANISKILL:
        return prepare_actions_for_maniskill(
            raw_chunk_actions, num_action_chunks, action_dim, action_scale, policy
        )
    if env_type == SupportedEnvType.LIBERO:
        return prepare_actions_for_libero(
            raw_chunk_actions,
            num_action_chunks,
            action_dim,
            device=device,
            model_type=model_type,
        )
    if env_type == SupportedEnvType.WANWM:
        backend = wm_env_type or env_ft_type or kwargs.get("env_ft_type")
        if backend is None:
            backend = "libero"
        if str(backend).lower() == "libero":
            return prepare_actions_for_libero(
                raw_chunk_actions,
                num_action_chunks,
                action_dim,
                device=device,
                model_type=model_type,
            )
        raise NotImplementedError(f"WAN world model backend {backend!r} not implemented")
    if isinstance(raw_chunk_actions, np.ndarray):
        return torch.from_numpy(raw_chunk_actions).to(device=device)
    return raw_chunk_actions.to(device=device)
