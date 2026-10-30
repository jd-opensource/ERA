# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(x) -> np.ndarray:
    """Tensor/array to host numpy (joyrlinf ManiSkill keeps RGB on GPU)."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def convert_libero_obs_to_gr00t_format(env_obs):
    """Convert env obs to GR00T LeRobot-style keys (LIBERO)."""
    groot_obs = {}
    groot_obs["video.image"] = _to_numpy(env_obs["main_images"].unsqueeze(1))
    groot_obs["video.wrist_image"] = _to_numpy(env_obs["wrist_images"].unsqueeze(1))
    states = env_obs["states"].unsqueeze(1)
    groot_obs["state.x"] = _to_numpy(states[:, :, 0:1])
    groot_obs["state.y"] = _to_numpy(states[:, :, 1:2])
    groot_obs["state.z"] = _to_numpy(states[:, :, 2:3])
    groot_obs["state.roll"] = _to_numpy(states[:, :, 3:4])
    groot_obs["state.pitch"] = _to_numpy(states[:, :, 4:5])
    groot_obs["state.yaw"] = _to_numpy(states[:, :, 5:6])
    groot_obs["state.gripper"] = _to_numpy(states[:, :, 6:])
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_maniskill_obs_to_gr00t_format(env_obs):
    """Convert ManiSkill WidowX obs to GR00T LeRobot-style keys."""
    groot_obs = {}
    env_obs["main_images"] = cut_and_resize_images(
        env_obs["main_images"],
        env_obs["main_images"].shape[-3],
        256,
    )
    groot_obs["video.ego_view"] = _to_numpy(env_obs["main_images"].unsqueeze(1))
    if "state" in env_obs:
        raise NotImplementedError("State from simulation are not unified yet.")
    groot_obs["state.left_arm"] = np.zeros((env_obs["main_images"].shape[0], 1, 7))
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_maniskill_panda_obs_to_gr00t_format(env_obs):
    """Convert ManiSkill Panda obs to GR00T keys (dual camera + 8-dim state)."""
    groot_obs = {}
    main_images = env_obs["main_images"]
    main_images = cut_and_resize_images(main_images, main_images.shape[-3], 256)
    groot_obs["video.image"] = _to_numpy(main_images.unsqueeze(1))

    wrist_images = env_obs.get("wrist_images")
    if wrist_images is None:
        wrist_images = main_images
    else:
        wrist_images = cut_and_resize_images(wrist_images, wrist_images.shape[-3], 256)
    groot_obs["video.wrist_image"] = _to_numpy(wrist_images.unsqueeze(1))

    states = env_obs["states"]
    if torch.is_tensor(states):
        states = states.detach().cpu().numpy()
    states = np.asarray(states)
    if states.ndim == 2:
        states = states[:, None, :]
    if states.shape[-1] < 8:
        pad = np.zeros((*states.shape[:-1], 8 - states.shape[-1]), dtype=states.dtype)
        states = np.concatenate([states, pad], axis=-1)
    elif states.shape[-1] > 8:
        states = states[..., :8]
    groot_obs["state.arm"] = states
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_to_libero_action(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array = normalize_gripper_action(action_array, binarize=True)
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_maniskill_action(
    action_chunk: dict[str, np.array], chunk_size: int = 16
) -> np.ndarray:
    return action_chunk["action.left_arm"][:, :chunk_size]


def convert_to_maniskill_panda_action(
    action_chunk: dict[str, np.array], chunk_size: int = 16
) -> np.ndarray:
    return action_chunk["action.arm"][:, :chunk_size]


def convert_to_isaaclab_stack_cube_action(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array[..., -1] = np.sign(action_array[..., -1])
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


OBS_CONVERSION = {
    "maniskill": convert_maniskill_obs_to_gr00t_format,
    "maniskill_panda": convert_maniskill_panda_obs_to_gr00t_format,
    "libero": convert_libero_obs_to_gr00t_format,
    "isaaclab_stack_cube": convert_libero_obs_to_gr00t_format,
}

ACTION_CONVERSION = {
    "libero": convert_to_libero_action,
    "maniskill": convert_to_maniskill_action,
    "maniskill_panda": convert_to_maniskill_panda_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
}


def cut_and_resize_images(
    images: torch.Tensor, crop_size: int, target_size: int = 256
) -> torch.Tensor:
    """Crop center strip and resize. ManiSkill RGB is uint8; interpolate needs float."""
    return_uint8 = images.dtype == torch.uint8
    if return_uint8:
        images = images.float()
    images_nchw = images.permute(0, 3, 1, 2)
    original_width = images_nchw.shape[-1]
    start = (original_width - crop_size) // 2
    end = start + crop_size
    cropped_tensor = images_nchw[:, :, :, start:end]
    resized_tensor = F.interpolate(
        cropped_tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    out = resized_tensor.permute(0, 2, 3, 1).contiguous()
    if return_uint8:
        out = out.round().clamp(0, 255).to(torch.uint8)
    return out


def normalize_gripper_action(action, binarize=True):
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action
