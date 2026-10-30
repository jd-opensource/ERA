"""GPU-side OpenPI observation preprocessing (batch path).

This module is the target API for moving image/state prep off the per-sample
numpy ``input_transform`` loop. Implementations land step-by-step; callers
should gate on ``OpenPi0Config.use_gpu_input_transform``.

Step inventory (unit-tested independently):
  1. ``resize_with_pad``          – aspect-preserving resize + constant pad
  2. ``images_uint8_to_float``    – uint8 [0,255] -> float32 [-1, 1]
  3. ``pack_maniskill_images``    – base RGB + zero wrists + masks
  4. ``pack_libero_images``       – base + wrist RGB + masks
  5. ``normalize_state``          – z-score / quantile using norm stats
  6. ``pad_state``                – zero-pad state to model action_dim
  7. ``PromptTokenCache``         – episode-stable prompt tokenization cache
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F


def resize_with_pad(
    images: torch.Tensor,
    height: int,
    width: int,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize with aspect ratio preserved, then pad to (height, width).

    Args:
        images: ``[H,W,C]`` or ``[B,H,W,C]`` uint8 or float32.
            Float inputs are assumed in ``[-1, 1]``.
        height / width: target spatial size (OpenPI uses 224x224).
        mode: ``bilinear`` (default) or ``nearest``.

    Returns:
        Same rank/layout/dtype as input, spatial size ``(height, width)``.
        uint8 pads with ``0``; float32 pads with ``-1.0``.
        Stays on the input device (CPU or CUDA).
    """
    if images.ndim not in (3, 4):
        raise ValueError(f"expected 3D/4D HWC, got shape={tuple(images.shape)}")

    is_3d = images.ndim == 3
    img = images.unsqueeze(0) if is_3d else images

    # Fast path: already at target size — return original tensor identity.
    if img.shape[1] == height and img.shape[2] == width:
        return images

    img_t = img.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW
    cur_h, cur_w = img_t.shape[2], img_t.shape[3]
    ratio = max(cur_w / width, cur_h / height)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)

    interp_mode = "nearest" if "nearest" in mode.lower() else "bilinear"
    # Cast uint8 -> float for interpolate (deterministic across devices).
    resized = F.interpolate(
        img_t.float() if img_t.dtype == torch.uint8 else img_t,
        size=(resized_h, resized_w),
        mode=interp_mode,
        align_corners=False if interp_mode == "bilinear" else None,
    )
    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
        pad_value = 0
    elif images.dtype == torch.float32:
        resized = resized.clamp(-1.0, 1.0)
        pad_value = -1.0
    else:
        raise ValueError(f"unsupported dtype: {images.dtype}")

    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    resized = F.pad(
        resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=pad_value
    )
    out = resized.permute(0, 2, 3, 1).contiguous()
    return out.squeeze(0) if is_3d else out


def images_uint8_to_float(images: torch.Tensor) -> torch.Tensor:
    """Convert uint8 images in [0, 255] to float32 in [-1, 1].

    Formula matches ``Observation.from_dict``::

        x.to(float32) / 255.0 * 2.0 - 1.0

    Layout is unchanged (HWC or BHWC). Does not permute to CHW.
    """
    if images.dtype != torch.uint8:
        raise ValueError(f"expected uint8 images, got dtype={images.dtype}")
    return images.to(torch.float32) / 255.0 * 2.0 - 1.0


def _masked_image_fill(like: torch.Tensor) -> torch.Tensor:
    """Fill for masked-out cameras.

    CPU path uses uint8 zeros; ``Observation.from_dict`` maps those to ``-1.0``.
    When packing already-float images, use ``-1.0`` so embeddings match.
    """
    if like.dtype == torch.uint8:
        return torch.zeros_like(like)
    return torch.full_like(like, -1.0)


def pack_maniskill_images(
    base_rgb: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Pack ManiSkill base camera into OpenPI image + mask dicts.

    Args:
        base_rgb: ``[B,H,W,3]`` (typically float32 in [-1,1] after convert).

    Returns:
        ``(image, image_mask)`` where image has keys
        ``base_0_rgb``, ``left_wrist_0_rgb``, ``right_wrist_0_rgb`` and
        masks are ``True/False`` matching ``ManiSkillInputs`` (only base valid).
    """
    if base_rgb.ndim != 4:
        raise ValueError(
            f"expected base_rgb with shape [B,H,W,C], got {tuple(base_rgb.shape)}"
        )
    # Match ManiSkillInputs np.zeros_like(uint8) → Observation.from_dict → -1.0.
    fill = _masked_image_fill(base_rgb)
    bsz = base_rgb.shape[0]
    device = base_rgb.device
    image = {
        "base_0_rgb": base_rgb,
        "left_wrist_0_rgb": fill,
        "right_wrist_0_rgb": fill.clone(),
    }
    image_mask = {
        "base_0_rgb": torch.ones(bsz, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.zeros(bsz, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.zeros(bsz, dtype=torch.bool, device=device),
    }
    return image, image_mask


def pack_libero_images(
    base_rgb: torch.Tensor,
    wrist_rgb: torch.Tensor,
    *,
    mask_right_wrist: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Pack Libero base + wrist cameras into OpenPI image + mask dicts.

    Args:
        base_rgb / wrist_rgb: ``[B,H,W,3]``.
        mask_right_wrist: if True, right wrist mask is True (PI0_FAST);
            otherwise False (pi0 / pi05 default in LiberoInputs).
    """
    if base_rgb.ndim != 4 or wrist_rgb.ndim != 4:
        raise ValueError(
            "expected base_rgb/wrist_rgb with shape [B,H,W,C], "
            f"got {tuple(base_rgb.shape)} / {tuple(wrist_rgb.shape)}"
        )
    if base_rgb.shape != wrist_rgb.shape:
        raise ValueError(
            f"base_rgb and wrist_rgb shape mismatch: "
            f"{tuple(base_rgb.shape)} vs {tuple(wrist_rgb.shape)}"
        )
    zeros = _masked_image_fill(base_rgb)
    bsz = base_rgb.shape[0]
    device = base_rgb.device
    image = {
        "base_0_rgb": base_rgb,
        "left_wrist_0_rgb": wrist_rgb,
        "right_wrist_0_rgb": zeros,
    }
    right_mask = (
        torch.ones(bsz, dtype=torch.bool, device=device)
        if mask_right_wrist
        else torch.zeros(bsz, dtype=torch.bool, device=device)
    )
    image_mask = {
        "base_0_rgb": torch.ones(bsz, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(bsz, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": right_mask,
    }
    return image, image_mask


def normalize_state(
    state: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor | None = None,
    *,
    q01: torch.Tensor | None = None,
    q99: torch.Tensor | None = None,
    use_quantiles: bool = False,
) -> torch.Tensor:
    """Normalize robot state with OpenPI norm stats.

    z-score (default)::

        (x - mean[..., :D]) / (std[..., :D] + 1e-6)

    quantile::

        (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
    """
    dim = state.shape[-1]
    if use_quantiles:
        if q01 is None or q99 is None:
            raise ValueError("q01 and q99 are required when use_quantiles=True")
        q01_ = q01[..., :dim]
        q99_ = q99[..., :dim]
        return (state - q01_) / (q99_ - q01_ + 1e-6) * 2.0 - 1.0
    if std is None:
        raise ValueError("std is required when use_quantiles=False")
    mean_ = mean[..., :dim]
    std_ = std[..., :dim]
    return (state - mean_) / (std_ + 1e-6)


def pad_state(state: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Zero-pad state on the last dim to ``target_dim`` (no-op if already >=)."""
    cur = state.shape[-1]
    if cur >= target_dim:
        return state
    return F.pad(state, (0, target_dim - cur), mode="constant", value=0.0)



def unnormalize_actions(
    actions: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    q01: torch.Tensor | None = None,
    q99: torch.Tensor | None = None,
    use_quantiles: bool = False,
) -> torch.Tensor:
    """Unnormalize model actions with OpenPI norm stats (GPU batch path).

    Mirrors ``openpi.transforms.Unnormalize`` exactly, including the
    stats-narrower-than-actions semantics (model action dim is padded, stats
    cover only the env action dims):

    quantile::

        x[..., :D] -> (x + 1) / 2 * (q99 - q01 + 1e-6) + q01 ; rest passthrough

    z-score (mean padded with 0, std padded with 1 to the action dim)::

        x * (std + 1e-6) + mean
    """
    dim = actions.shape[-1]
    if use_quantiles:
        if q01 is None or q99 is None:
            raise ValueError("q01 and q99 are required when use_quantiles=True")
        d = q01.shape[-1]
        if d < dim:
            head = (actions[..., :d] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            return torch.cat([head, actions[..., d:]], dim=-1)
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    if mean is None or std is None:
        raise ValueError("mean and std are required when use_quantiles=False")
    mean_p = pad_state(mean, dim)
    std_p = F.pad(std, (0, dim - std.shape[-1]), value=1.0) if std.shape[-1] < dim else std
    return actions * (std_p + 1e-6) + mean_p


class PromptTokenCache:
    """Cache tokenized prompts per env index across an episode.

    For pi05 with ``discrete_state_input=False``, tokens depend only on the
    prompt string and can be reused until reset / prompt change.
    """

    def __init__(self, tokenize_fn: Any):
        """
        Args:
            tokenize_fn: ``(prompt: str) -> tuple[np.ndarray|Tensor, np.ndarray|Tensor]``
                returning ``(tokens, mask)``.
        """
        self._tokenize_fn = tokenize_fn
        # env_id -> (prompt, tokens, mask)
        self._cache: dict[int, tuple[str, torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _as_tensor(x: Any) -> torch.Tensor:
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x)

    def get(
        self,
        env_ids: Sequence[int],
        prompts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return batched tokens/masks, computing only on cache miss."""
        if len(env_ids) != len(prompts):
            raise ValueError(
                f"env_ids and prompts length mismatch: {len(env_ids)} vs {len(prompts)}"
            )
        tokens_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor] = []
        for env_id, prompt in zip(env_ids, prompts):
            cached = self._cache.get(env_id)
            if cached is not None and cached[0] == prompt:
                tokens, mask = cached[1], cached[2]
            else:
                raw_tokens, raw_mask = self._tokenize_fn(prompt)
                tokens = self._as_tensor(raw_tokens)
                mask = self._as_tensor(raw_mask)
                self._cache[int(env_id)] = (prompt, tokens, mask)
            tokens_list.append(tokens)
            masks_list.append(mask)
        return torch.stack(tokens_list, dim=0), torch.stack(masks_list, dim=0)

    def invalidate(self, env_ids: Sequence[int] | None = None) -> None:
        """Drop cache entries. ``None`` clears all."""
        if env_ids is None:
            self._cache.clear()
            return
        for env_id in env_ids:
            self._cache.pop(int(env_id), None)


def preprocess_obs_batch(
    *,
    base_images: torch.Tensor,
    state: torch.Tensor,
    prompts: Sequence[str],
    image_size: tuple[int, int] = (224, 224),
    state_mean: torch.Tensor | None = None,
    state_std: torch.Tensor | None = None,
    state_q01: torch.Tensor | None = None,
    state_q99: torch.Tensor | None = None,
    use_quantile_norm: bool = False,
    action_dim: int = 32,
    wrist_images: torch.Tensor | None = None,
    pack_mode: str = "maniskill",
    token_cache: PromptTokenCache | None = None,
    tokenize_fn: Any | None = None,
    env_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """End-to-end GPU preprocess composing the steps above.

    Expects ``base_images`` as uint8 ``[B,H,W,3]`` on the target device.
    """
    if base_images.ndim != 4 or base_images.dtype != torch.uint8:
        raise ValueError(
            "base_images must be uint8 [B,H,W,3], "
            f"got dtype={base_images.dtype} shape={tuple(base_images.shape)}"
        )
    if len(prompts) != base_images.shape[0]:
        raise ValueError(
            f"prompts length {len(prompts)} != batch size {base_images.shape[0]}"
        )
    if token_cache is None and tokenize_fn is None:
        raise ValueError("provide tokenize_fn and/or token_cache")

    height, width = image_size
    device = base_images.device

    # 1-2. resize + uint8 -> float[-1, 1]
    base_f = images_uint8_to_float(resize_with_pad(base_images, height, width))

    # 3. pack cameras
    mode = pack_mode.lower()
    if mode == "maniskill":
        image, image_mask = pack_maniskill_images(base_f)
    elif mode == "libero":
        if wrist_images is None:
            raise ValueError("pack_mode='libero' requires wrist_images")
        wrist_f = images_uint8_to_float(
            resize_with_pad(wrist_images, height, width)
        )
        image, image_mask = pack_libero_images(base_f, wrist_f)
    else:
        raise ValueError(f"unsupported pack_mode={pack_mode!r}")

    # 4-5. normalize + pad state
    state_out = state
    if use_quantile_norm:
        if state_q01 is None or state_q99 is None:
            raise ValueError("quantile norm requires state_q01 and state_q99")
        state_out = normalize_state(
            state_out,
            mean=state_mean if state_mean is not None else state_q01,
            use_quantiles=True,
            q01=state_q01,
            q99=state_q99,
        )
    elif state_mean is not None:
        state_out = normalize_state(state_out, state_mean, state_std)
    state_out = pad_state(state_out, action_dim)

    # 6. tokenize (CPU) then move tokens to image device
    bsz = base_images.shape[0]
    ids = list(env_ids) if env_ids is not None else list(range(bsz))
    if len(ids) != bsz:
        raise ValueError(f"env_ids length {len(ids)} != batch size {bsz}")
    if token_cache is not None:
        tokens, token_masks = token_cache.get(ids, prompts)
    else:
        assert tokenize_fn is not None
        tokens, token_masks = PromptTokenCache(tokenize_fn).get(ids, prompts)
    tokens = tokens.to(device=device)
    token_masks = token_masks.to(device=device)

    return {
        "image": image,
        "image_mask": image_mask,
        "state": state_out,
        "tokenized_prompt": tokens,
        "tokenized_prompt_mask": token_masks,
    }
