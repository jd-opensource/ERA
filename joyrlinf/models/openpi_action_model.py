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

import logging
import math
import random
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from torch.utils._pytree import tree_map

from joyrlinf.models.modules.explore_noise_net import ExploreNoiseNet
from joyrlinf.models.modules.value_head import ValueHead
from joyrlinf.models.gqa_attention import resolve_attn_implementation

logger = logging.getLogger(__name__)


def _to_numpy(x):
    """Convert tensor to numpy, handling BFloat16/Float16 conversion."""
    if torch.is_tensor(x):
        x_cpu = x.detach().cpu()
        if x_cpu.dtype in (torch.bfloat16, torch.float16):
            x_cpu = x_cpu.float()
        return np.asarray(x_cpu)
    return x


def _pack_prefix_kv(past_key_values) -> torch.Tensor:
    """Pack prefix KV into one dense tensor [B, L, 2, S, D] for batch plumbing.

    A single tensor flows through the existing buffer stack / pin / flatten /
    microbatch-slice machinery without nested-structure support. Assumes MQA
    (num_kv_heads == 1, Gemma-2B) and squeezes the head dim.
    """
    if hasattr(past_key_values, "key_cache"):
        layers = list(zip(past_key_values.key_cache, past_key_values.value_cache))
    else:
        layers = [(kv[0], kv[1]) for kv in past_key_values]
    per_layer = []
    for k, v in layers:
        if k.shape[1] != 1:
            raise ValueError(
                f"prefix KV pack expects MQA (1 kv head), got {k.shape[1]}"
            )
        # [B, 1, S, D] -> [B, 2, S, D]
        per_layer.append(torch.stack([k[:, 0], v[:, 0]], dim=1))
    return torch.stack(per_layer, dim=1).detach()  # [B, L, 2, S, D]


def _unpack_prefix_kv(packed: torch.Tensor):
    """Rebuild an HF DynamicCache from a packed [B, L, 2, S, D] tensor.

    A bare list of (K, V) tuples is rejected by GemmaModel.forward (it calls
    past_key_values.get_seq_length()), so a Cache object is required.
    """
    from transformers.cache_utils import DynamicCache

    if packed.dim() != 5 or packed.shape[2] != 2:
        raise ValueError(
            f"packed prefix KV must be [B, L, 2, S, D], got {tuple(packed.shape)}"
        )
    cache = DynamicCache()
    for layer_idx in range(packed.shape[1]):
        k = packed[:, layer_idx, 0].unsqueeze(1)  # [B, 1, S, D]
        v = packed[:, layer_idx, 1].unsqueeze(1)
        cache.update(k, v, layer_idx)
    return cache


def _detach_kv_tree(obj):
    """Detach every tensor in a past_key_values structure.

    Supports plain tensors, (nested) list/tuple layouts, and HF
    DynamicCache-like objects exposing key_cache/value_cache. Injected
    rollout KV must never carry autograd history into the actor graph.
    """
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, list):
        return [_detach_kv_tree(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_kv_tree(x) for x in obj)
    if hasattr(obj, "key_cache") and hasattr(obj, "value_cache"):
        obj.key_cache = [_detach_kv_tree(k) for k in obj.key_cache]
        obj.value_cache = [_detach_kv_tree(v) for v in obj.value_cache]
        return obj
    return obj


class BasePolicy(torch.nn.Module):
    """Minimal base policy stub (replaces rlinf.models.embodiment.base_policy)."""

    def forward(self, forward_type="default_forward", **kwargs):
        if forward_type == "default_forward":
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(self, **kwargs):
        raise NotImplementedError


@dataclass(frozen=True)
class OpenPi0Config(Pi0Config):
    # config for rl
    config_name: str = "pi0_libero"  # pi0_libero, pi05_libero, pi0_maniskill, pi05_maniskill, pi0_metaworld, pi05_metaworld
    num_images_in_input: int = 2  # number of images in input
    noise_method: str = "flow_sde"  # flow_sde, flow_noise, flow_cps
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    # training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_after_vlm: bool = False  # value after vlm, pi05 mode
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token
    # input transform parallelization.
    # Default 0 = serial. workers>0 shares one TokenizePrompt/SentencePiece
    # across threads (not thread-safe); only for benches. Prefer
    # use_gpu_input_transform for speed.
    input_transform_num_workers: int = 0
    # GPU batch preprocess (bypass per-sample numpy input_transform)
    use_gpu_input_transform: bool = False
    cache_tokenized_prompt: bool = True  # episode-stable prompt token cache
    # Reuse rollout prefix KV in actor training (skip prefix recompute).
    # Requires train_expert_only and a frozen VLM.
    reuse_rollout_prefix_kv: bool = False
    # How reused prefix_kv reaches the actor GPU during run_training:
    #   bulk_gpu   — one H2D of the full tensor, then GPU shuffle (fast, high VRAM)
    #   microbatch — stay on (pinned) CPU; per-microbatch index_select + H2D
    # (microbatch transfer wiring lands in a later migration phase.)
    prefix_kv_transfer: str = "microbatch"
    # Cross-process KV handoff for colocated async MPS:
    #   off       — Gloo-send full prefix_kv (multi-copy host peak)
    #   host_shm  — double-buffered POSIX shm; Gloo only ships small tensors
    prefix_kv_ipc: str = "off"
    # Overlap next micro-batch H2D with current fwd/bwd (later phase; default off
    # on colocated async to protect shared-GPU VRAM).
    microbatch_prefetch: bool = False
    # When to D2H rollout buffer tensors (esp. prefix_kv / forward_inputs):
    #   end       — keep on GPU for the full epoch; stack then D2H (high VRAM)
    #   per_chunk — async D2H each chunk (later phase: PrefixKvHostPool)
    rollout_buffer_d2h: str = "end"
    # Physically strip the VLM from the actor (training) instance. Requires
    # reuse_rollout_prefix_kv. Rollout always keeps the full model.
    expert_only_actor: bool = False
    # GPU batch output transform: unnormalize + slice on GPU, return CUDA
    # actions from predict (drops the per-sample numpy loop + predict-exit D2H)
    use_gpu_output_transform: bool = False
    # Same-GPU MPS: mutex predict ⊥ train; only cached chunk_step may overlap
    # actor training. "none" | "per_chunk". See joyrlinf.core.gpu_phase_gate.
    gpu_phase_gate: str = "none"
    # Actor-side grant window: "microbatch" (predict between MBs) or
    # "training" (hold mutex for entire run_training after first predict).
    phase_gate_granularity: str = "microbatch"
    # Attention kernel for the Gemma LM + expert: "eager" | "gqa_eager".
    # "gqa_eager" runs the same eager math against unexpanded K/V (no
    # repeat_kv copies saved for backward): bit-exact outputs, cuts the
    # dominant training activation (~37GiB/64-sample microbatch of expanded
    # prefix K/V). Fused SDPA is not offered: head_dim=256 + arbitrary mask
    # has no usable fused kernel on torch 2.6.
    attn_implementation: str = "eager"


class OpenPi0ForRLActionPrediction(BasePolicy, PI0Pytorch):
    """
    Pi0 model for reinforcement learning action prediction.
    """

    config: OpenPi0Config

    @property
    def _no_split_modules(self) -> list[str]:
        # Currently, PaliGemmaForConditionalGeneration only support DDP, as many of it's modules are called without forward
        return [
            "PaliGemmaForConditionalGeneration",
            "GemmaDecoderLayer",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
            "GemmaForCausalLM",
            "GemmaRotaryEmbedding",
        ]

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        PI0Pytorch.__init__(self, config)
        self.sample_actions = sample_actions_func
        self.global_step = 0
        # assert
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )
        # "eager" or the registered "gqa_eager" key; applied at every
        # paligemma/expert forward call site below.
        self._attn_impl = resolve_attn_implementation(
            getattr(self.config, "attn_implementation", "eager")
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            if self.config.config_name == "pi05_maniskill":
                value_head_hidden_sizes = (1024, 512, 256)
            else:
                value_head_hidden_sizes = (512, 256, 128)
            value_head_activation = "relu"
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,
                activation=value_head_activation,
                bias_last=True,
            )
            self.value_head = self.value_head.to(
                dtype=self.action_out_proj.weight.dtype
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        # noise head for flow-noise
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )
            self.noise_head = self.noise_head.to(
                dtype=self.action_out_proj.weight.dtype
            )
        # Populated in setup_wrappers when use_gpu_input_transform is enabled.
        self._gpu_tokenize_fn = None
        self._gpu_token_cache = None
        self._gpu_state_mean = None
        self._gpu_state_std = None
        self._gpu_state_q01 = None
        self._gpu_state_q99 = None
        self._gpu_use_quantiles = False
        self._gpu_action_mean = None
        self._gpu_action_std = None
        self._gpu_action_q01 = None
        self._gpu_action_q99 = None
        self._gpu_output_ok = False
        self._gpu_output_use_quantiles = False
        self._gpu_output_slice_dim = 7
        self._gpu_action_dim = int(getattr(config, "action_dim", 32))
        self._gpu_pack_mode = (
            "libero" if "libero" in str(config.config_name).lower() else "maniskill"
        )
        self._gpu_discrete_state_input = bool(
            getattr(config, "discrete_state_input", False)
        )

    def set_global_step(self, global_step):
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._init_gpu_preprocess(transforms)
        self._init_gpu_output_transform(output_transforms)

    def _init_gpu_preprocess(
        self, transforms: Sequence[_transforms.DataTransformFn]
    ) -> None:
        """Extract tokenizer / norm stats for the GPU preprocess path."""
        from joyrlinf.models.openpi_gpu_preprocess import PromptTokenCache

        tokenize_tf = None
        normalize_tf = None
        pad_tf = None
        for tf in transforms:
            if isinstance(tf, _transforms.TokenizePrompt):
                tokenize_tf = tf
            elif isinstance(tf, _transforms.Normalize):
                normalize_tf = tf
            elif isinstance(tf, _transforms.PadStatesAndActions):
                pad_tf = tf

        if pad_tf is not None:
            self._gpu_action_dim = int(pad_tf.model_action_dim)

        if normalize_tf is not None and normalize_tf.norm_stats is not None:
            self._gpu_use_quantiles = bool(normalize_tf.use_quantiles)
            state_stats = None
            if isinstance(normalize_tf.norm_stats, dict):
                state_stats = normalize_tf.norm_stats.get("state")
            if state_stats is not None:
                self._gpu_state_mean = torch.as_tensor(
                    np.asarray(state_stats.mean), dtype=torch.float32
                )
                self._gpu_state_std = torch.as_tensor(
                    np.asarray(state_stats.std), dtype=torch.float32
                )
                if state_stats.q01 is not None and state_stats.q99 is not None:
                    self._gpu_state_q01 = torch.as_tensor(
                        np.asarray(state_stats.q01), dtype=torch.float32
                    )
                    self._gpu_state_q99 = torch.as_tensor(
                        np.asarray(state_stats.q99), dtype=torch.float32
                    )
            action_stats = None
            if isinstance(normalize_tf.norm_stats, dict):
                action_stats = normalize_tf.norm_stats.get("actions")
            if action_stats is not None:
                self._gpu_action_mean = torch.as_tensor(
                    np.asarray(action_stats.mean), dtype=torch.float32
                )
                self._gpu_action_std = torch.as_tensor(
                    np.asarray(action_stats.std), dtype=torch.float32
                )
                if action_stats.q01 is not None and action_stats.q99 is not None:
                    self._gpu_action_q01 = torch.as_tensor(
                        np.asarray(action_stats.q01), dtype=torch.float32
                    )
                    self._gpu_action_q99 = torch.as_tensor(
                        np.asarray(action_stats.q99), dtype=torch.float32
                    )

        if tokenize_tf is not None:
            self._gpu_discrete_state_input = bool(tokenize_tf.discrete_state_input)
            tokenizer = tokenize_tf.tokenizer

            def tokenize_fn(prompt: str):
                # pi05_maniskill uses discrete_state_input=False → state unused.
                tokens, mask = tokenizer.tokenize(prompt, None)
                return tokens, mask

            self._gpu_tokenize_fn = tokenize_fn
            if (
                bool(self.config.cache_tokenized_prompt)
                and not self._gpu_discrete_state_input
            ):
                self._gpu_token_cache = PromptTokenCache(tokenize_fn)
            else:
                self._gpu_token_cache = None

        if bool(self.config.use_gpu_input_transform):
            if self._gpu_tokenize_fn is None:
                logger.warning(
                    "use_gpu_input_transform=True but TokenizePrompt was not found; "
                    "falling back to CPU input_transform"
                )
            else:
                logger.info(
                    "GPU input transform enabled: pack_mode=%s quantiles=%s "
                    "token_cache=%s action_dim=%d",
                    self._gpu_pack_mode,
                    self._gpu_use_quantiles,
                    self._gpu_token_cache is not None,
                    self._gpu_action_dim,
                )

    def _init_gpu_output_transform(
        self, output_transforms: Sequence[_transforms.DataTransformFn]
    ) -> None:
        """Validate the output chain and enable the GPU output path.

        Only chains that are exactly {Unnormalize, ManiSkill/LiberoOutputs}
        can be replicated on GPU; anything else (e.g. AbsoluteActions from
        extra_delta_transform, FAST token decode) falls back to the CPU
        per-sample compose.
        """
        self._gpu_output_ok = False
        self._gpu_output_use_quantiles = False
        # ManiSkillOutputs / LiberoOutputs both do actions[:, :7].
        self._gpu_output_slice_dim = 7
        if not bool(getattr(self.config, "use_gpu_output_transform", False)):
            return
        for tf in output_transforms:
            name = type(tf).__name__
            if isinstance(tf, _transforms.Unnormalize):
                self._gpu_output_use_quantiles = bool(tf.use_quantiles)
                if tf.norm_stats is not None and self._gpu_action_mean is None and (
                    self._gpu_action_q01 is None
                ):
                    logger.warning(
                        "use_gpu_output_transform=True but action norm stats "
                        "were not captured; falling back to CPU output_transform"
                    )
                    return
            elif name in ("ManiSkillOutputs", "LiberoOutputs"):
                pass
            else:
                logger.warning(
                    "use_gpu_output_transform=True but output chain contains "
                    "unsupported transform %s; falling back to CPU "
                    "output_transform",
                    name,
                )
                return
        if self._gpu_output_use_quantiles and (
            self._gpu_action_q01 is None or self._gpu_action_q99 is None
        ):
            logger.warning(
                "use_gpu_output_transform=True with quantile norm but action "
                "q01/q99 missing; falling back to CPU output_transform"
            )
            return
        self._gpu_output_ok = True
        logger.info(
            "GPU output transform enabled: quantiles=%s slice_dim=%d chunk=%d",
            self._gpu_output_use_quantiles,
            self._gpu_output_slice_dim,
            int(self.config.action_chunk),
        )

    def _gpu_output_transform(self, actions: torch.Tensor) -> torch.Tensor:
        """Batch GPU replacement for the per-sample ``output_transform`` loop.

        Numerically mirrors: Unnormalize -> actions[:, :7] (env dims) ->
        actions[:, :action_chunk] (horizon). Returns float32 on the input
        device (no D2H).
        """
        from joyrlinf.models.openpi_gpu_preprocess import unnormalize_actions

        device = actions.device
        x = actions.detach().to(torch.float32)

        def _to_dev(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device=device)

        if self._gpu_action_mean is not None or self._gpu_action_q01 is not None:
            x = unnormalize_actions(
                x,
                mean=_to_dev(self._gpu_action_mean),
                std=_to_dev(self._gpu_action_std),
                q01=_to_dev(self._gpu_action_q01),
                q99=_to_dev(self._gpu_action_q99),
                use_quantiles=self._gpu_output_use_quantiles,
            )
        x = x[..., : self._gpu_output_slice_dim]
        return x[:, : self.config.action_chunk].contiguous()

    @staticmethod
    def _ensure_nhwc_uint8(images: torch.Tensor) -> torch.Tensor:
        """Convert env images to uint8 NHWC for GPU preprocess."""
        if not torch.is_tensor(images):
            images = torch.as_tensor(images)
        if images.ndim != 4:
            raise ValueError(
                f"expected image batch [B,H,W,C] or [B,C,H,W], got {tuple(images.shape)}"
            )
        # CHW -> HWC
        if images.shape[1] == 3 and images.shape[-1] != 3:
            images = images.permute(0, 2, 3, 1).contiguous()
        if images.dtype == torch.uint8:
            return images
        if images.dtype.is_floating_point:
            # Accept [0,1] or [0,255] float.
            max_v = (
                float(images.detach().float().max().item()) if images.numel() else 0.0
            )
            if max_v <= 1.0 + 1e-3:
                images = images.clamp(0.0, 1.0) * 255.0
            return images.round().clamp(0, 255).to(torch.uint8)
        return images.to(torch.uint8)

    def _can_use_gpu_input_transform(self, obs: dict, *, need_prompt: bool) -> bool:
        if not bool(getattr(self.config, "use_gpu_input_transform", False)):
            return False
        if self._gpu_tokenize_fn is None:
            return False
        if self._gpu_discrete_state_input:
            # State-conditioned tokenization still uses the CPU compose path.
            return False
        if "calvin" in str(self.config.config_name).lower():
            return False
        if "observation/image" not in obs or "observation/state" not in obs:
            return False
        if need_prompt and "prompt" not in obs:
            return False
        if self._gpu_use_quantiles and (
            self._gpu_state_q01 is None or self._gpu_state_q99 is None
        ):
            return False
        return True

    def _gpu_input_transform(
        self,
        obs: dict,
        *,
        prompts: list[str] | None = None,
        reuse_tokens: bool = False,
        env_ids: Sequence[int] | None = None,
    ) -> dict:
        """Batch GPU preprocess replacing per-sample numpy ``input_transform``."""
        from joyrlinf.models.openpi_gpu_preprocess import preprocess_obs_batch

        device = next(self.parameters()).device
        base = self._ensure_nhwc_uint8(obs["observation/image"]).to(device=device)
        state = obs["observation/state"]
        if not torch.is_tensor(state):
            state = torch.as_tensor(state)
        state = state.to(device=device, dtype=torch.float32)

        wrist = obs.get("observation/wrist_image")
        if wrist is not None and self._gpu_pack_mode == "libero":
            wrist = self._ensure_nhwc_uint8(wrist).to(device=device)
        else:
            wrist = None

        if prompts is None:
            raw_prompts = obs.get("prompt", None)
            if raw_prompts is None:
                prompts = ["xxxx"] * base.shape[0]
            elif isinstance(raw_prompts, (list, tuple)):
                prompts = list(raw_prompts)
            else:
                prompts = [str(raw_prompts)] * base.shape[0]

        def _to_dev(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device=device)

        if reuse_tokens and "tokenized_prompt" in obs:
            # Training forward: images/state on GPU, keep cached tokens.
            from joyrlinf.models.openpi_gpu_preprocess import (
                images_uint8_to_float,
                normalize_state,
                pack_libero_images,
                pack_maniskill_images,
                pad_state,
                resize_with_pad,
            )

            base_f = images_uint8_to_float(resize_with_pad(base, 224, 224))
            if self._gpu_pack_mode == "libero" and wrist is not None:
                wrist_f = images_uint8_to_float(resize_with_pad(wrist, 224, 224))
                image, image_mask = pack_libero_images(base_f, wrist_f)
            else:
                image, image_mask = pack_maniskill_images(base_f)

            if self._gpu_use_quantiles:
                state_out = normalize_state(
                    state,
                    mean=_to_dev(self._gpu_state_mean)
                    if self._gpu_state_mean is not None
                    else _to_dev(self._gpu_state_q01),
                    use_quantiles=True,
                    q01=_to_dev(self._gpu_state_q01),
                    q99=_to_dev(self._gpu_state_q99),
                )
            elif self._gpu_state_mean is not None:
                state_out = normalize_state(
                    state,
                    _to_dev(self._gpu_state_mean),
                    _to_dev(self._gpu_state_std),
                )
            else:
                state_out = state
            state_out = pad_state(state_out, self._gpu_action_dim)

            tok = obs["tokenized_prompt"]
            tok_mask = obs["tokenized_prompt_mask"]
            if not torch.is_tensor(tok):
                tok = torch.as_tensor(tok)
            if not torch.is_tensor(tok_mask):
                tok_mask = torch.as_tensor(tok_mask)
            # Match Observation.from_dict(torch.uint8) layout: float NCHW in [-1, 1].
            image = {
                k: v.permute(0, 3, 1, 2).contiguous() for k, v in image.items()
            }
            return {
                "image": image,
                "image_mask": image_mask,
                "state": state_out,
                "tokenized_prompt": tok.to(device=device),
                "tokenized_prompt_mask": tok_mask.to(device=device),
            }

        out = preprocess_obs_batch(
            base_images=base,
            state=state,
            prompts=prompts,
            state_mean=_to_dev(self._gpu_state_mean),
            state_std=_to_dev(self._gpu_state_std),
            state_q01=_to_dev(self._gpu_state_q01),
            state_q99=_to_dev(self._gpu_state_q99),
            use_quantile_norm=self._gpu_use_quantiles,
            action_dim=self._gpu_action_dim,
            wrist_images=wrist,
            pack_mode=self._gpu_pack_mode,
            tokenize_fn=self._gpu_tokenize_fn,
            token_cache=self._gpu_token_cache,
            env_ids=env_ids,
        )
        out["image"] = {
            k: v.permute(0, 3, 1, 2).contiguous() for k, v in out["image"].items()
        }
        return out

    def _gpu_state_only_transform(self, obs: dict) -> torch.Tensor:
        """State normalization only (injected-KV training path).

        Must stay numerically identical to the state branch of
        ``_gpu_input_transform(reuse_tokens=True)``: normalize with the same
        stats, then zero-pad to the model action dim.
        """
        from joyrlinf.models.openpi_gpu_preprocess import (
            normalize_state,
            pad_state,
        )

        device = next(self.parameters()).device
        state = obs["observation/state"]
        if not torch.is_tensor(state):
            state = torch.as_tensor(state)
        state = state.to(device=device, dtype=torch.float32)

        def _to_dev(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device=device)

        if self._gpu_use_quantiles:
            state_out = normalize_state(
                state,
                mean=_to_dev(self._gpu_state_mean)
                if self._gpu_state_mean is not None
                else _to_dev(self._gpu_state_q01),
                use_quantiles=True,
                q01=_to_dev(self._gpu_state_q01),
                q99=_to_dev(self._gpu_state_q99),
            )
        elif self._gpu_state_mean is not None:
            state_out = normalize_state(
                state,
                _to_dev(self._gpu_state_mean),
                _to_dev(self._gpu_state_std),
            )
        else:
            state_out = state
        return pad_state(state_out, self._gpu_action_dim)

    def input_transform(self, obs: dict, transpose=True):
        _diag = logger.isEnabledFor(logging.DEBUG)
        t0 = time.perf_counter() if _diag else 0.0
        inputs = tree_map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}
        t1 = time.perf_counter() if _diag else 0.0

        # tensor -> numpy (Convert BFloat16/Float16 to float32 for numpy compatibility)
        inputs = tree_map(_to_numpy, inputs)
        t2 = time.perf_counter() if _diag else 0.0
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split
        batch_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: x[i], inputs)
            if transpose:
                # convert from [3,256,256] -> [256,256,3]
                sample = tree_map(
                    lambda x: (
                        x.transpose(1, 2, 0) if len(x.shape) == 3 and transpose else x
                    ),
                    sample,
                )
            else:
                sample = tree_map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"
            batch_samples.append(sample)
        # transform (parallel or serial)
        num_workers = self.config.input_transform_num_workers
        if num_workers <= 1 or len(batch_samples) <= 1:
            transformed_samples = [
                self._input_transform(s) for s in batch_samples
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=min(len(batch_samples), num_workers)
            ) as ex:
                transformed_samples = list(
                    ex.map(self._input_transform, batch_samples)
                )
        t_loop_end = time.perf_counter() if _diag else 0.0
        # recombine
        inputs = tree_map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        t_recombine_end = time.perf_counter() if _diag else 0.0
        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        t_end = time.perf_counter() if _diag else 0.0
        if _diag:
            if not hasattr(self, "_input_transform_dbg_idx"):
                self._input_transform_dbg_idx = 0
            self._input_transform_dbg_idx += 1
            idx = self._input_transform_dbg_idx
            if idx <= 4 or idx % 4 == 0:
                logger.debug(
                    "input_transform[%d]: prep_keys=%.3fs to_numpy=%.3fs "
                    "openpi_compose=%.3fs from_numpy=%.3fs tail=%.3fs total=%.3fs bs=%d",
                    idx,
                    t1 - t0,
                    t2 - t1,
                    t_loop_end - t2,
                    t_recombine_end - t_loop_end,
                    t_end - t_recombine_end,
                    t_end - t0,
                    batch_size,
                )
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: _to_numpy(x[i]), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = tree_map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    def forward(self, forward_type="default_forward", **kwargs):
        if forward_type == "sft_forward":
            return self.sft_forward(**kwargs)
        elif forward_type == "default_forward":
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        observation = data["observation"]
        actions = data["actions"]
        return PI0Pytorch.forward(self, observation, actions)

    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        import time as _time
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]

        reuse_kv = (
            getattr(self.config, "reuse_rollout_prefix_kv", False)
            and data.get("prefix_kv") is not None
        )

        _t0 = _time.time()
        device = chains.device
        # Injected-KV path never consumes images / language tokens (the
        # prefix is already baked into the KV), so skip image resize/pack
        # entirely. Mandatory for expert-only actors, a small win otherwise.
        state_only = reuse_kv and self._can_use_gpu_input_transform(
            data, need_prompt=False
        )
        if state_only:
            state = self._gpu_state_only_transform(data)
            images = img_masks = lang_tokens = lang_masks = None
            _t_after_it = _t_after_from_dict = _t_after_preprocess = _time.time()
        else:
            if self._can_use_gpu_input_transform(data, need_prompt=False):
                observation = self._gpu_input_transform(data, reuse_tokens=True)
            else:
                observation = self.input_transform(data, transpose=False)

            _t_after_it = _time.time()
            observation = _model.Observation.from_dict(observation)
            _t_after_from_dict = _time.time()
            images, img_masks, lang_tokens, lang_masks, state = (
                self._preprocess_observation(observation, train=False)
            )
            _t_after_preprocess = _time.time()
            images = [img.to(device) for img in images]
            img_masks = [img_mask.to(device) for img_mask in img_masks]
            state = state.to(device)
        _t_after_h2d = _time.time()
        _t_prep = _t_after_h2d - _t0

        _diag = logger.isEnabledFor(logging.DEBUG)
        if _diag:
            _ta0 = _time.time()
            torch.cuda.synchronize(device)
            _t_global_sync = _time.time() - _ta0
            _evt_start = torch.cuda.Event(enable_timing=True)
            _evt_end = torch.cuda.Event(enable_timing=True)
            _evt_start.record()
        else:
            _t_global_sync = 0.0

        _t1 = _time.time()
        if reuse_kv:
            injected_kv = _unpack_prefix_kv(data["prefix_kv"])
            injected_pad = data["prefix_pad_masks"]
            if injected_pad.dtype != torch.bool:
                injected_pad = injected_pad.bool()
            log_probs, value_t, entropy = self.get_log_prob_value(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                chains,
                denoise_inds,
                compute_values,
                past_key_values=injected_kv,
                prefix_pad_masks=injected_pad,
            )
        else:
            log_probs, value_t, entropy = self.get_log_prob_value(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                chains,
                denoise_inds,
                compute_values,
            )

        if _diag:
            _evt_end.record()
            _tc0 = _time.time()
            torch.cuda.synchronize(device)
            _t_post_sync = _time.time() - _tc0
            _gpu_ms = _evt_start.elapsed_time(_evt_end)
            _t_wall = _time.time() - _t1
            if not hasattr(self, "_dfwd_call_idx"):
                self._dfwd_call_idx = 0
            self._dfwd_call_idx += 1
            _in_step = ((self._dfwd_call_idx - 1) % 16) + 1
            if _in_step <= 4 or _in_step % 8 == 0:
                logger.debug(
                    "default_forward[%d]: prep_total=%.3fs "
                    "(input_transform=%.3fs from_dict=%.3fs preprocess=%.3fs h2d=%.3fs) | "
                    "pre_sync=%.3fs gpu_event=%.1fms post_sync=%.3fs wall=%.3fs",
                    self._dfwd_call_idx,
                    _t_prep,
                    _t_after_it - _t0,
                    _t_after_from_dict - _t_after_it,
                    _t_after_preprocess - _t_after_from_dict,
                    _t_after_h2d - _t_after_preprocess,
                    _t_global_sync,
                    _gpu_ms,
                    _t_post_sync,
                    _t_wall,
                )
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation - ensure float32 to prevent BFloat16 conversion issues
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            state = env_obs["states"]
            if torch.is_tensor(state):
                state = state.to(dtype=torch.float32)
            processed_obs["observation/state"] = state
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        processed_obs[key][sub_key] = sub_value.to(
                            device=device
                        ).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        return_obs=True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        if self._can_use_gpu_input_transform(to_process_obs, need_prompt=True):
            processed_obs = self._gpu_input_transform(to_process_obs)
        else:
            processed_obs = self.input_transform(
                to_process_obs, transpose=False
            )  # policy input obs -> model input obs
            processed_obs = self.precision_processor(
                processed_obs
            )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)
        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        if self._gpu_output_ok:
            # Stays on GPU; env consumes CUDA tensors directly.
            actions = self._gpu_output_transform(outputs["actions"])
        else:
            actions = self.output_transform(
                {"actions": outputs["actions"], "state": observation.state}
            )["actions"].numpy()

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        if "prefix_kv" in outputs:
            forward_inputs["prefix_kv"] = outputs["prefix_kv"]
            forward_inputs["prefix_pad_masks"] = outputs["prefix_pad_masks"]
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
        return actions, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        self._ensure_vlm_present("sample_actions")
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = self._attn_impl  # noqa: SLF001

        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)

        # add value based on the vlm for pi05, expert for pi0
        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(prefix_output)
        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        if mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                num_steps,
                compute_values,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        # post process for logprob
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]
        # post process for value
        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        outputs = {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }
        if getattr(self.config, "reuse_rollout_prefix_kv", False):
            outputs["prefix_kv"] = _pack_prefix_kv(past_key_values)
            outputs["prefix_pad_masks"] = prefix_pad_masks
        return outputs

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.config.noise_level).to(device)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        # velocity prediction
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
        )
        v_t = self.action_out_proj(
            suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        )  # [bs,n_action_steps,max_action_dim]
        # value prediction
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            # use chunk critic input
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            # detach critic input
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)
        # ode sde mix sampling
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(
                    suffix_out.to(dtype=self.action_out_proj.weight.dtype)
                )
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            self._attn_impl  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    # TODO: to check potential nan here
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
        past_key_values=None,
        prefix_pad_masks=None,
    ):
        bsize = state.shape[0]
        if past_key_values is not None:
            # External (rollout-produced) prefix KV injection: skip the
            # prefix recompute below. Only valid when the VLM is frozen
            # (train_expert_only) and no value head reads prefix_output.
            if prefix_pad_masks is None:
                raise ValueError(
                    "prefix_pad_masks is required when past_key_values "
                    "is injected"
                )
            if self.use_vlm_value:
                raise ValueError(
                    "external past_key_values is incompatible with "
                    "value_after_vlm (prefix_output unavailable)"
                )
            if not getattr(self.config, "train_expert_only", False):
                raise ValueError(
                    "external past_key_values requires train_expert_only "
                    "(frozen VLM); otherwise rollout KV is stale"
                )
            past_key_values = _detach_kv_tree(past_key_values)
            prefix_output = None
            return self._log_prob_value_from_kv(
                state,
                chains,
                denoise_inds,
                compute_values,
                prefix_pad_masks,
                past_key_values,
                prefix_output,
            )
        self._ensure_vlm_present("get_log_prob_value (prefix recompute)")
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = self._attn_impl  # noqa: SLF001

        # Compute image and language key value cache
        [prefix_output, _], past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return self._log_prob_value_from_kv(
            state,
            chains,
            denoise_inds,
            compute_values,
            prefix_pad_masks,
            past_key_values,
            prefix_output,
        )

    def _log_prob_value_from_kv(
        self,
        state,
        chains,
        denoise_inds,
        compute_values,
        prefix_pad_masks,
        past_key_values,
        prefix_output,
    ):
        """Denoise-loop logprob given a ready prefix KV (recomputed or injected)."""
        bsize = state.shape[0]
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # get log prob
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            if self.use_vlm_value:
                chains_values.append(self.get_value_from_vlm(prefix_output))
            else:
                chains_values.append(value_t)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        # entropy is only available for flow-noise method
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy

    def get_value_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            paligemma = self.paligemma_with_expert.paligemma
            if paligemma is None:  # expert-only actor: nothing to freeze
                return
            paligemma.eval()
            for params in paligemma.parameters():
                params.requires_grad = False

    @property
    def vlm_stripped(self) -> bool:
        return self.paligemma_with_expert.paligemma is None

    def strip_vlm_for_expert_only_actor(self) -> None:
        """Physically remove the VLM (SigLIP + 2.9B PaliGemma LLM) from this
        instance (expert-only actor).

        Only valid for the actor (training) instance with
        ``reuse_rollout_prefix_kv``: the injected-KV path runs exclusively on
        the gemma expert + heads, so VLM weights never contribute. After
        stripping, any prefix recompute path (``sample_actions`` or
        ``get_log_prob_value`` without external KV) raises.
        """
        if not getattr(self.config, "reuse_rollout_prefix_kv", False):
            raise ValueError(
                "expert_only_actor requires reuse_rollout_prefix_kv: without "
                "injected KV the actor must recompute the VLM prefix."
            )
        # Assigning None to a registered submodule is supported by nn.Module
        # (same pattern as gemma_expert.model.embed_tokens = None upstream).
        self.paligemma_with_expert.paligemma = None

    def _ensure_vlm_present(self, caller: str) -> None:
        if self.paligemma_with_expert.paligemma is None:
            raise RuntimeError(
                f"{caller} needs the VLM prefix forward, but this instance "
                "was built with expert_only_actor (VLM stripped). Only the "
                "injected-KV training path is available."
            )
