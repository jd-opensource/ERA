"""OpenVLA-OFT model wrapper for joyrlinf (inference and training).

Loads base OpenVLA model + LoRA adapter and provides the same interface
as OpenPiModel so that rollout_worker and actor_worker work unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from prismatic.vla.constants import STOP_INDEX
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.generation import TopKLogitsWarper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal logprob / entropy helpers (inlined from rlinf)
# ---------------------------------------------------------------------------

def _compute_logprobs_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logprobs = F.log_softmax(logits, dim=-1)
    return torch.gather(logprobs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)


def _compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy over valid action bins only (avoid -inf * 0 = NaN)."""
    logprobs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(logprobs)
    # Mask out -inf positions to avoid (-inf) * 0 = NaN
    finite_mask = torch.isfinite(logprobs)
    safe_logprobs = torch.where(finite_mask, logprobs, torch.zeros_like(logprobs))
    p_log_p = safe_logprobs * probs
    return -p_log_p.sum(dim=-1)


# ---------------------------------------------------------------------------
# OpenVLA-OFT for RL (adapted from rlinf)
# ---------------------------------------------------------------------------

class OpenVLAOFTForRLActionPrediction(torch.nn.Module):
    """Thin RL wrapper around prismatic OpenVLAForActionPrediction.

    Adds:
      - predict_action_batch(env_obs=...) for rollout
      - default_forward(forward_inputs=...) for actor training
      - action bin discretisation + unnormalization
    """

    def __init__(
        self,
        base_model,
        norm_stats: dict,
        unnorm_key: str,
        action_dim: int,
        num_action_chunks: int,
        max_prompt_length: int,
        add_value_head: bool = False,
        value_output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.model = base_model                # already loaded with LoRA
        self.norm_stats = norm_stats
        self.unnorm_key = self._resolve_unnorm_key(unnorm_key)
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.max_prompt_length = max_prompt_length
        self.input_processor = None
        self.bin_centers = None
        self.vocab_size = None
        self.config = self.model.config

        # PPO critic head (only created when add_value_head is set; GRPO
        # configs never instantiate it and pay no extra forward cost).
        self.value_head = None
        if add_value_head:
            from joyrlinf.models.modules.value_head import ValueHead

            out_dim = int(value_output_dim or num_action_chunks)
            if out_dim not in (1, num_action_chunks):
                raise ValueError(
                    f"value_output_dim must be 1 or num_action_chunks="
                    f"{num_action_chunks}, got {out_dim}"
                )
            hidden_size = self.config.text_config.hidden_size
            self.value_head = ValueHead(
                input_dim=hidden_size,
                hidden_sizes=(1024, 512, 256),
                output_dim=out_dim,
                activation="relu",
                bias_last=True,
            )

    def _resolve_unnorm_key(self, key: str) -> str:
        if key in self.norm_stats:
            return key
        alt = f"{key}_no_noops"
        if alt in self.norm_stats:
            return alt
        # fallback: try first key containing the requested string
        for k in self.norm_stats:
            if key in k:
                return k
        raise KeyError(f"unnorm_key '{key}' not in norm_stats: {list(self.norm_stats.keys())}")

    # -- setup ----------------------------------------------------------------

    def setup_processor(self, model_path: str):
        """Create tokenizer + image processor from the base checkpoint."""
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig as OpenVLAOFTConfig
        from joyrlinf.models.prismatic.processing_prismatic import (
            MultiInputPrismaticProcessor,
            PrismaticImageProcessor,
        )

        AutoConfig.register("openvla", OpenVLAOFTConfig)
        AutoProcessor.register(OpenVLAOFTConfig, MultiInputPrismaticProcessor)

        image_processor = PrismaticImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        self.input_processor = MultiInputPrismaticProcessor.from_pretrained(
            model_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=True,
        )

    def setup_action_bins(self):
        cfg = self.config
        self.vocab_size = cfg.text_config.vocab_size - cfg.pad_to_multiple_of
        bins = np.linspace(-1, 1, cfg.n_action_bins)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens."""
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], self.action_dim * self.num_action_chunks))
            .to(input_ids.device)
            .to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        stop_token_id = (
            torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype)
            * STOP_INDEX
        )
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        mask_extension = (
            torch.ones(
                (
                    attention_mask.shape[0],
                    input_ids.shape[-1] - attention_mask.shape[-1],
                )
            )
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _build_embedding(self, input_ids, attention_mask, pixel_values):
        """Build multimodal embeddings for action prediction (aligned with RLinf)."""
        assert torch.all(input_ids[:, -1] == STOP_INDEX)
        assert input_ids.shape[0] == attention_mask.shape[0]
        assert input_ids.shape[1] == attention_mask.shape[1]

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        n_patch_tokens = (
            self.model.vision_backbone.get_num_patches()
            * self.model.vision_backbone.get_num_images_in_input()
        )

        all_actions_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        all_actions_mask[:, -self.action_dim * self.num_action_chunks :] = True

        input_embeddings = self.model.get_input_embeddings()(input_ids)
        input_embeddings = input_embeddings * (~all_actions_mask.unsqueeze(-1))

        projected_patch_embeddings = self.model._process_vision_features(
            pixel_values, None, use_film=False
        )
        assert projected_patch_embeddings.shape[1] == n_patch_tokens

        projected_patch_embeddings = projected_patch_embeddings.reshape(
            input_embeddings.shape[0], -1, *projected_patch_embeddings.shape[2:]
        )
        mm_embeddings, mm_attention_mask = self.model._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        assert (
            mm_embeddings.shape[1]
            == input_embeddings.shape[1] + projected_patch_embeddings.shape[1]
        )
        assert (
            mm_attention_mask.shape[1]
            == attention_mask.shape[1] + projected_patch_embeddings.shape[1]
        )

        return mm_embeddings, mm_attention_mask

    def _compute_values(self, action_hidden: torch.Tensor) -> torch.Tensor:
        """Critic values from action-position hidden states.

        Args:
            action_hidden: [B, C*A, H] last-layer hidden states at the
                action-decision positions (same slice as the action logits).

        Returns:
            values: [B, C] float32, one value per action chunk.
        """
        if self.value_head is None:
            raise RuntimeError(
                "value computation requested but value_head is not built; "
                "set actor.model.add_value_head: true for PPO."
            )
        pooled = action_hidden.mean(dim=1)
        head_dtype = self.value_head.mlp[0].weight.dtype
        values = self.value_head(pooled.to(head_dtype))
        if values.shape[-1] == 1 and self.num_action_chunks > 1:
            values = values.expand(-1, self.num_action_chunks)
        if values.shape != (action_hidden.shape[0], self.num_action_chunks):
            raise RuntimeError(
                f"value head output shape {tuple(values.shape)} != "
                f"[B={action_hidden.shape[0]}, C={self.num_action_chunks}]"
            )
        return values.float()

    # -- forward dispatch (FSDP compatibility) --------------------------------

    def forward(self, forward_type: str = "default_forward", **kwargs):
        """Dispatch used by FSDP-wrapped actor worker.

        The actor passes ``data=mb`` where ``mb`` is a dict containing
        both model-specific tensors (input_ids, attention_mask, pixel_values,
        action_tokens) and RL metadata (prev_logprobs, advantages, ...).
        We extract only what the model needs.
        """
        if forward_type == "default_forward":
            data = kwargs.pop("data", None)
            compute_values = kwargs.pop("compute_values", False)
            if data is not None:
                # Extract model inputs from the mixed dict
                model_keys = ("input_ids", "attention_mask", "pixel_values", "action_tokens")
                fi = {k: data[k] for k in model_keys if k in data}
                return self.default_forward(
                    forward_inputs=fi,
                    compute_logprobs=True,
                    compute_values=compute_values,
                    **kwargs,
                )
            return self.default_forward(
                compute_logprobs=True,
                compute_values=compute_values,
                **kwargs,
            )
        raise NotImplementedError(f"Unknown forward_type: {forward_type}")

    # -- predict (rollout) ----------------------------------------------------

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict,
        mode: str = "train",
        calculate_logprobs: bool = True,
        calculate_values: bool = True,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        # 1. build prompts
        task_descriptions = [
            f"In: What action should the robot take to {t.lower()}?\nOut: "
            for t in env_obs["task_descriptions"]
        ]

        # 2. images
        main_images = env_obs["main_images"]
        if main_images.ndim == 4:
            main_images = main_images.unsqueeze(1)          # [B, H, W, C] -> [B, 1, H, W, C]
        main_images = main_images.permute(0, 1, 4, 2, 3)    # -> [B, 1, C, H, W]

        images_tensor = main_images.to(device=device, dtype=torch.float32)

        # 3. tokenize + process
        inputs = self.input_processor(
            text=task_descriptions,
            images={"images": images_tensor},
            proprio_states=env_obs.get("states"),
            padding="max_length",
            max_length=self.max_prompt_length,
        )
        input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = inputs["attention_mask"].to(device=device, dtype=torch.bool)
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        # Reshape: [B, N, C, H, W] -> [B, N*C, H, W] (channel-stacked for fused backbone)
        B_v, N_v, C_v, H_v, W_v = pixel_values.shape
        pixel_values = pixel_values.reshape(B_v, N_v * C_v, H_v, W_v)

        # 4. prepare for action prediction tokens
        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        act_token_count = self.action_dim * self.num_action_chunks
        n_prompt_tokens = input_ids.shape[-1] - 1
        n_patches = (
            self.model.vision_backbone.get_num_patches()
            * self.model.vision_backbone.get_num_images_in_input()
        )

        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)
        assert torch.all(
            attention_mask[:, -1 - act_token_count :] == 1
        )

        # 5. build multimodal embedding
        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids, attention_mask, pixel_values
        )
        mm_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        # 6. forward pass
        outputs = self.model.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=mm_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=calculate_values,
            return_dict=True,
        )

        logits_tensor = outputs.logits[
            :,
            n_patches + n_prompt_tokens : n_patches + n_prompt_tokens + act_token_count,
            :,
        ]

        n_action_bins = self.config.n_action_bins
        logits_tensor[..., :self.vocab_size - n_action_bins] = -torch.inf
        logits_tensor[..., self.vocab_size:] = -torch.inf

        # 7. sample or argmax actions
        if do_sample:
            processed = logits_tensor / temperature
            if top_k > 0:
                processed = TopKLogitsWarper(top_k)(None, processed)
            logprobs_tensor = F.log_softmax(processed, dim=-1)
            probs = torch.exp(logprobs_tensor).view(-1, logprobs_tensor.shape[-1])
            idxs = torch.multinomial(probs, num_samples=1).view(
                logprobs_tensor.shape[0], logprobs_tensor.shape[1]
            )
        else:
            idxs = logits_tensor.argmax(dim=-1)

        # 8. discretise -> unnormalize -> numpy actions
        discretized = self.vocab_size - idxs.cpu().numpy()
        discretized = np.clip(discretized - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized = np.asarray([self.bin_centers[d] for d in discretized])
        normalized = normalized.reshape(-1, self.action_dim)

        actions = self._unnormalize_actions(normalized)
        actions = actions.reshape(idxs.shape)

        # 9. compute logprobs
        action_logits = logits_tensor.clone() if do_sample else logits_tensor
        if do_sample:
            action_logits = logits_tensor / temperature
        action_logits[..., :self.vocab_size - n_action_bins] = -torch.inf
        action_logits[..., self.vocab_size:] = -torch.inf
        chunk_logprobs = _compute_logprobs_from_logits(action_logits, idxs)

        # 10. forward inputs for actor training (all on CPU, actor moves to GPU)
        forward_inputs = {
            "input_ids": inputs["input_ids"].cpu(),
            "attention_mask": inputs["attention_mask"].to(torch.long).cpu(),
            "pixel_values": pixel_values.detach().cpu(),  # already reshaped to [B, N*C, H, W]
            "action_tokens": idxs.reshape(-1, self.num_action_chunks, self.action_dim).cpu(),
        }

        result = {
            "prev_logprobs": chunk_logprobs,
            "forward_inputs": forward_inputs,
        }
        if calculate_values:
            # Hidden states at the same positions that produce action logits.
            action_hidden = outputs.hidden_states[-1][
                :,
                n_patches + n_prompt_tokens : n_patches + n_prompt_tokens + act_token_count,
            ]
            result["prev_values"] = self._compute_values(action_hidden)

        chunk_actions = torch.as_tensor(
            actions.reshape(-1, self.num_action_chunks, self.action_dim)
        )
        return chunk_actions, result

    # -- forward (actor training) ---------------------------------------------

    def default_forward(
        self,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: bool = False,
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        **kwargs,
    ) -> dict[str, Any]:
        if forward_inputs is not None:
            input_ids = forward_inputs["input_ids"]
            attention_mask = forward_inputs["attention_mask"]
            pixel_values = forward_inputs["pixel_values"]
            action_tokens = forward_inputs["action_tokens"]
            # action_tokens: [B, num_action_chunks, action_dim] -> flatten to [B, C*A]
            action_tokens = action_tokens.reshape(
                action_tokens.shape[0], self.action_dim * self.num_action_chunks
            )

        act_token_count = self.action_dim * self.num_action_chunks

        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        attention_mask = attention_mask.to(torch.long)
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)
        assert torch.all(
            input_ids[:, -act_token_count - 2] == 29871
        )
        assert torch.all(
            attention_mask[:, -2 - act_token_count :] == 1
        )

        mm_embeddings, mm_atn_mask = self._build_embedding(
            input_ids, attention_mask, pixel_values
        )
        mm_position_ids = mm_atn_mask.cumsum(dim=1) - 1

        outputs = self.model.language_model(
            input_ids=None,
            attention_mask=mm_atn_mask,
            position_ids=mm_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=output_hidden_states or compute_values,
            return_dict=True,
        )

        logprobs = None
        entropy = None
        values = None

        if compute_logprobs:
            logits = outputs.logits[:, -act_token_count - 1 : -1]
            processed = logits / temperature

            n_action_bins = self.config.n_action_bins
            action_logits = processed.clone()
            action_logits[..., :self.vocab_size - n_action_bins] = -torch.inf
            action_logits[..., self.vocab_size:] = -torch.inf

            if top_k > 0:
                action_logits = TopKLogitsWarper(top_k)(None, action_logits)

            logprobs = _compute_logprobs_from_logits(action_logits, action_tokens)
            if compute_entropy:
                entropy = _compute_entropy_from_logits(action_logits)

        if compute_values:
            action_hidden = outputs.hidden_states[-1][:, -act_token_count - 1 : -1]
            values = self._compute_values(action_hidden)

        return {"logprobs": logprobs, "entropy": entropy, "values": values}

    # -- helpers --------------------------------------------------------------

    def _unnormalize_actions(self, normalized_actions):
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]

        if "q01" in action_norm_stats:
            mask = np.array(action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool)))
            action_high = np.array(action_norm_stats["q99"])
            action_low = np.array(action_norm_stats["q01"])
        else:
            mask = np.array(action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool)))
            action_high = np.array(action_norm_stats["max"])
            action_low = np.array(action_norm_stats["min"])

        action_dim = normalized_actions.shape[-1]
        repeat_factor = action_dim // action_high.shape[0]
        if repeat_factor > 1:
            action_high = np.tile(action_high, repeat_factor)
            action_low = np.tile(action_low, repeat_factor)
            mask = np.tile(mask, repeat_factor)

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )
        return actions

    def preprocess_for_train(self, data: dict) -> dict:
        return data


# ---------------------------------------------------------------------------
# Model wrapper (joyrlinf interface)
# ---------------------------------------------------------------------------

class OpenVlaOftModel:
    """Wraps OpenVLA-OFT model for both rollout inference and actor training."""

    def __init__(self, model_cfg: DictConfig, device: torch.device, mode: str = "inference"):
        self.cfg = model_cfg
        self.device = device
        self.mode = mode
        self.model = self._build_model()

    def _build_model(self) -> OpenVLAOFTForRLActionPrediction:
        from peft import PeftConfig, PeftModel
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

        # prismatic.overwatch runs logging.config.dictConfig at import time and
        # installs a RichHandler (| >> format). Restore joyrlinf handlers before
        # any model-load logs so console + file output stay consistent.
        try:
            from joyrlinf.utils.logging_setup import reassert_logging
            reassert_logging()
        except Exception:
            pass

        model_path = self.cfg.model_path
        lora_path = self.cfg.get("lora_path", None)

        logger.info("Loading base OpenVLA model from %s", model_path)
        AutoConfig.register("openvla", OpenVLAConfig)

        base_model = OpenVLAForActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )

        # Load norm_stats from both base model and lora checkpoint
        norm_stats = getattr(base_model.config, "norm_stats", {})

        # Also load dataset_statistics.json files
        for stats_dir in [model_path, lora_path]:
            if stats_dir is None:
                continue
            stats_file = os.path.join(stats_dir, "dataset_statistics.json")
            if os.path.isfile(stats_file):
                with open(stats_file, "r") as f:
                    extra = json.load(f)
                    norm_stats.update(extra)
                    logger.info("Loaded norm_stats from %s: keys=%s", stats_file, list(extra.keys()))

        # Load LoRA adapter
        if lora_path and os.path.isdir(lora_path):
            logger.info("Loading LoRA adapter from %s", lora_path)
            peft_config = PeftConfig.from_pretrained(lora_path)
            base_model = PeftModel.from_pretrained(base_model, lora_path)
            # Merge LoRA for inference speed (optional - can set merge_lora: false in config)
            if self.mode == "inference" and self.cfg.get("merge_lora", True):
                base_model = base_model.merge_and_unload()
                logger.info("LoRA merged for inference")
            elif self.mode == "training":
                # LoRA adapter saved in inference_mode=True, explicitly enable training
                for n, p in base_model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = True
                trainable = sum(p.numel() for n, p in base_model.named_parameters() if p.requires_grad)
                logger.info("LoRA training enabled: %d trainable params (%.1fM)", trainable, trainable / 1e6)
        else:
            logger.warning("No LoRA adapter found at %s, using base model only", lora_path)

        # Use m3s400oft for ManiSkill (from LoRA adapter's dataset_statistics.json)
        unnorm_key = self.cfg.get("unnorm_key", "m3s400oft")
        if unnorm_key not in norm_stats:
            # Fallback: try known keys
            for candidate in ["m3s400oft", "libero_10_no_noops", "bridge_orig"]:
                if candidate in norm_stats:
                    unnorm_key = candidate
                    break
        logger.info("Using unnorm_key=%s", unnorm_key)

        action_dim = self.cfg.get("action_dim", 7)
        num_action_chunks = self.cfg.get("num_action_chunks", 8)
        max_prompt_length = self.cfg.get("max_prompt_length", 30)

        rl_model = OpenVLAOFTForRLActionPrediction(
            base_model=base_model,
            norm_stats=norm_stats,
            unnorm_key=unnorm_key,
            action_dim=action_dim,
            num_action_chunks=num_action_chunks,
            max_prompt_length=max_prompt_length,
            add_value_head=bool(self.cfg.get("add_value_head", False)),
            value_output_dim=self.cfg.get("value_output_dim", None),
        )
        rl_model.setup_processor(model_path)
        rl_model.setup_action_bins()
        rl_model.to(device=self.device, dtype=torch.bfloat16)

        total = sum(p.numel() for p in rl_model.parameters())
        trainable = sum(p.numel() for p in rl_model.parameters() if p.requires_grad)
        logger.info(
            "OpenVLA-OFT model ready: total=%.1fM trainable=%.1fM (%d keys)",
            total / 1e6, trainable / 1e6, trainable,
        )

        if self.mode == "inference":
            rl_model.eval()
        else:
            # Enable gradient checkpointing on the language model to save VRAM.
            # The PeftModel wraps the base model; traverse to find language_model.
            inner = rl_model.model
            for attr in ("base_model", "model"):
                if hasattr(inner, attr):
                    candidate = getattr(inner, attr)
                    if hasattr(candidate, "language_model"):
                        inner = candidate
                        break
            if hasattr(inner, "language_model"):
                # use_reentrant=False avoids FSDP use_orig_params writeback errors.
                inner.language_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                logger.info(
                    "Gradient checkpointing enabled on language_model "
                    "(use_reentrant=False)"
                )
            else:
                logger.warning("Could not find language_model for gradient checkpointing")

        return rl_model

    # -- public API (matches OpenPiModel) -------------------------------------

    @torch.no_grad()
    def predict(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        do_sample = (mode == "train")
        temperature = 1.0 if mode == "train" else 0.6

        chunk_actions, result = self.model.predict_action_batch(
            env_obs=env_obs,
            mode=mode,
            do_sample=do_sample,
            temperature=temperature,
            calculate_logprobs=True,
            calculate_values=compute_values,
        )
        return chunk_actions.cpu().numpy(), result

    def forward(
        self,
        data: dict[str, torch.Tensor],
        compute_values: bool = False,
        compute_entropy: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        return self.model.default_forward(
            forward_inputs=data,
            compute_logprobs=True,
            compute_entropy=compute_entropy,
            compute_values=compute_values,
            **kwargs,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        self.model.load_state_dict(state_dict, strict=strict)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def set_global_step(self, global_step: int) -> None:
        pass  # OpenVLA-OFT doesn't have step-dependent logic

    def offload_to_cpu(self) -> None:
        """Move model weights to CPU to free GPU for actor phase."""
        self.model = self.model.cpu()
        self.device = torch.device("cpu")

    def onload_to_device(self, device: torch.device) -> None:
        """Restore model to GPU for rollout inference (sync colocated reuse)."""
        self.model = self.model.to(device)
        self.device = device
        if self.mode == "inference":
            self.model.eval()

    def preprocess_obs(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Return env_obs as-is; predict_action_batch handles preprocessing."""
        return env_obs