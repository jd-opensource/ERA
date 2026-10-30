"""Shared rollout inference micro-batching for VLA model wrappers."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


def read_predict_micro_batch_size(model_cfg: Any) -> int:
    """Read ``actor.model.predict_micro_batch_size`` (0 = no chunking)."""
    return int(model_cfg.get("predict_micro_batch_size", 0))


def infer_batch_size(env_obs: dict[str, Any]) -> int:
    """Infer batch dimension from the first tensor/list value in ``env_obs``."""
    for v in env_obs.values():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            return int(v.shape[0])
        if isinstance(v, list):
            return len(v)
    return 0


def split_obs(
    env_obs: dict[str, Any], chunks: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Split ``env_obs`` into micro-batches by slicing along dim 0."""
    result: list[dict[str, Any]] = []
    for start, end in chunks:
        batch: dict[str, Any] = {}
        for k, v in env_obs.items():
            if v is None:
                batch[k] = None
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                batch[k] = v[start:end]
            elif isinstance(v, list):
                batch[k] = v[start:end]
            elif isinstance(v, dict):
                batch[k] = {
                    sk: sv[start:end] if isinstance(sv, (torch.Tensor, np.ndarray)) else sv
                    for sk, sv in v.items()
                }
            else:
                batch[k] = v
        result.append(batch)
    return result


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def merge_predict_results(
    actions_list: list[Any],
    results_list: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Concatenate per-chunk actions and merge auxiliary result dicts."""
    merged_actions = np.concatenate(
        [_to_numpy_array(a) for a in actions_list], axis=0,
    )
    merged: dict[str, Any] = {}
    for key in results_list[0]:
        vals = [r[key] for r in results_list]
        if key == "forward_inputs":
            fi_merged: dict[str, Any] = {}
            for fi_key in vals[0]:
                fi_vals = [fi[fi_key] for fi in vals]
                sample = fi_vals[0]
                if isinstance(sample, torch.Tensor):
                    fi_merged[fi_key] = torch.cat(fi_vals, dim=0)
                elif isinstance(sample, np.ndarray):
                    fi_merged[fi_key] = np.concatenate(fi_vals, axis=0)
                elif isinstance(sample, list):
                    fi_merged[fi_key] = sum(fi_vals, [])
                else:
                    fi_merged[fi_key] = sample
            merged[key] = fi_merged
        elif isinstance(vals[0], torch.Tensor):
            merged[key] = torch.cat(vals, dim=0)
        elif isinstance(vals[0], np.ndarray):
            merged[key] = np.concatenate(vals, axis=0)
        else:
            merged[key] = vals[0]
    return merged_actions, merged


def predict_with_micro_batch(
    predict_fn: Callable[[dict[str, Any]], tuple[Any, dict[str, Any]]],
    env_obs: dict[str, Any],
    micro_batch_size: int,
    *,
    log_label: str = "predict",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run ``predict_fn`` in env-dim chunks when ``micro_batch_size > 0``."""
    batch_size = infer_batch_size(env_obs)
    if micro_batch_size <= 0 or batch_size <= micro_batch_size:
        actions, results = predict_fn(env_obs)
        return _to_numpy_array(actions), results

    chunks = [
        (i, min(i + micro_batch_size, batch_size))
        for i in range(0, batch_size, micro_batch_size)
    ]
    logger.debug(
        "%s micro-batch: total=%d mbs=%d chunks=%d",
        log_label, batch_size, micro_batch_size, len(chunks),
    )
    obs_chunks = split_obs(env_obs, chunks)
    actions_list: list[Any] = []
    results_list: list[dict[str, Any]] = []
    for obs_chunk in obs_chunks:
        act, res = predict_fn(obs_chunk)
        actions_list.append(act)
        results_list.append(res)
    return merge_predict_results(actions_list, results_list)
