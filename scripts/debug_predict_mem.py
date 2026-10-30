"""Minimal repro: rollout predict loop VRAM reserved growth.

Usage (inside container, openpi venv):
  cd /ufs_500T/jh/code/joyrlinf-dev/colocated_async/joyrlinf
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8 \
  /opt/venv/openpi/bin/python scripts/debug_predict_mem.py
"""
from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf


def main() -> None:
    cfg = OmegaConf.load("configs/experiments/maniskill_grpo_pi05_async_mps_mem.yaml")
    device = torch.device("cuda:0")

    from joyrlinf.models.openpi_model import OpenPiModel

    model = OpenPiModel(cfg.actor.model, device, mode="inference")

    bs = 16
    obs = {
        "main_images": torch.randint(0, 255, (bs, 224, 224, 3), dtype=torch.uint8, device=device),
        "wrist_images": None,
        "extra_view_images": None,
        "states": torch.randn(bs, 8, dtype=torch.float32, device=device),
        "task_descriptions": ["put the object on the plate"] * bs,
    }

    def report(tag: str) -> None:
        alloc = torch.cuda.memory_allocated(device) / 2**20
        rsv = torch.cuda.memory_reserved(device) / 2**20
        peak = torch.cuda.max_memory_allocated(device) / 2**20
        print(f"[{tag}] alloc={alloc:.0f}MiB peak_alloc={peak:.0f}MiB reserved={rsv:.0f}MiB", flush=True)

    report("after_build")
    results = None
    for i in range(8):
        actions, results = model.predict(obs, mode="train", compute_values=False)
        # emulate rollout loop: keep result alive until next chunk like buffer append does
        fi = results.get("forward_inputs", {})
        kv = fi.get("prefix_kv")
        if kv is not None and i == 0:
            print(f"prefix_kv shape={tuple(kv.shape)} dtype={kv.dtype} "
                  f"size={kv.numel() * kv.element_size() / 2**20:.1f}MiB", flush=True)
        report(f"predict_{i}")
    # dump allocator snapshot summary
    segs = torch.cuda.memory_snapshot()
    per_stream: dict[int, float] = {}
    n_exp = 0
    for s in segs:
        per_stream[s.get("stream", -1)] = per_stream.get(s.get("stream", -1), 0.0) + s["total_size"] / 2**20
        if s.get("is_expandable"):
            n_exp += 1
    print(f"segments={len(segs)} expandable={n_exp} per_stream_MiB={ {k: round(v) for k, v in per_stream.items()} }", flush=True)


if __name__ == "__main__":
    main()
