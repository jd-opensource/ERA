"""Record one rollout predict's allocation history; tabulate oversize (>128MB) allocs."""
from __future__ import annotations

from collections import Counter

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

    model.predict(obs, mode="train", compute_values=False)  # warmup
    torch.cuda.memory._record_memory_history(max_entries=200000)
    model.predict(obs, mode="train", compute_values=False)
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)

    sizes = Counter()
    total = 0
    for ev in snap.get("device_traces", [[]])[0]:
        if ev.get("action") == "alloc" and ev["size"] > 128 * 2**20:
            sizes[ev["size"]] += 1
            total += ev["size"]
    print(f"oversize (>128MiB) allocs in ONE predict: count={sum(sizes.values())} "
          f"total={total / 2**30:.1f} GiB")
    for size, cnt in sorted(sizes.items(), key=lambda kv: -kv[0] * kv[1])[:12]:
        print(f"  {size / 2**20:8.1f} MiB  x{cnt:3d}  subtotal={size * cnt / 2**30:5.2f} GiB")


if __name__ == "__main__":
    main()
