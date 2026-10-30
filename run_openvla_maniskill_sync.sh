#!/usr/bin/env bash
# Sync colocated OpenVLA-OFT ManiSkill GRPO (8 GPUs, env == rollout == actor).
set -euo pipefail
torchrun --nproc_per_node=8 scripts/train.py --config configs/maniskill_grpo_openvla_oft_sync.yaml
