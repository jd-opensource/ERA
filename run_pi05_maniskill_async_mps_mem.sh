#!/usr/bin/env bash
# Colocated async MPS + memory opts (S1 reuse/expert_only + Phase2 per_chunk D2H
# + predict⊥train gpu_phase_gate).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-.}"
torchrun --nproc_per_node=8 scripts/train_async_mps.py \
  --config configs/experiments/maniskill_grpo_pi05_async_mps_mem.yaml \
  "$@"
