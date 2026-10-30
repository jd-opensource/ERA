#!/usr/bin/env bash
# GR00T sync colocated training (8 GPUs: env == rollout == actor).
#
#   source switch_env gr00t
#   bash run_gr00t_sync.sh              # ManiSkill WidowX
#   bash run_gr00t_sync.sh libero long  # LIBERO-10 + Long SFT
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "libero" ]]; then
  shift
  exec bash "$REPO_ROOT/run_libero_gr00t.sh" sync "${1:-long}"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="configs/maniskill_grpo_gr00t_sync.yaml"
python scripts/validate_embodied_config.py --config "$CONFIG"

NPROC="${NPROC_PER_NODE:-8}"
exec torchrun --nproc_per_node="${NPROC}" scripts/train.py --config "$CONFIG"
