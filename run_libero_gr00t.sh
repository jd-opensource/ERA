#!/usr/bin/env bash
# LIBERO + GR00T GRPO launch script.
#
# Usage (inside docker / node with 8 GPUs):
#   cd /home/gyc/gitcode/joyrlinf
#   source switch_env gr00t          # required: gr00t venv, not openpi
#
# Async (env+rollout 0-5, actor 6-7):
#   bash run_libero_gr00t.sh long
#   bash run_libero_gr00t.sh async spatial
#
# Sync colocated (env == rollout == actor on 0-7):
#   bash run_libero_gr00t.sh sync long
#   bash run_libero_gr00t.sh sync spatial
#   bash run_libero_gr00t.sh sync goal
#
# Validate only:
#   bash run_libero_gr00t.sh validate sync long
#
# Available GR00T LIBERO SFT checkpoints:
#   long    -> /ufs_500T/model/colm_paper_model_test/RLinf-Gr00t-SFT-Long      (libero_10)
#   spatial -> /ufs_500T/model/RLinf-Gr00t-SFT-Spatial                            (libero_spatial)
#   goal    -> /ufs_500T/model/RLinf-Gr00t-SFT-Goal                               (libero_goal)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODE="async"
ACTION="train"
SUITE="long"

while [[ $# -gt 0 ]]; do
  case "$1" in
    async|sync)
      MODE="$1"
      ;;
    train|validate)
      ACTION="$1"
      ;;
    long|spatial|goal)
      SUITE="$1"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [async|sync] [validate] [long|spatial|goal]" >&2
      exit 1
      ;;
  esac
  shift
done

case "$SUITE" in
  long)
    if [[ "$MODE" == "sync" ]]; then
      CONFIG="configs/libero_grpo_gr00t_long_sync.yaml"
    else
      CONFIG="configs/libero_grpo_gr00t_long.yaml"
    fi
    MODEL="/ufs_500T/model/colm_paper_model_test/RLinf-Gr00t-SFT-Long"
    ;;
  spatial)
    if [[ "$MODE" == "sync" ]]; then
      CONFIG="configs/libero_grpo_gr00t_spatial_sync.yaml"
    else
      CONFIG="configs/libero_grpo_gr00t_spatial.yaml"
    fi
    MODEL="/ufs_500T/model/RLinf-Gr00t-SFT-Spatial"
    ;;
  goal)
    if [[ "$MODE" == "sync" ]]; then
      CONFIG="configs/libero_grpo_gr00t_goal_sync.yaml"
    else
      CONFIG="configs/libero_grpo_gr00t_goal.yaml"
    fi
    MODEL="/ufs_500T/model/RLinf-Gr00t-SFT-Goal"
    ;;
  *)
    echo "Unknown suite: $SUITE (expected long|spatial|goal)" >&2
    exit 1
    ;;
esac

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ ! -d "$MODEL" ]]; then
  echo "Model path not found: $MODEL" >&2
  exit 1
fi
if [[ ! -f "$MODEL/experiment_cfg/metadata.json" ]]; then
  echo "Missing GR00T metadata: $MODEL/experiment_cfg/metadata.json" >&2
  exit 1
fi

echo "==> mode=$MODE suite=$SUITE config=$CONFIG model=$MODEL"

python scripts/validate_embodied_config.py --config "$CONFIG"

if [[ "$ACTION" == "validate" ]]; then
  echo "Config validation passed."
  exit 0
fi

NPROC="${NPROC_PER_NODE:-8}"
echo "==> torchrun --nproc_per_node=${NPROC} scripts/train.py --config ${CONFIG}"
exec torchrun --nproc_per_node="${NPROC}" scripts/train.py --config "$CONFIG"
