#!/usr/bin/env bash
# 2 nodes × 2 GPUs — hybrid placement (node0: env+rollout ranks 0-1, node1: actor ranks 2-3).
#
# Usage:
#   export MASTER_ADDR=<ip_of_node0>
#   export MASTER_PORT=29500   # optional
#   On node0: NODE_RANK=0 ./scripts/run_2n2g_hybrid.sh sync
#   On node1: NODE_RANK=1 ./scripts/run_2n2g_hybrid.sh sync
#
#   Same for async:
#   NODE_RANK=0 ./scripts/run_2n2g_hybrid.sh async
#   NODE_RANK=1 ./scripts/run_2n2g_hybrid.sh async
#
# Args: sync | async  (default: sync)
#
# Training timing breakdown: JOYRLINF_DEBUG=1 (CUDA-sync + DEBUG logs; see docs/actor_rollout_perf_faq.md)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-sync}"
case "$MODE" in
  sync)
    CONFIG="configs/maniskill_grpo_pi05_sync_2n2g_hybrid.yaml"
    ;;
  async)
    CONFIG="configs/maniskill_grpo_pi05_async_2n2g_hybrid.yaml"
    ;;
  *)
    echo "usage: NODE_RANK=0|1 $0 [sync|async] [-- extra torchrun/train args]" >&2
    exit 1
    ;;
esac
shift || true

export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export NODE_RANK="${NODE_RANK:-0}"

exec "${ROOT}/scripts/run_multinode.sh" "$CONFIG" "$@"
