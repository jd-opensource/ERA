#!/usr/bin/env bash
# Single-node multi-GPU launcher (torchrun). For multi-node, use run_multinode.sh.
#
# Usage: ./scripts/run_distributed.sh [config.yaml]
# Env: CONFIG, NPROC_PER_NODE (default 8), TORCHRUN_EXTRA

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/maniskill_grpo_pi05_async.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TORCHRUN_EXTRA="${TORCHRUN_EXTRA:-}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

if [[ -n "${1:-}" ]]; then
    CONFIG="$1"
    shift
fi

[[ -f "$CONFIG" ]] || { echo "error: config not found: $CONFIG" >&2; exit 1; }
command -v torchrun >/dev/null || { echo "error: torchrun not in PATH" >&2; exit 1; }

exec torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT:-29500}" \
    ${TORCHRUN_EXTRA} \
    scripts/train.py --config "$CONFIG" "$@"
