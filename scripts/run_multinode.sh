#!/usr/bin/env bash
# Multi-node launcher for joyrlinf (PyTorch torchrun).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Fix "[c10d] The hostname of the client socket cannot be retrieved":
# ensure the container's hostname resolves to a local IP.
_hostname="$(hostname 2>/dev/null || true)"
if [[ -n "${_hostname}" ]] && ! getent hosts "${_hostname}" >/dev/null 2>&1; then
    _ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [[ -n "${_ip}" ]]; then
        echo "${_ip} ${_hostname}" >> /etc/hosts 2>/dev/null || true
        echo "[run_multinode] Added /etc/hosts: ${_ip} ${_hostname}"
    fi
fi

CONFIG="${CONFIG:-configs/maniskill_grpo_pi05_disagg.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-}"
TORCHRUN_EXTRA="${TORCHRUN_EXTRA:-}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,1\}//' | head -40
    exit 0
fi

if [[ -n "${1:-}" ]]; then
    CONFIG="$1"
    shift
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "error: config not found: $CONFIG" >&2
    exit 1
fi

if ! command -v torchrun >/dev/null 2>&1; then
    echo "error: torchrun not in PATH" >&2
    exit 1
fi

if [[ "${NNODES}" -gt 1 ]]; then
    if [[ -z "${MASTER_ADDR}" ]]; then
        echo "error: NNODES=${NNODES} > 1 but MASTER_ADDR is empty." >&2
        echo "Set MASTER_ADDR to the IP or hostname of node 0 (rank 0)." >&2
        exit 1
    fi
    if [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
        echo "warning: MASTER_ADDR is ${MASTER_ADDR} with NNODES>1; other nodes cannot reach it." >&2
        echo "Use the real address of the first node." >&2
    fi
else
    if [[ -z "${MASTER_ADDR}" ]]; then
        MASTER_ADDR="127.0.0.1"
    fi
fi

if [[ "${NODE_RANK}" -lt 0 || "${NODE_RANK}" -ge "${NNODES}" ]]; then
    echo "error: NODE_RANK (${NODE_RANK}) must satisfy 0 <= NODE_RANK < NNODES (${NNODES})" >&2
    exit 1
fi

# Auto-detect NCCL_SOCKET_IFNAME if not set (multi-node NCCL must bind the
# interface that reaches MASTER_ADDR, not docker0 / 169.254.x / wrong NIC).
if [[ -z "${NCCL_SOCKET_IFNAME:-}" && "${NNODES}" -gt 1 ]]; then
    _detected=""
    # 1) Same L3 subnet as MASTER_ADDR (flat LAN)
    _master_prefix="${MASTER_ADDR%.*}"
    while IFS= read -r line; do
        _iface=$(echo "$line" | awk '{print $NF}')
        _ip=$(echo "$line" | awk '{print $2}' | cut -d/ -f1)
        if [[ "${_ip%.*}" == "${_master_prefix}" ]]; then
            _detected="${_iface}"
            break
        fi
    done < <(ip -4 -o addr show 2>/dev/null | grep -v '127.0.0.1')
    # 2) Fallback: routing table "dev" for MASTER_ADDR (MASTER on another subnet / routed)
    if [[ -z "${_detected}" ]]; then
        _route_out="$(ip -4 route get "${MASTER_ADDR}" 2>/dev/null || true)"
        if [[ -n "${_route_out}" ]]; then
            _detected="$(echo "${_route_out}" | awk '{for(i=1;i<NF;i++) if($i=="dev"){print $(i+1); exit}}')"
        fi
    fi
    # Never use loopback / docker for NCCL (route to local MASTER_ADDR yields dev lo)
    if [[ "${_detected}" == "lo" || "${_detected}" == docker0 \
        || "${_detected}" == br-* || "${_detected}" == veth* ]]; then
        echo "[run_multinode] Ignoring unsuitable NCCL_SOCKET_IFNAME=${_detected} (use a real LAN NIC)" >&2
        _detected=""
    fi
    if [[ -n "${_detected}" ]]; then
        export NCCL_SOCKET_IFNAME="${_detected}"
        echo "[run_multinode] Auto-detected NCCL_SOCKET_IFNAME=${_detected} (for ${MASTER_ADDR})"
    else
        echo "[run_multinode] WARNING: could not auto-detect NCCL_SOCKET_IFNAME for MASTER_ADDR=${MASTER_ADDR}" >&2
        echo "[run_multinode] Run on this host: ip -4 route get ${MASTER_ADDR}  &&  ip -4 -o addr show" >&2
        echo "[run_multinode] Then: export NCCL_SOCKET_IFNAME=<dev from 'dev ...'> (e.g. bond0, eth0, ib0)" >&2
    fi
fi

# Gloo TCP must not fall back to docker0 / random NIC while NCCL uses bond0,
# or P2P warmup can see peers as 169.254.x.x and fail with ncclSystemError.
if [[ "${NNODES}" -gt 1 && -n "${NCCL_SOCKET_IFNAME:-}" && -z "${GLOO_SOCKET_IFNAME+x}" ]]; then
    export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
    echo "[run_multinode] Exported GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} (aligned with NCCL)"
fi

# Default: Socket over Ethernet for multi-node; avoids bogus 169.254 IB paths.
if [[ "${NNODES}" -gt 1 && -z "${NCCL_IB_DISABLE+x}" ]]; then
    export NCCL_IB_DISABLE=1
    echo "[run_multinode] NCCL_IB_DISABLE=1 (multi-node default). For IB: export NCCL_IB_DISABLE=0 on all nodes."
fi

# CUDA_VISIBLE_DEVICES isolation (train.py: each rank sees one GPU as cuda:0)
# breaks NCCL P2P — each process can't see the other's GPU, so NCCL probes
# P2P, times out (~180s), then falls back to SHM.  Skip the probe.
if [[ -z "${NCCL_P2P_DISABLE+x}" ]]; then
    export NCCL_P2P_DISABLE=1
    echo "[run_multinode] NCCL_P2P_DISABLE=1 (CUDA_VISIBLE_DEVICES isolation makes P2P unavailable)"
fi

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
echo "[run_multinode] ROOT=${ROOT}"
echo "[run_multinode] CONFIG=${CONFIG}"
echo "[run_multinode] NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[run_multinode] NPROC_PER_NODE=${NPROC_PER_NODE} (world_size=${WORLD_SIZE})"

exec torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ${TORCHRUN_EXTRA} \
    scripts/train.py --config "$CONFIG" "$@"
