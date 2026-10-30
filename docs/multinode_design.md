# 多机训练设计

本文描述 joyrlinf 在 **多节点 torchrun** 下的 rank 布局、通信分层、启动脚本与排错。单节点可只读 §2–§3 中的 placement 概念。

---

## 1. 拓扑假设

| 假设 | 说明 |
|------|------|
| **同构集群** | 每节点 GPU 数相同、placement 局部模式相同 |
| **torchrun 启动** | `WORLD_SIZE = NNODES × NPROC_PER_NODE` |
| **全局 rank** | 节点优先（node-major）：node0 占 `0..G-1`，node1 占 `G..2G-1`，… |
| **无 Ray** | 每 rank 一个进程，由 `EmbodiedRunner` 按 role 分支 |

---

## 2. Placement：global vs per_node

### 2.1 `placement_basis: global`（单节点）

直接写全局 CUDA rank：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor: 6-7
    env: 0-5
    rollout: 0-5
```

### 2.2 `placement_basis: per_node`（多节点推荐）

每节点只写 **本地** rank，由 launcher 平铺到全局：

```yaml
cluster:
  placement_basis: per_node
  num_nodes: 2
  gpus_per_node: 8
  component_placement:
    actor: 6-7      # 每节点 local GPU 6–7
    env: 0-5
    rollout: 0-5
```

**2 节点 × 8 GPU** 展开后：

| 节点 | Rollout (global) | Actor (global) |
|------|------------------|----------------|
| 0 | 0–5 | 6–7 |
| 1 | 8–13 | 14–15 |

实现：`launcher._maybe_expand_per_node()`；`num_nodes` 须与 `world_size / gpus_per_node` 一致。

---

## 3. 跨机通信分层

```text
                    Node 0                          Node 1
         ┌─────────────────────────┐      ┌─────────────────────────┐
         │ rollout 0-5             │      │ rollout 8-13            │
         │   │ NCCL all2all (节点内) │      │   │ NCCL all2all (节点内) │
         │   ▼                     │      │   ▼                     │
         │ actor 6-7               │      │ actor 14-15             │
         │   │                     │      │   │                     │
         │   └───────┬─────────────┘      │   └───────┬─────────────┘
         └───────────┼────────────────────┴───────────┼─────────────┘
                     │   NCCL FSDP actor group (跨机)   │
                     └──────────────────────────────────┘

         权重: 每节点 Gloo broadcast (actor_src → 本节点 rollout)，不跨机传权重
         标量: rollout Gloo group 可跨机 all_reduce（仅小数据）
```

| 数据类型 | 跨机？ | 机制 |
|----------|--------|------|
| Rollout batch | **否**（v2） | 节点内 `_a2a_node_groups` |
| 梯度 / FSDP | **是** | `actor` NCCL 组 |
| 推理权重 → rollout | **否** | 每节点 `_wb_node_groups` Gloo |
| 日志时间戳 | **是** | 多机用 Gloo CPU broadcast（`train._sync_run_timestamp`） |

因此 **多机扩展 rollout 算力** 时，每机应有成对的 rollout+actor；仅堆 rollout 而无本机 actor 时 v2 all2all 无法建组。

---

## 4. 启动流程

### 4.1 脚本

`scripts/run_multinode.sh` 封装 torchrun：

```bash
# 节点 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0_ip> NPROC_PER_NODE=8 \
  NCCL_SOCKET_IFNAME=bond0 \
  ./scripts/run_multinode.sh configs/maniskill_grpo_pi05_async_2node.yaml

# 节点 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0_ip> NPROC_PER_NODE=8 \
  NCCL_SOCKET_IFNAME=bond0 \
  ./scripts/run_multinode.sh configs/maniskill_grpo_pi05_async_2node.yaml
```

### 4.2 脚本自动行为

| 变量 | 默认/行为 |
|------|-----------|
| `NCCL_SOCKET_IFNAME` | 多机时按 `MASTER_ADDR` 子网或路由表自动检测 |
| `GLOO_SOCKET_IFNAME` | 与 NCCL 对齐（避免 P2P warmup 走 docker0） |
| `NCCL_IB_DISABLE` | 多机默认 `1`（以太网）；IB 环境显式设 `0` |
| `NCCL_P2P_DISABLE` | 默认 `1`（`CUDA_VISIBLE_DEVICES` 每进程只见一卡） |
| `/etc/hosts` | 修复 hostname 无法解析导致的 c10d 错误 |

### 4.3 `train.py` 多机差异

- `dist.init_process_group(backend="nccl")` 为默认 PG（GPU collective）
- 日志：`logs/<ts>_<config>_node<N>.log`，**每物理节点一个文件**（同节点 8 rank 共享）
- 控制台默认仅 `LOCAL_RANK=0` 打 INFO

---

## 5. 配置示例（2 节点）

`configs/maniskill_grpo_pi05_async_2node.yaml`：

- `total_num_envs: 1152` → 12 rollout GPU × 96 env/GPU（需满足 v2 整除校验）
- `micro_batch_size: 128`、`gradient_accumulation_steps: 4`（per-actor batch = 512）须与 rollout 展平样本数整除
- `comm_mode: v2`、`pipeline_mode: async`

单机 test 配置见 `maniskill_grpo_pi05_async_test1.yaml`（`num_nodes: 1`）。

---

## 6. Batch 与全局 batch 规模

多机时：

\[
\text{envs/GPU} = \frac{\text{total\_num\_envs}}{\text{\#rollout GPUs global}}
\]

\[
B_{\text{actor,local}} = \text{envs/GPU} \times \frac{\text{\#rollout}}{\text{\#actor}} \times T_{\text{eff}}
\]

（\(T_{\text{eff}}\) 为有效时间步，见 `config_validation`）

**要求**：各节点上 per-actor batch 一致，否则 v2 all2all 分片不均。校验在 `validate_embodied_training_config`。

---

## 7. NCCL 连通性测试

两节点各执行（改 IP 与 `NODE_RANK`）：

```bash
NCCL_SOCKET_IFNAME=bond0 torchrun --nnodes=2 --nproc_per_node=1 \
  --node_rank=0 --master_addr=<ip> --master_port=29500 scripts/test_nccl.py
```

通过后再跑完整训练。

---

## 8. 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `num_nodes` 与 world_size 不符 | torchrun 节点数/每节点 GPU 与 YAML 不一致 | 对齐 `NNODES`、`NPROC_PER_NODE`、`gpus_per_node` |
| Gloo 连 169.254.x | 网卡选错 | 显式 `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` |
| 仅 node0 有 all2all 组 | 某节点缺 actor 或 rollout | 调整 placement，保证每节点 0–5 + 6–7 |
| FSDP timeout | 跨机 actor 组 NCCL 不通 | 先 `test_nccl.py`，查防火墙、`MASTER_ADDR` |
| hostname 报错 | 容器 DNS | `run_multinode.sh` 写 `/etc/hosts` |

---

## 9. 与单机的差异小结

| 项 | 单机 | 多机 |
|----|------|------|
| placement | 可直接 global rank | 推荐 `per_node` |
| batch NCCL | 节点内 all2all | 同左，每节点一组 |
| 权重广播 | 单组 Gloo | 每节点一组 Gloo |
| 日志时间戳 | NCCL CUDA broadcast | Gloo CPU broadcast |
| IB | 可选 | 脚本默认禁 IB，按需开启 |

---

## 10. 代码与脚本索引

| 项 | 路径 |
|----|------|
| per_node 平铺 | `joyrlinf/core/launcher.py` |
| 建组（含节点 all2all） | `create_process_groups` |
| 配置校验 | `joyrlinf/core/config_validation.py` |
| 多机启动 | `scripts/run_multinode.sh` |
| 训练入口 | `scripts/train.py` |
| 2 节点样例 | `configs/maniskill_grpo_pi05_async_2node.yaml` |
