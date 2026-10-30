# 通信设计

本文描述 joyrlinf embodied 训练中 **Rollout ↔ Actor**（及 Disaggregated 下 **Env ↔ Rollout**）的通信拓扑、后端选择与实现约束。核心代码：`joyrlinf/core/launcher.py`、`joyrlinf/core/direct_pipe.py`、`joyrlinf/runners/embodied_runner.py`。

---

## 1. 设计原则

| 原则 | 说明 |
|------|------|
| **大流量走 NCCL（GPU）** | Rollout batch（观测、动作、logprob 等）体量大，使用 NCCL P2P 或 `all_to_all_single` |
| **小标量走 Gloo（CPU）** | 权重广播、shutdown barrier、success/eval 聚合，避免 NCCL 在大 world 上为少量数据占用 GPU 代理缓冲 |
| **通道分离** | Async 下 **batch 通道** 与 **weight 通道** 使用不同 Process Group，主线程与后台线程可并发 |
| **多机 batch 本地化** | `comm_mode: v2` 时 all2all **按节点**建组，跨机梯度仍由 FSDP `actor` 组 all-reduce |

---

## 2. 角色与数据通路

### 2.1 Co-located（当前 ManiSkill 默认）

`env_ranks == rollout_ranks`（如 GPU 0–5）：

```text
┌─────────────────────────────────────┐
│  Rollout rank (同进程)               │
│  ManiSkill env  ←→  OpenPI 推理      │  零拷贝函数调用
└──────────────┬──────────────────────┘
               │ NCCL (v1 P2P 或 v2 all2all)
               ▼
┌─────────────────────────────────────┐
│  Actor rank (FSDP 训练)              │
└──────────────┬──────────────────────┘
               │ Gloo CPU broadcast（权重）
               ▼
         回传到本节点 Rollout
```

### 2.2 Disaggregated

`env_ranks != rollout_ranks`：Env 与 Rollout 之间用 **NCCL P2P**（`NCCLPipe` / `env_rollout_pair_*` 组）；Rollout → Actor 与 co-located 相同。

---

## 3. Process Group 一览

由 `create_process_groups(placement, comm_mode=...)` 创建：

| Group 名 | Backend | 成员 | 用途 |
|----------|---------|------|------|
| `actor` | NCCL | 全部 actor ranks | FSDP backward all-reduce（可跨节点） |
| `rollout` | Gloo | 全部 rollout ranks | success / eval 标量 `all_reduce` |
| `pair_{r}_{a}` | NCCL | 一对 rollout+actor | **comm v1** batch P2P |
| `_a2a_node_groups[node]` | NCCL | 同节点 rollout+actor | **comm v2** `all_to_all_single` |
| `_wb_node_groups[node]` | Gloo | 同节点 rollout + 首个 actor | **权重广播**（每节点一个 src_actor） |
| `env_rollout_pair_*` | NCCL | env+rollout 对 | Disaggregated env 数据 |
| `shutdown` | Gloo | 全体参与 rank | 退出 barrier（避免 NCCL 600s watchdog） |

**权重广播源**：每节点 `actor_ranks` 中第一个（如单节点 8 GPU 上 actor 6–7，则 `wb_src_actor=6`）。

---

## 4. comm v1：配对 P2P

配置：`algorithm.comm_mode: v1`

- 每个 actor 与 **一个** rollout rank 配对（`get_paired_rollout`）。
- 发送：`NCCLPipe.send_dict` — 先 meta/header，再逐 tensor `dist.send`。
- 多 rollout → 多 actor 时，每个 actor 只收自己配对 rollout 的完整 batch（无 env 维切分）。

**适用**：单 rollout GPU 对单 actor GPU，或拓扑简单、batch 不需在 actor 间切分 env 维。

---

## 5. comm v2：节点内 All-to-All

配置：`algorithm.comm_mode: v2`（推荐多 rollout × 多 actor）

### 5.1 组布局

同节点 ranks 排列为：

```text
[ rollout_0, rollout_1, ..., rollout_{N-1}, actor_0, actor_1, ..., actor_{M-1} ]
  group_rank 0 .. N-1                              group_rank N .. N+M-1
```

实现：`direct_pipe.all_to_all_batch()`。

### 5.2 切分语义

- Rollout 侧：每个 tensor 沿 **`split_dim=1`**（batch/env 维）切成 `n_actors` 份，经 **一次** `all_to_all_single` 发给各 actor。
- Actor 侧：从每个 rollout 收一块，沿同一维 **concat**，得到本地训练 batch 片段。
- 约束：`shape[split_dim] % n_actors == 0`（由 `validate_embodied_training_config` 与 `total_num_envs` 保证）。

### 5.3 元数据

Rollout group rank 0 先 `broadcast_object_list` JSON schema（key、shape、dtype），再按 key 做 all2all，避免多次小 P2P 握手。

### 5.4 多机行为

- **Batch 交换不跨机**：仅 `_a2a_node_groups[my_node]`。
- **梯度同步跨机**：FSDP 在全局 `actor` NCCL 组上 all-reduce。
- 若某节点只有 rollout 或只有 actor，该节点 **跳过** all2all 组（打 warning）；此类拓扑需改用 v1 或调整 placement。

### 5.5 Warmup

启动时 `warmup_all2all()` 用极小 dummy tensor 预建 NCCL 通信器，减少首步延迟。

---

## 6. 权重同步（Gloo / CPU）

### 6.1 为何不用 NCCL 传权重

ManiSkill `env.reset()` 等会触发 `cudaDeviceSynchronize()`，与 NCCL CUDA stream 争用；权重改为 **CPU tensor + Gloo broadcast**。

### 6.2 打包优化

可训练参数 **flatten 为单个连续 buffer**，一次 collective + 一次 meta broadcast，替代数百次 per-tensor broadcast（见 `embodied_runner._broadcast_weights_from_sd`）。

### 6.3 Sync 模式

每步 actor 训练前/后，rollout 主线程 `_sync_weights_from_actor()` 阻塞接收。

### 6.4 Async 模式

| 端 | 线程 | 行为 |
|----|------|------|
| Rollout | `weight_recv_loop`（后台） | 唯一调用 `_recv_weights_raw()` |
| Rollout | 主线程 | `load_weights(pending_weights)`，**禁止** `_sync_weights_from_actor()` |
| Actor src | `weight_broadcast_loop`（后台） | `training_done_evt` 后 `get_state_dict` → broadcast |
| Actor | 主线程 | recv batch → train → `training_done_evt.set()` |

**死锁教训**：Async eval 若主线程再 `_sync_weights_from_actor()`，会与 `weight_recv_loop` 双消费同一 Gloo broadcast → 长时间 hang。

---

## 7. Runner 指标随 batch 传递（comm v2）

标量 metrics（`rollout/success_rate`、`eval/*`、`time/rollout_*`）经 **P1 标量通道**（`scalar_transport.py`，gloo CPU `dist.send`）从 `first_rollout` 发往 `wb_src_actor`；不再随 batch all2all。旧版 `_runner_metric_*` batch 捎带仍可由 `strip_runner_metrics()` 兼容剥离，**不参与 loss**。

---

## 8. 配置对照

```yaml
algorithm:
  comm_mode: v2          # v1 | v2
  pipeline_mode: async   # async：混合/分离；sync：仅共置全卡（见 sequential_sync_design.md）

cluster:
  component_placement:
    actor: 6-7
    env: 0-5
    rollout: 0-5
```

离线校验：

```bash
python scripts/validate_embodied_config.py --config configs/maniskill_grpo_pi05_async_test1.yaml
```

---

## 9. 调试

| 现象 | 排查 |
|------|------|
| 600s NCCL timeout | Async 下 rollout 先结束、actor 仍 recv；检查 send/recv 顺序、勿在 eval 主线程双收权重 |
| `all_to_all` shape 错误 | `total_num_envs` 与 actor 数、每 GPU env 数不整除 |
| 多机 all2all 无组 | 节点上需同时有 rollout 与 actor ranks |
| Gloo 连到 docker0 | 设置 `GLOO_SOCKET_IFNAME` 与 `NCCL_SOCKET_IFNAME` 一致（见 multinode 文档） |

```bash
export JOYRLINF_DEBUG=1
```

---

## 10. 代码索引

| 功能 | 位置 |
|------|------|
| 建组 | `launcher.create_process_groups` |
| v1 P2P | `direct_pipe.NCCLPipe` |
| v2 all2all | `direct_pipe.all_to_all_batch` |
| 发送/接收 batch | `EmbodiedRunner._send_batch_v2` / `_recv_batch_v2` |
| 权重收发 | `EmbodiedRunner._broadcast_weights*` / `_recv_weights_raw` |
