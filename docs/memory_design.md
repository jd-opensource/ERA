# 显存设计

本文说明 co-located **Env + Rollout + OpenPI 推理** 同卡时的 VRAM 构成、预算机制与调参方法。Actor 训练在独立 GPU 上，显存模型不同但遵循相同 `memory:` 配置段。

---

## 1. 显存分区（Rollout GPU）

单张 Rollout GPU 上典型占用：

```text
┌──────────────────────────────────────────────────────────┐
│  Total VRAM (e.g. H800 143GB)                             │
├──────────────────────────────────────────────────────────┤
│  PhysX / SAPIEN GPU sim (train env)     ← 随 env 数近线性 │
│  PhysX (eval env, lazy)                 ← eval 开启后叠加   │
│  OpenPI 推理 (weights + activations)    ← predict_micro_batch │
│  PyTorch caching allocator (pt_rsv)     ← 推理中间张量      │
│  CUDA driver / 碎片 / 未归类 (non_pt)                        │
│  空闲 (free)                          ← safety/critical 监控 │
└──────────────────────────────────────────────────────────┘
```

**Actor GPU**：主要是 FSDP 模型 + optimizer states + 训练 activation；`sharding_strategy: no_shard` 时单卡需容纳完整可训练参数（Pi0.5 约数十 GB 级）。

---

## 2. 实测基线（H800，co-located）

来自 `configs/maniskill_grpo_pi05_async_test1.yaml` 注释（4 rollout GPU 环境）：

| 配置 | PhysX | pt_rsv (mbs=64) | 约 free |
|------|-------|-----------------|---------|
| 384 env/GPU | ~62 GB | ~41 GB | ~40 GB |
| 512 env/GPU | ~81 GB (估) | ~41 GB | ~21 GB |
| 640 env/GPU | ~100 GB (估) | — | **不安全** |

经验公式：

- PhysX：约 **+148 MiB / env**（64→384 线性外推）
- `predict_micro_batch_size`：64 → pt_rsv ~41GB；128 → ~48GB；192 → ~55GB

---

## 3. 配置项 `memory:`

```yaml
memory:
  pytorch_fraction: 0.0      # 0=不硬 cap PyTorch；>0 则限制 caching allocator 上限
  env_reserved_mib: 0        # PhysX 预期峰值（ advisory，用于日志）
  safety_margin_mib: 8192    # free < 此值 → WARNING
  critical_margin_mib: 2048  # free < 此值 → 可 abort rollout
  gc_threshold: 0.8          # PYTORCH_CUDA_ALLOC_CONF
  max_split_size_mb: 128
  expandable_segments: true
```

实现：`joyrlinf/core/gpu_memory_guard.py`，在 `scripts/train.py` 启动时 `apply_memory_budget_from_cfg()`。

### 3.1 设计原则

| 原则 | 说明 |
|------|------|
| **预防优于回收** | 热路径不调用 `empty_cache()`（会破坏 cuBLAS workspace，推理变慢） |
| **软监控** | `torch.cuda.mem_get_info()` 低开销 snapshot，在 rollout chunk 间 `check()` |
| **可选硬 cap** | `pytorch_fraction > 0` 时用 `set_per_process_memory_fraction` 给 PhysX 留固定空间 |

当前 test1 推荐 **`pytorch_fraction: 0.0`**：大 env 场景下靠 `gc_threshold` 回收碎片，而不强行压低 PyTorch 上限。

---

## 4. PhysX `gpu_memory_config`

在 `env.train.init_params.sim_config.gpu_memory_config`（eval 可单独一份）：

- 这些是 **预分配容量**，不是“最多 env 数”的直接上限，但过小会在高 env 下触发 CUDA error 700。
- test1 使用相对默认 **8×** 的 buffer（`found_lost_pairs_capacity`、`heap_capacity` 等）。
- **仅当** 提高 `total_num_envs` 且出现 PhysX OOM/700 时再加大；eval env 数较少时可沿用 train 配置或略减。

---

## 5. OpenPI 推理显存

| 旋钮 | 作用 |
|------|------|
| `actor.model.predict_micro_batch_size` | 将 env 分块做 VLM forward；↑ 延迟 ↓，pt_rsv ↑ |
| `rollout_epoch` / `max_steps_per_rollout_epoch` | 影响单步轨迹张量峰值（时间维 × batch） |
| `num_action_chunks` | 与模型 action horizon 对齐，影响 head 输出形状 |

---

## 6. Eval 对显存的影响

| 组件 | 行为 |
|------|------|
| `env.train` | 启动即建，常驻 |
| `env.eval` | **lazy** 首次 eval 创建，之后常驻 |
| 模型 | train/eval 共用同一 OpenPI 实例 |

**峰值** ≈ train PhysX + eval PhysX + 模型 + pt_rsv。

缓解（配置层，无需改代码）：

- 减小 `env.eval.total_num_envs`
- 增大 `runner.val_check_interval`
- 训练阶段设 `val_check_interval: -1`，离线 eval
- eval 使用更小的 `gpu_memory_config`（若 eval env 更少）

---

## 7. Actor 侧显存

| 项 | 说明 |
|------|------|
| FSDP `no_shard` | 每 actor GPU 持有一份完整参数 + optimizer（2× actor 则各一份） |
| `micro_batch_size` / `gradient_accumulation_steps` | 单次 forward 的 micro-batch 与梯度累积步数；`per_actor_batch = mbs × ga` |
| `gradient_checkpointing` | 配置片段中保留字段，当前 `fsdp_trainer` 未接入 |
| `enable_offload` | 可选 CPU offload（当前 test1 为 false） |

comm v2 下每个 actor 只收到 **部分 env 维** 的 batch，单卡 activation 随 `total_num_envs / (n_rollout × n_actor)` 缩放。

---

## 8. 调参流程建议

1. **固定** `predict_micro_batch_size=64`，从 **384 env/GPU** 起扫 `total_num_envs`。
2. 观察日志中 `GPUMemoryGuard` / rollout `log_profile` 的 `free` 是否持续 > `safety_margin_mib`。
3. 需要吞吐时再升 `predict_micro_batch_size`，同时监控 pt_rsv。
4. 开启 eval 后单独跑一步 eval，确认 lazy env 后 free 仍 > `critical_margin_mib`。
5. OOM 时优先 **减 env 数** 或 **减 eval env**，再考虑 `pytorch_fraction` 硬 cap。

---

## 9. 多机注意

- 每节点 rollout GPU 数相同且 **每 GPU env 数相同** 时，各机显存画像一致。
- `placement_basis: per_node` 不改变单机 env/GPU，只扩展 rank 数（见 multinode 文档）。
- 跨机 **无** rollout batch NCCL，故无额外跨机 GPU 通信缓冲。

---

## 10. 代码索引

| 模块 | 路径 |
|------|------|
| 预算与监控 | `joyrlinf/core/gpu_memory_guard.py` |
| 启动注入 | `scripts/train.py` |
| Rollout 内 check | `joyrlinf/workers/rollout_worker.py` |
| FSDP 训练后清理 | `joyrlinf/training/fsdp_trainer.py` (`_clear_memory`) |
| lazy eval env | `rollout_worker._ensure_eval_env` |
