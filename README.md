# joyrlinf

Embodied RL post-training（OpenPI / GR00T / OpenVLA-OFT + ManiSkill / LIBERO）。支持 co-located / 分离部署、sync / async 调度、NCCL comm v1（P2P）与 v2（all-to-all）。

## 环境

镜像内通过 `switch_env` 切换 venv，**模型类型与 venv 必须对应**：

| venv | 适用模型 / 场景 |
|------|----------------|
| `openpi` | Pi0 / Pi0.5（ManiSkill、LIBERO、WAN 世界模型） |
| `gr00t` | GR00T 1.5（ManiSkill WidowX / Panda、LIBERO Franka） |
| `openvla-oft` | OpenVLA-OFT |
| `wan` | WAN 世界模型 + LIBERO |

```bash
# OpenPI（ManiSkill / LIBERO）
docker load < /ufs_500T/gyc/joyrlinf_v1.0.tar.gz
docker run XXXX
source switch_env openpi

# GR00T（ManiSkill / LIBERO）
source switch_env gr00t
export MUJOCO_GL=egl   # LIBERO 需 EGL

# OpenVLA-OFT（ManiSkill）
source switch_env openvla-oft
```

LIBERO 额外环境变量（MuJoCo EGL）：

```bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0   # 与 rollout 本地 GPU 一致
```

## 快速开始

```bash
git clone git@coding.jd.com:AI_infra_training/joyrlinf.git
cd joyrlinf
export PYTHONPATH=.

# 1) 离线检查配置（batch 整除、comm_mode、placement）
python scripts/validate_embodied_config.py --config configs/maniskill_grpo_pi05_async_test1.yaml
python scripts/validate_embodied_config.py --config configs/maniskill_grpo_gr00t_async_test1.yaml
python scripts/validate_embodied_config.py --config configs/maniskill_grpo_openvla_oft_async.yaml
python scripts/validate_embodied_config.py --config configs/maniskill_grpo_openvla_oft_sync.yaml
python scripts/validate_embodied_config.py --config configs/libero_grpo_pi05.yaml
python scripts/validate_embodied_config.py --config configs/libero_grpo_gr00t_long_sync.yaml
python scripts/validate_embodied_config.py --config configs/maniskill_grpo_gr00t_sync.yaml

# 2) shape 冒烟（chunk_level / action_level，无需 GPU）
python scripts/test_batch_layout.py

# 3) 单节点 8 GPU 训练
source switch_env openpi   # 或 gr00t
torchrun --nproc_per_node=8 scripts/train.py --config configs/maniskill_grpo_pi05_async_test1.yaml
# 或
bash run.sh
```

## 配置速查

主要改 YAML：`total_num_envs`、`micro_batch_size`、`gradient_accumulation_steps`、`predict_micro_batch_size`（VLA rollout 推理）、`env.train.infer_micro_batch_size`（WAN 世界模型推理）、`algorithm.comm_mode`（`v1` / `v2`）、`algorithm.pipeline_mode`（共置 `sync` / 分离 `async`）、`cluster.component_placement`。

### 推荐配置

| 场景 | 配置文件 | venv | 说明 |
|------|----------|------|------|
| ManiSkill + Pi0.5 async | `configs/maniskill_grpo_pi05_async_test1.yaml` | openpi | 默认实验线，`comm_mode: v2` |
| ManiSkill + Pi0.5 sync | `configs/maniskill_grpo_pi05_sync.yaml` | openpi | 全卡共置 sync |
| ManiSkill + Pi0.5 sync profile | `configs/maniskill_grpo_pi05_sync_profile.yaml` | openpi | sync + torch profiler 示例 |
| ManiSkill + GR00T async | `configs/maniskill_grpo_gr00t_async_test1.yaml` | **gr00t** | WidowX，`comm_mode: v2` |
| ManiSkill + GR00T sync | `configs/maniskill_grpo_gr00t_sync.yaml` | **gr00t** | 8 卡共置 sync |
| LIBERO + GR00T async | `configs/libero_grpo_gr00t_long.yaml` 等 | **gr00t** | long / spatial / goal |
| LIBERO + GR00T sync | `configs/libero_grpo_gr00t_long_sync.yaml` 等 | **gr00t** | 8 卡共置，需 `MUJOCO_GL=egl` |
| ManiSkill + OpenVLA-OFT async | `configs/maniskill_grpo_openvla_oft_async.yaml` | **openvla-oft** | rollout 0–3 / actor 4–7，`action_level` + `token_level` |
| ManiSkill + OpenVLA-OFT sync | `configs/maniskill_grpo_openvla_oft_sync.yaml` | **openvla-oft** | 8 卡共置 sync，`env == rollout == actor` |
| LIBERO-90 + Pi0.5 | `configs/libero_grpo_pi05.yaml` | openpi | `comm_mode: v1`，需 LIBERO 权重 |
| LIBERO-130 + Pi0.5 | `configs/libero_grpo_pi05_130.yaml` | openpi | 全 suite，env 数较少 |
| WAN + LIBERO spatial | `configs/wan_libero_spatial_grpo_pi05_async_test1.yaml` | wan | 世界模型 rollout |
| 2 节点 × 8 GPU | `configs/maniskill_grpo_pi05_async_2node.yaml` | openpi | `placement_basis: per_node` |

### chunk_level shape 约定

`reward_type` / `logprob_type` 为 `chunk_level` 时（上述 GRPO 配置默认如此）：

- Rollout 张量：`rewards` / `dones` 为 `[T, B, C]`（C = `num_action_chunks`，通常 5）
- 训练展平后：`advantages` / `loss_mask` 为 `[N]`，N = T × B
- `prev_logprobs`：OpenPI / GR00T（`joint_logprob=True`）为 `[T, B, C, D]` 或等价的 `[T, B, C×D]`，均由 `embodied_batch_layout` 统一处理

OpenVLA-OFT 使用 `reward_type: action_level` + `logprob_type: token_level`（非 `chunk_level`），`num_action_chunks` 通常为 8；展平后 `advantages` / `loss_mask` 为 `[N, C]`。

启动前跑 `python scripts/test_batch_layout.py` 可快速确认 shape 链路。

### 常见调参注意

- **`auto_reset`**：ManiSkill GRPO 常用 `False`（group 内共享初始状态、每 rollout 一条 episode）。LIBERO 长 horizon 建议 `True`（跨 training step 连续仿真、done 后 partial reset）；训练监控看 `train/success_once` / `train/success_rate`。
- **`filter_rewards`**：GRPO 按组过滤 reward 区间；与 `auto_reset` 独立。
- **batch 整除**：`T_eff × B_actor` 须被 `micro_batch_size × gradient_accumulation_steps`（per-actor batch）整除；改 env 数或 `max_steps_per_rollout_epoch` 后先跑 validate。
- **推理 micro-batch**：`actor.model.predict_micro_batch_size` 作用于 OpenPI / GR00T / OpenVLA-OFT 的 rollout 推理；WAN 环境用 `env.train.infer_micro_batch_size`（`0` = 整批 forward）。
- **模型路径**：YAML 里 `actor.model.model_path` 须在本机存在（如 `/ufs/models/gr00t_1.5/`、`/ufs/xxl/RLinf-Pi05-LIBERO-SFT`、`/ufs_500T/wlq/models/openvla-oft/` + `lora_path`）。

## 启动命令示例

```bash
# ManiSkill + Pi0.5（默认）
source switch_env openpi
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/maniskill_grpo_pi05_async_test1.yaml

# ManiSkill + GR00T
source switch_env gr00t
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/maniskill_grpo_gr00t_async_test1.yaml

# ManiSkill + OpenVLA-OFT async（rollout 0–3，actor 4–7）
source switch_env openvla-oft
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/maniskill_grpo_openvla_oft_async.yaml
# 或
bash run_openvla_maniskill.sh

# ManiSkill + OpenVLA-OFT sync（8 卡共置）
source switch_env openvla-oft
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/maniskill_grpo_openvla_oft_sync.yaml
# 或
bash run_openvla_maniskill_sync.sh

# LIBERO + Pi0.5
source switch_env openpi
export MUJOCO_GL=egl
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/libero_grpo_pi05.yaml

# LIBERO + GR00T（async / sync，suite: long | spatial | goal）
source switch_env gr00t
export MUJOCO_GL=egl
bash run_libero_gr00t.sh long              # async, libero_10 + Long SFT
bash run_libero_gr00t.sh sync long         # sync colocated
bash run_libero_gr00t.sh validate sync long

# ManiSkill + GR00T sync
bash run_gr00t_sync.sh
bash run_gr00t_sync.sh libero long
```

## 训练前自检

```bash
export PYTHONPATH=.

# 配置算术
python scripts/validate_embodied_config.py --config <your_config>.yaml

# tensor shape 链路（CPU）
python scripts/test_batch_layout.py

# LIBERO 环境（不加载模型，首次约 1–3 分钟）
source switch_env openpi
export MUJOCO_GL=egl
python scripts/smoke_test_libero_env.py --suite libero_10 --num-envs 8
```

`validate_embodied_config.py` 在离线模式下会根据 `cluster.gpus_per_node`，或单节点连续 rank（如 0–7 → 8 卡）推断每节点 GPU 数；多节点 `placement_basis: per_node` 须在 YAML 写明 `gpus_per_node` 和 `num_nodes`。

## 训练监控与断点（runner）

默认目录（从头训自动带时间戳，避免互相覆盖）：

`{logger.log_path}/{logger.experiment_name}/runs/YYYYMMDD_HHMMSS/`（含 `tb/`、`config.yaml`、`checkpoints/`、`perf_summary.json`）。

| 配置项 | 说明 |
|--------|------|
| `logger.logger_backends` | `tensorboard`（可选 `wandb`） |
| `logger.histogram_interval` | 每 N 步写 `train/ratio`、`adv/advantages` 直方图；`0` 关闭 |
| `val_check_interval` | `>0` 开启周期性 eval；`≤0` 关闭 |
| `save_interval` | `>0` 周期性存 checkpoint；须能被 `val_check_interval` 整除 |
| `resume_dir` | 续训路径，如 `./results/.../runs/<run_id>/checkpoints/global_step_40`；`null` 从头训 |

续训时设 `resume_dir` 即可继续写该次 run 目录；独立实验线可改 `experiment_name`。

**指标体系（`joyrlinf/observability/`）**

结构化指标经 `MetricsEmitter` 写入 TensorBoard / WandB，并实时刷新 `perf_summary.json`（吞吐峰值、分阶段耗时等）。

| TB 前缀 | 来源 | 示例 |
|---------|------|------|
| `train/` | Actor 训练 | `policy_loss`, `ratio_max`, `entropy`, `proximal_approx_kl` |
| `adv/` | 优势计算 | `adv_mean`, `adv_std`, `values_mean` |
| `data/` | GRPO 诊断 / reward filter | `groups_with_var_pct`, `reward_nonzero_pct` |
| `rollout/` | Rollout 标量 | `success_rate` |
| `eval/` | 周期性 eval | `success_rate`, `mean_return` |
| `time/` | 分阶段计时 | `rollout_send`, `rollout_core`, `actor_train`, `throughput_env_s` |
| `memory/` | 显存守护（sync 共置） | `free_mib`, `critical_events` |

Async 模式下 rollout 标量经 **gloo CPU 专用通道**（`scalar_transport.py`）从 `first_rollout` 发往 metrics rank，不再占用 NCCL batch 带宽；Sync 共置模式在进程内 staging 传递。

**TensorBoard**

```bash
tensorboard --logdir ./results/maniskill_grpo_pi05_async/runs/<时间戳>/tb
```

控制台每步摘要包含 `adv` / `train` / `data` 关键指标；详细日志在 `logs/<时间戳>_<config>_nodeN.log`。

**WandB（可选）**

```yaml
runner:
  logger:
    logger_backends: [tensorboard, wandb]
    histogram_interval: 10
```

开启后自动写入完整 yaml 配置，并用 `define_metric` 将各桶指标与 `global_step` 对齐。

**perf_summary.json**

训练过程中每步原子写入 run 目录下的 `perf_summary.json`（`status: in_progress`），结束后标记 `completed` 并记录峰值吞吐与平均耗时，便于脚本化对比实验。

### Profiler（Chrome trace）

基于 `torch.profiler` 采集训练性能 trace，导出 Chrome trace JSON，可在 [chrome://tracing](chrome://tracing) 或 [Perfetto UI](https://ui.perfetto.dev/) 中查看。

**输出目录**（默认）：`{run_output_dir}/profiler/<role>/`，例如 sync 共置模式下：

```
profiler/sync/step-1_rank-0_role-sync_tag-phase_a_rollout_....trace.json
profiler/sync/step-1_rank-0_role-sync_tag-phase_b_train_....trace.json
```

**配置**（YAML 顶层 `profiler` 段，默认 `enable: false`）：

| 配置项 | 说明 |
|--------|------|
| `enable` | 是否开启 profiler |
| `tool` | 目前支持 `torch` |
| `steps` | 要采集的 **global step** 列表（从 1 起计，与日志 `Step 1/N` 一致） |
| `ranks` / `all_ranks` | 采集哪些 rank（默认仅 rank 0） |
| `roles` | 限制角色：`sync` / `rollout` / `actor` / `env`；空列表表示全部 |
| `profile_continuous_steps` | `steps: [1,2,3]` 时合并为一次连续采集 |
| `save_path` | 自定义保存路径；`null` 则用 `{run_output_dir}/profiler` |
| `tool_config.torch.contents` | `cuda` / `cpu` / `memory` / `shapes` / `stack` |
| `tool_config.torch.discrete` | `true`：每个 `range` 区域单独一个 trace；`false`：整步一个 trace |
| `tool_config.torch.compress` | `true` 输出 `.json.gz`；`false` 输出 `.trace.json` |

**示例**（仅 profile 第 1 步、rank 0，sync 分阶段 trace）：

```yaml
profiler:
  enable: true
  tool: torch
  steps: [1]
  ranks: [0]
  tool_config:
    torch:
      contents: [cuda, cpu, memory]
      discrete: true
```

可直接使用示例配置：

```bash
source switch_env openpi
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/maniskill_grpo_pi05_sync_profile.yaml
```

**step 编号说明**：`max_steps: 2` 时训练循环为 step `1` 和 `2`（`range(1, max_steps+1)`）。`steps: [1]` 表示只采集第一步；采集第二步写 `steps: [2]`；两步都采集写 `steps: [1, 2]`。

**Rollout 内部细粒度标记（区分 env 与推理）**

`phase_a_rollout` 之前只能看到 rollout 整体耗时。现在 rollout 内循环与 env 内部都打了 `torch.profiler.record_function` 标记，在同一份 trace 中即可分离（无 profiler 时开销可忽略）：

| trace 标签 | 位置 | 含义 |
|-----------|------|------|
| `rollout/reset` | rollout_worker | 每个 epoch 首次 env.reset |
| `rollout/infer_predict` | rollout_worker | VLA 推理（`model.predict`），逐 chunk |
| `rollout/prepare_actions` | rollout_worker | 动作后处理 / 反归一化 |
| `rollout/env_chunk_step` | rollout_worker | env 交互整体（含物理+渲染） |
| `rollout/prepare_obs` | rollout_worker | obs 组装给模型 |
| `rollout/bootstrap_predict` | rollout_worker | 末步 bootstrap 推理 |
| `rollout/to_cpu` | rollout_worker | buffer 拷回 CPU |
| `env/sim_and_render` | maniskill_env | PhysX 物理 + 相机渲染（ManiSkill 内融合，无法在 wrapper 层拆分） |
| `env/wrap_obs` | maniskill_env | 从原始 obs 提取图像/本体状态 |
| `env/reward_metrics` | maniskill_env | reward 计算与指标累计 |
| 分离部署：`rollout/send_actions` `rollout/recv_step` / `env/recv_actions` `env/chunk_step` `env/send_results` | rollout/env worker | env≠rollout 时的 NCCL 收发与 env 交互 |

**控制台 breakdown（无需打开 trace）**：每个 rollout epoch 结束时，leader rank 打印一行 `[ROLLOUT-BREAKDOWN]`，含各段耗时、占比，以及 `infer/env` 比值。默认用低开销墙钟计时（依赖 predict 返回 numpy 时的自然同步点，为近似值）。需要**精确**的分段 GPU 耗时时设：

```bash
export JOYRLINF_ROLLOUT_TIMING=1   # 每段后插 cuda.synchronize()，会轻微扰动流水线，仅用于测量
```

判断"单卡 env/rollout 异步流水线是否值得做"时，重点看 `infer/env` 比值与 `env/sim_and_render`、`env/wrap_obs` 是否留有空闲 SM（结合 trace 中推理 kernel 的 GPU 占用）。

**Eval**

- 需配置 `env.eval`（如 `obj_set: test`、较小的 `total_num_envs`）。
- `total_num_envs` 须能被 rollout GPU 数整除。
- Eval 环境 lazy 创建，不占用训练启动显存。

**Checkpoint**

```
results/<experiment_name>/checkpoints/global_step_<N>/actor/
  model.pt  optimizer.pt  scheduler.pt  meta.json
```

**续训**

```yaml
runner:
  resume_dir: ./results/maniskill_grpo_pi05_async/checkpoints/global_step_40
```

从该步继续训练至 `max_steps`；TB 步数与 OpenPI `global_step` 连续。

## 多节点

```bash
# 2 节点 × 8 GPU
NNODES=2 NODE_RANK=0 MASTER_ADDR=<ip> NPROC_PER_NODE=8 NCCL_SOCKET_IFNAME=bond0 \
  ./scripts/run_multinode.sh configs/maniskill_grpo_pi05_async_2node.yaml
NNODES=2 NODE_RANK=1 MASTER_ADDR=<ip> NPROC_PER_NODE=8 NCCL_SOCKET_IFNAME=bond0 \
  ./scripts/run_multinode.sh configs/maniskill_grpo_pi05_async_2node.yaml
```

多节点配置使用 `placement_basis: per_node`，本地 rank 只写一次（如 `env/rollout: 0-5`，`actor: 6-7`），并设置 `num_nodes` 与 `gpus_per_node`。

## 调试与其它

```bash
export JOYRLINF_DEBUG=1              # 详细日志
```

NCCL 连通性测试（两节点各执行一条，改 `MASTER_ADDR` / `NODE_RANK`）：

```bash
NCCL_SOCKET_IFNAME=bond0 torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=<ip> --master_port=29500 scripts/test_nccl.py
```

## 设计文档

详见 [docs/](docs/README.md)：

- [通信设计](docs/communication_design.md) — Process Group、comm v1/v2、权重 Gloo 广播
- [显存设计](docs/memory_design.md) — PhysX / PyTorch 预算与调参
- [异步流水线](docs/async_pipeline_design.md) — Sync/Async、线程模型、权重陈旧度
- [多机训练](docs/multinode_design.md) — torchrun、per_node placement、NCCL
- [监控与断点](docs/observability_and_checkpoint_design.md) — TensorBoard、Eval、Checkpoint、续训

## 能力概览

- **模型**：OpenPI Pi0/Pi0.5、GR00T 1.5、OpenVLA-OFT
- **环境**：ManiSkill、LIBERO、WAN 世界模型（LIBERO spatial）
- **部署**：Hybrid / Disaggregated placement
- **调度**：Sync / Async pipeline
- **通信**：comm v1（P2P）、comm v2（all-to-all）
- **可观测性**：结构化 metrics（P0/P1）、TensorBoard / WandB、`perf_summary.json`、gloo 标量通道、torch profiler（Chrome trace JSON）
