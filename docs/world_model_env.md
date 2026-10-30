# 世界模型环境 (World Model Environment)

## 概述

世界模型环境是基于生成式世界模型的强化学习环境，主要用于在仿真环境中训练和评估智能体。该系统使用 Wan 视频生成模型作为世界模型来模拟环境动态，并通过奖励模型来评估智能体的行为。

## 核心组件

### 1. BaseWorldEnv (基础世界环境)
位于 `base_world_env.py`，提供所有世界模型环境的通用功能和接口。

**主要功能：**
- 环境初始化和配置管理
- 指标跟踪（成功率、回报等）
- 抽象方法定义，强制子类实现核心功能
- 自动重置和终止条件处理

### 2. WanEnv (Wan 世界环境)
位于 `world_model_wan_env.py`，基于 Wan 视频生成模型的具体实现。

**核心特性：**
- **视频生成**: 使用 Wan 视频管道生成环境状态序列
- **奖励预测**: 集成奖励模型评估智能体行为
- **批量处理**: 支持多环境并行处理
- **内存管理**: 智能的模型加载/卸载机制

**关键参数：**
- `num_inference_steps`: 推理步骤数量
- `chunk`: 动作块大小（默认8）
- `condition_frame_length`: 条件帧长度（默认5）
- `image_size`: 图像尺寸（默认256x256）

### 3. NpyTrajectoryDatasetWrapper (轨迹数据集包装器)
位于 `world_model_dataset.py`，处理 .npy 格式的轨迹数据。

**功能特点：**
- 支持多种帧选择策略（首帧、末帧、最近时间戳等）
- 数据格式转换（从 npy 到 LeRobot 格式）
- 图像预处理和变换
- 动作和状态提取

## 接口契约

下面所有项均来自实际调用点（rollout_worker / env_worker / 算法层），
遗漏任意一项都会破坏现有流水线。

### 属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `num_envs` | `int` | 当前 rank 上的并行 env 数 |
| `device` | `torch.device` | env 张量所在 CUDA 设备 |
| `elapsed_steps` | `Tensor[B] int64 GPU` | 每个 env 的当前 step 计数 |
| `instruction` | `list[str]`（长度 = `num_envs`）| 每个 env 的语言指令 |
| `total_num_group_envs` | `int` | 初始状态库大小，用于 `update_reset_state_ids` |
| `segment_success_once` | `Tensor[B] bool GPU` | 当前 rollout 段内是否成功过；rollout_worker 用 `.sum().item()` 上报 |
| `returns` | `Tensor[B] float32 GPU` | 累计回报（用于日志） |
| `prev_step_reward` | `Tensor[B] float32 GPU` | 相对 reward 计算用 |

### 方法

| 方法 | 形参 / 返回值 | 备注 |
| --- | --- | --- |
| `update_reset_state_ids()` | → `None` | 每个 rollout epoch 开始时 rollout/env worker 调用一次；保证 group_size 个连续 env 共享同一 state_id（GRPO group baseline 的前提） |
| `reset(*, options=None)` | → `(obs_dict, info)` | `options["env_idx"]` 不为空时仅重置子集（auto-reset 用） |
| `step(actions, auto_reset=True)` | `actions:Tensor[B,A]` → `(obs, reward[B], term[B], trunc[B], info)` | 单 sub-step；`auto_reset=False` 时禁用内部 reset |
| `chunk_step(chunk_actions)` | `Tensor[B,K,A]` → `(obs, [B,K], [B,K], [B,K], info)` | 把 K 个 sub-step 折叠并 stack；`done` 折叠到最后一个 sub-step |
| `wrap_obs(raw_obs=None)` | → obs dict | disagg 模式下 EnvWorker 显式调用，需要返回符合 §2.3 schema 的 dict |

### obs dict

```python
{
    "main_images":       Tensor[B, H, W, 3] uint8 GPU,   # 必填
    "states":            Tensor[B, dof]     fp32  GPU,   # 必填
    "task_descriptions": list[str]  长度 == B,           # 必填
    "wrist_images":      Optional[Tensor[B, H, W, 3] uint8 GPU],
    "extra_view_images": Optional[Tensor[B, H, W, 3] uint8 GPU],
}
```

下游 [`OpenPiModel`](../joyrlinf/models/openpi_model.py) 直接消费上述 key；改 key 名 = 改 model 接口。

### info 必含字段（由 rollout 消费）

```python
info["_final_observation"]: Tensor[B] bool   # auto-reset 标记
info["final_observation"]:  obs dict         # 终止前一帧 obs
info["episode"]:            { "return", "success_once", "episode_len" }
info.get("success"):        Tensor[B] bool   # 用于 segment_success_once 累计
```

### reward / done 语义

- `term`：任务成功 / 失败导致的终止
- `trunc`：超时（`elapsed >= max_episode_steps`）
- `cfg.use_rel_reward=True`：使用相对 reward `r_t - r_{t-1}`（与 ManiSkill 一致）
- `cfg.ignore_terminations=True`：屏蔽 success/fail 终止，只用 trunc


## 环境使用方法

### 环境初始化

```python
from joyrlinf.envs.world_model.world_model_wan_env import WanEnv

# 配置参数
cfg = {
    "model_path": "path/to/wan/model",
    "VAE_path": "path/to/vae/model",
    "reward_model": {
        "type": "ResnetRewModel",
        "from_pretrained": "path/to/reward/model"
    },
    "initial_image_path": "path/to/trajectory/data",
    "num_inference_steps": 50,
    "chunk": 8,
    "condition_frame_length": 5,
    "image_size": [256, 256],
    "seed": 42
}

# 创建环境
env = WanEnv(cfg, num_envs=8, seed_offset=0)
```

### 环境交互

```python
# 重置环境
obs, info = env.reset()

# 执行动作块
actions = np.random.randn(8, 8, 7)  # [num_envs, chunk, action_dim]
obs, rewards, terminations, truncations, infos = env.chunk_step(actions)
```


## 奖励模型

支持两种奖励模型：

1. **ResnetRewModel**: 基于 ResNet 的奖励模型
2. **TaskEmbedResnetRewModel**: 带任务嵌入的 ResNet 奖励模型

## 内存优化

系统实现了智能的内存管理：

- **offload()**: 将模型和数据移动到 CPU
- **onload()**: 将模型和数据加载到 GPU
- 自动清理 CUDA 缓存

## 配置参数

### 环境配置
- `use_fixed_reset_state_ids`: 是否使用固定重置状态ID
- `group_size`: 环境分组大小
- `max_episode_steps`: 最大回合步数
- `success_reward_threshold`: 成功奖励阈值

### 模型配置
- `num_inference_steps`: 推理步骤数
- `cfg_scale`: 分类器自由引导强度
- `retain_action`: 是否保留动作历史

## 注意事项

1. 确保有足够的 GPU 内存来加载 Wan 模型和 VAE
2. 轨迹数据需要按照指定格式组织
3. 奖励模型需要与任务类型匹配
4. 建议根据可用内存调整 `num_envs`

