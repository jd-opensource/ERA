# joyrlinf 设计文档

| 文档 | 内容 |
|------|------|
| [communication_design.md](communication_design.md) | Process Group、comm v1/v2、权重广播、数据通路 |
| [memory_design.md](memory_design.md) | Rollout/Actor 显存预算、PhysX vs PyTorch、调参 |
| [async_pipeline_design.md](async_pipeline_design.md) | Sync/Async 调度、线程模型、重叠与权重陈旧度 |
| [sequential_sync_design.md](sequential_sync_design.md) | 共置全卡 Sync（`pipeline_mode: sync`）、phase 状态机；混合/分离仅 async |
| [multinode_design.md](multinode_design.md) | 多机 torchrun、placement 平铺、NCCL/Gloo、2 节点配置 |
| [observability_and_checkpoint_design.md](observability_and_checkpoint_design.md) | TensorBoard、Eval、Checkpoint、断点续训 |
| [fix_openvla_sync_fsdp.md](fix_openvla_sync_fsdp.md) | OpenVLA sync + `use_orig_params` FSDP writeback 修复（`546ad0e` / `e8e7dd7`） |

快速入口见仓库根目录 [README.md](../README.md)。
