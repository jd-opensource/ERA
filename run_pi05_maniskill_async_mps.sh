# env rollout_infer actor_train 全共置异步
set -euo pipefail
torchrun --nproc_per_node=8 scripts/train_async_mps.py --config configs/experiments/maniskill_grpo_pi05_async_mps.yaml