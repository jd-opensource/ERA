# Isaac-GR00T N1.5 SFT for ManiSkill Panda (joyrlinf RL)

Requires SFT checkpoint from **Isaac-GR00T-n1.5-release** — see
[examples/ManiSkillPanda/README.md](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release/examples/ManiSkillPanda)
(or your fork's copy).

## Dataset

HuggingFace: `rechim25/maniskill_peginsertionside_240`

Follow the Isaac-GR00T ManiSkillPanda README for `modality.json`, `fix_stats.py`, and validate.

## SFT (Isaac-GR00T repo)

```bash
cd /path/to/Isaac-GR00T-n1.5-release
pip install -e . --no-deps
export WANDB_MODE=offline

python scripts/gr00t_finetune.py \
  --dataset-path /path/to/datasets/maniskill_peginsertionside_240 \
  --data-config maniskill_panda \
  --embodiment-tag maniskill_panda \
  --video-backend torchvision_av \
  --base-model-path /path/to/models/gr00t_1.5 \
  --output-dir /path/to/models/gr00t_1.5_sft_peginsertion \
  --num-gpus 1 \
  --batch-size 32 \
  --max-steps 3000 \
  --save-steps 1000
```

Required output: `experiment_cfg/metadata.json` with key `maniskill_panda`.
Embodiment projector index: **31** (must match `joyrlinf/models/gr00t/embodiment_tags.py`).

## RL (joyrlinf)

```bash
# Edit configs/maniskill_grpo_gr00t_panda_peginsertion.yaml: actor.model.model_path
torchrun --nproc_per_node=8 scripts/train_gr00t.py \
  --config configs/maniskill_grpo_gr00t_panda_peginsertion.yaml
```

Configs: `configs/maniskill_grpo_gr00t_panda_peginsertion.yaml`, `configs/model/gr00t_maniskill_panda.yaml`
