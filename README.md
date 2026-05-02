# OpenArm VLA RL Deployment

English documentation. 中文文档见 [README_zh.md](README_zh.md).

This repository contains the VLA/RL layer for OpenArm Isaac Lab experiments. The base Isaac Lab extension lives in the `openarm_isaac_lab` submodule; this repo adds `openarm_vla`, DiffIK task variants, PPO train/play scripts, and lightweight data-collection utilities.

## Repository Layout

```text
openarm/
├── openarm_isaac_lab/          # git submodule: base OpenArm Isaac Lab tasks/assets
└── openarm_vla/
    ├── Data_Collector/                     # PPO train/play scripts and action wrappers
    ├── EE_API_Test/            # EE action-channel smoke tests
    ├── VLA_FT/                 # optional SmolVLA fine-tuning helpers
    └── source/openarm_vla/     # installable Python package and Gym registration
```

`lerobot/` is intentionally ignored in this cleanup. Keep it as a local checkout or manage it separately if needed.

## Installation

Clone with submodules, then install both editable packages inside the Isaac Lab environment:

```bash
git clone --recurse-submodules https://github.com/apolloil/openarm-vla-rl-deployment.git
cd openarm-vla-rl-deployment

pip install -e openarm_isaac_lab/source/openarm --no-build-isolation
pip install -e openarm_vla/source/openarm_vla --no-build-isolation
```

If you cloned without submodules:

```bash
git submodule update --init --recursive
```

## RL Train And Play

Run commands from the repository root with the Isaac Lab conda environment active.

```bash
# Lift
python openarm_vla/Data_Collector/train_lift.py
python openarm_vla/Data_Collector/play_lift.py

# Soccer
python openarm_vla/Data_Collector/train_soccer.py
python openarm_vla/Data_Collector/play_soccer.py
```

Useful training overrides:

```bash
python openarm_vla/Data_Collector/train_lift.py --num_envs 2048 --max_iterations 5000 --seed 42
python openarm_vla/Data_Collector/train_soccer.py --num_envs 2048 --max_iterations 4000 --seed 42
python openarm_vla/Data_Collector/train_soccer.py --video --video_interval 500
```

The play scripts are configured by constants at the top of each file, especially `CHECKPOINT_PATH`, `PLAY_GUI`, and video output settings.

## Soccer Pipeline

Soccer training only teaches the policy to bring the ball to `Pre_Goal`. The play script then switches to a deterministic heuristic once the ball is stable near `Pre_Goal`: release/lift the arm, move to `P_kick`, lower to the field, and kick toward the goal. `Pre_Goal` and `P_kick` are rendered as debug markers during play and EE smoke tests.

## Gym IDs

| Task | Training ID | Play ID |
| --- | --- | --- |
| Lift | `Isaac-VLA-Lift-Cube-OpenArm-v0` | `Isaac-VLA-Lift-Cube-OpenArm-Play-v0` |
| Soccer | `Isaac-VLA-Soccer-OpenArm-v0` | `Isaac-VLA-Soccer-OpenArm-Play-v0` |

## Smoke Tests

```bash
python openarm_vla/EE_API_Test/test_ee_dims_video.py
python openarm_vla/EE_API_Test/test_soccer_ee_dims_video.py
```

Generated videos, logs, checkpoints, exports, and dataset shards are ignored by git.

## Submodule Push Order

When changes touch `openarm_isaac_lab`, commit and push the submodule first:

```bash
cd openarm_isaac_lab
git status
git add <changed files>
git commit -m "..."
git push origin main
cd ..
```

Then commit the main repository, including the updated submodule pointer:

```bash
git status
git add .gitignore README.md README_zh.md openarm_vla openarm_isaac_lab
git commit -m "..."
git push origin main
```
