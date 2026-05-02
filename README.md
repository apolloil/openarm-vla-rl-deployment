# OpenArm VLA RL Deployment

English documentation. 中文文档见 [README_zh.md](README_zh.md).

This repository contains the RL/VLA workflow for OpenArm in Isaac Lab. The base robot assets and task logic live in the `openarm_isaac_lab` submodule, while `openarm_vla` contains the VLA-facing environments, PPO expert scripts, smoke tests, and optional SmolVLA fine-tuning helpers.

## Repository Layout

```text
openarm/
├── openarm_isaac_lab/          # git submodule: base OpenArm Isaac Lab tasks/assets
└── openarm_vla/
    ├── Data_Collector/         # PPO training, play, wrappers, and dataset collection
    ├── EE_API_Test/            # EE action-channel smoke tests
    ├── VLA_FT/                 # optional SmolVLA fine-tuning scripts
    └── source/openarm_vla/     # Gym registration and VLA environment configs
```

## Installation

Clone with submodules, then install both editable packages inside the Isaac Lab environment:

```bash
git clone --recurse-submodules https://github.com/apolloil/openarm-vla-rl-deployment.git
cd openarm-vla-rl-deployment

pip install -e openarm_isaac_lab/source/openarm --no-build-isolation
pip install -e openarm_vla/source/openarm_vla --no-build-isolation
```

The optional SmolVLA fine-tuning helpers in `openarm_vla/VLA_FT/` require LeRobot; install it separately in the Python environment used for VLA training.

If you cloned without submodules:

```bash
git submodule update --init --recursive
```

## RL Train And Play

Run commands from the repository root with the Isaac Lab conda environment active.

```bash
# Lift
python openarm_vla/Data_Collector/train_lift.py
python openarm_vla/Data_Collector/play_lift.py --checkpoint_path logs/rsl_rl/openarm_lift/<run>/model_3999.pt

# Soccer
python openarm_vla/Data_Collector/train_soccer.py
python openarm_vla/Data_Collector/play_soccer.py --checkpoint_path logs/rsl_rl/openarm_soccer/<run>/model_3200.pt
```

Useful training overrides:

```bash
python openarm_vla/Data_Collector/train_lift.py --num_envs 2048 --max_iterations 5000 --seed 42
python openarm_vla/Data_Collector/train_soccer.py --num_envs 2048 --max_iterations 4000 --seed 42
python openarm_vla/Data_Collector/train_soccer.py --video --video_interval 500
```

The play scripts take the checkpoint path and common playback options from CLI flags such as `--checkpoint_path`, `--video_seconds`, `--scene_preset_id`, `--video_output_dir`, and `--gui`.

## Important Files In `openarm_vla`

Paths below are relative to `openarm_vla/`.

### `Data_Collector`

This is the main folder for RL work. If you only want to train or play expert policies, start here.

| File | What it does |
| --- | --- |
| `train_lift.py` | Trains a PPO expert for the Lift task. It uses a 4D policy action `[dx, dy, dz, grip]` and saves checkpoints under `logs/rsl_rl/openarm_lift/`. |
| `play_lift.py` | Loads a Lift checkpoint from `--checkpoint_path`, records a play video, and exports the policy. Use this to visually inspect a trained Lift policy. |
| `train_soccer.py` | Trains a PPO expert for the Soccer task. The policy learns to bring the ball to `Pre_Goal`; the actual kick is handled during play. |
| `play_soccer.py` | Loads a Soccer checkpoint from `--checkpoint_path`. It first runs the learned policy, then uses a simple scripted sequence to release, move behind the ball, and kick. |
| `collect_dataset.py` | Uses a trained Lift expert to collect successful demonstration episodes for VLA fine-tuning. It writes LeRobot-style dataset shards. |
| `lift_ee_action_wrapper.py` | Converts the Lift policy's 4D output into the 7D DiffIK action used by the environment. |
| `soccer_ee_action_wrapper.py` | Does the same conversion for Soccer, with an extra orientation correction so the gripper does not slowly tilt during contact. |
| `policy_action_vecenv_wrapper.py` | Makes the wrapped Gym environment look like an RSL-RL vector environment while keeping the smaller policy action space. |
| `soccer_targets.py` | Computes `Pre_Goal`, `P_kick`, goal direction, and debug marker positions for Soccer. |
| `play_scene_presets.py` | Defines visual presets for videos. These only change appearance, not physics or rewards. |
| `cli_args.py` | Shared command-line argument helpers used by the training, play, and collection scripts. |
| `run_play_multi_scene_videos.sh` | Runs `play_lift.py` over multiple visual presets. Set `CHECKPOINT_PATH=/path/to/model.pt` before running it. |
| `run_collect_dataset.sh` | Runs dataset collection over multiple visual presets. Set `CHECKPOINT_PATH=/path/to/model.pt` before running it. |

### `EE_API_Test`

These scripts are small sanity checks for the end-effector API. They are useful when you change an action wrapper, camera, marker, or environment config.

| File | What it does |
| --- | --- |
| `test_ee_dims_video.py` | Runs the Lift scene and moves one EE action channel at a time, recording short videos so you can verify action directions. |
| `test_soccer_ee_dims_video.py` | Runs the same kind of check in Soccer, including ball/goal/`Pre_Goal`/`P_kick` marker visualization. |

### `VLA_FT`

These scripts are optional. They are for turning PPO demonstrations into a SmolVLA fine-tuning run.

| File | What it does |
| --- | --- |
| `finetune_smolvla.sh` | Starts a SmolVLA full fine-tune or LoRA run from a collected dataset shard. Most options are environment variables. |
| `smoke_test_smolvla.sh` | Runs a tiny end-to-end check: collect or reuse a small dataset, fine-tune briefly, then verify the output. |
| `verify_finetune_output.py` | Checks whether a fine-tuning output directory contains usable checkpoints and valid logs. |

### `source/openarm_vla`

This is the Python package that makes the environments importable. Most users do not need to edit it directly unless they are changing task definitions.

| Area | What it contains |
| --- | --- |
| `tasks/utils/` | Common VLA utilities, including action conversion and `make_vla_env(...)`. |
| `unimanual/lift/config/` | Lift VLA environment registration, camera setup, initial pose, target marker, and PPO config. |
| `unimanual/soccer/config/` | Soccer VLA environment registration, camera setup, observations, episode settings, and PPO config. |
| `unimanual/reach/`, `unimanual/cabinet/`, `bimanual/` | Older or auxiliary VLA task variants kept for completeness. They are not the main Lift/Soccer workflow. |

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
