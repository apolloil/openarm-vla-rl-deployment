# OpenArm VLA RL Deployment

中文文档。English documentation: [README.md](README.md).

这个仓库用于在 Isaac Lab 里跑 OpenArm 的 RL/VLA 流程。机器人资产和基础任务逻辑在 `openarm_isaac_lab` submodule 里；`openarm_vla` 负责 VLA 环境、PPO 专家策略训练/播放、动作接口测试，以及可选的 SmolVLA 微调脚本。

## 仓库结构

```text
openarm/
├── openarm_isaac_lab/          # git submodule：基础 OpenArm Isaac Lab 任务和资产
└── openarm_vla/
    ├── Data_Collector/         # PPO train/play 脚本和 action wrapper
    ├── EE_API_Test/            # EE 动作通道冒烟测试
    ├── VLA_FT/                 # 可选 SmolVLA 微调辅助脚本
    └── source/openarm_vla/     # 可安装 Python 包和 Gym 注册
```

## 安装

带 submodule clone，然后在 Isaac Lab 环境中 editable 安装两个包：

```bash
git clone --recurse-submodules https://github.com/apolloil/openarm-vla-rl-deployment.git
cd openarm-vla-rl-deployment

pip install -e openarm_isaac_lab/source/openarm --no-build-isolation
pip install -e openarm_vla/source/openarm_vla --no-build-isolation
```

`openarm_vla/VLA_FT/` 中的 SmolVLA 微调辅助脚本需要 LeRobot；请在用于 VLA 训练的 Python 环境中单独安装。

如果 clone 时没有拉 submodule：

```bash
git submodule update --init --recursive
```

## RL 训练与播放

在仓库根目录运行，并先激活 Isaac Lab conda 环境。

```bash
# Lift
python openarm_vla/Data_Collector/train_lift.py
python openarm_vla/Data_Collector/play_lift.py --checkpoint_path logs/rsl_rl/openarm_lift/<run>/model_3999.pt

# Soccer
python openarm_vla/Data_Collector/train_soccer.py
python openarm_vla/Data_Collector/play_soccer.py --checkpoint_path logs/rsl_rl/openarm_soccer/<run>/model_3200.pt
```

常用训练参数：

```bash
python openarm_vla/Data_Collector/train_lift.py --num_envs 2048 --max_iterations 5000 --seed 42
python openarm_vla/Data_Collector/train_soccer.py --num_envs 2048 --max_iterations 4000 --seed 42
python openarm_vla/Data_Collector/train_soccer.py --video --video_interval 500
```

播放脚本通过 CLI 参数配置 checkpoint 和常用播放选项，例如 `--checkpoint_path`、`--video_seconds`、`--scene_preset_id`、`--video_output_dir` 和 `--gui`。

## `openarm_vla` 关键文件

下面的路径都相对于 `openarm_vla/`。平时训练、播放、采集数据，主要看 `Data_Collector/` 就够了；`source/openarm_vla/` 更多是环境注册和配置代码。

### `Data_Collector`

这是 RL 主目录。训练 PPO、播放 checkpoint、采集 VLA 数据集都从这里进。

| 文件 | 做什么 |
| --- | --- |
| `train_lift.py` | 训练 Lift 任务的 PPO 专家。策略动作是 `[dx, dy, dz, grip]`，训练日志和 checkpoint 会写到 `logs/rsl_rl/openarm_lift/`。 |
| `play_lift.py` | 播放 Lift checkpoint。用 `--checkpoint_path` 指定模型，会录视频，也会导出策略。 |
| `train_soccer.py` | 训练 Soccer 任务的 PPO 专家。它只学“把球带到 `Pre_Goal`”，不在训练里硬学完整射门。 |
| `play_soccer.py` | 播放 Soccer checkpoint。先用策略把球带到 `Pre_Goal`，然后用脚本规则释放、移动到球后方并踢球。 |
| `collect_dataset.py` | 用训练好的 Lift 专家采集成功演示，保存为 LeRobot 数据集 shard，给后续 VLA 微调用。 |
| `lift_ee_action_wrapper.py` | 把 Lift 策略的 4 维动作转成环境需要的 7 维 DiffIK 动作。 |
| `soccer_ee_action_wrapper.py` | Soccer 版动作 wrapper。除了动作转换，还会主动校正末端姿态，减少夹爪越跑越歪的问题。 |
| `policy_action_vecenv_wrapper.py` | 把 Isaac Lab/Gym 环境包装成 RSL-RL 能直接训练的 VecEnv。 |
| `soccer_targets.py` | 统一计算 Soccer 的 `Pre_Goal`、`P_kick`、球门方向和可视化 marker。 |
| `play_scene_presets.py` | 播放视频的视觉 preset，只改颜色和材质，不影响物理和奖励。 |
| `cli_args.py` | 训练、播放、采集脚本共用的命令行参数工具。 |
| `run_play_multi_scene_videos.sh` | 批量跑 `play_lift.py` 的不同视觉 preset。运行前设置 `CHECKPOINT_PATH=/path/to/model.pt`。 |
| `run_collect_dataset.sh` | 批量采集不同视觉 preset 的数据集 shard。运行前设置 `CHECKPOINT_PATH=/path/to/model.pt`。 |

### `EE_API_Test`

这里是动作接口的冒烟测试。改了 action wrapper、相机或者 marker 之后，可以先跑这些脚本肉眼看方向是否正确。

| 文件 | 做什么 |
| --- | --- |
| `test_ee_dims_video.py` | 在 Lift 场景里逐个测试 7 个末端动作通道，每个通道录一段视频。 |
| `test_soccer_ee_dims_video.py` | Soccer 版通道测试，同时会显示球、球门、`Pre_Goal` 和 `P_kick` marker。 |

### `VLA_FT`

这里是可选的 SmolVLA 微调辅助脚本。只有在你要把 PPO 演示数据拿去训练 VLA 模型时才需要看。

| 文件 | 做什么 |
| --- | --- |
| `finetune_smolvla.sh` | 启动 SmolVLA full fine-tune 或 LoRA fine-tune。主要参数通过环境变量传入。 |
| `smoke_test_smolvla.sh` | 跑一条很小的端到端测试：采一点数据、跑一次 full、跑一次 LoRA，再检查输出。 |
| `verify_finetune_output.py` | 检查微调输出是否真的能加载、loss 是否正常、checkpoint 是否完整。 |

### `source/openarm_vla`

这里是可安装的 Python 包，但平时不需要逐个文件看。你可以把它理解为“环境定义和 Gym 注册层”。

| 区域 | 做什么 |
| --- | --- |
| `tasks/utils/` | 放通用 VLA 工具，比如动作格式转换和 `make_vla_env(...)`。 |
| `unimanual/lift/config/` | Lift 的 VLA 环境配置：Gym ID、相机、初始姿态、目标 marker、PPO 配置等。 |
| `unimanual/soccer/config/` | Soccer 的 VLA 环境配置：Gym ID、相机、观测、episode 设置、PPO 配置等。 |
| `unimanual/reach/`、`unimanual/cabinet/`、`bimanual/` | 早期或辅助任务。主线训练/播放目前主要看 Lift 和 Soccer。 |

## Soccer Pipeline

Soccer 训练阶段只训练策略把球带到 `Pre_Goal`。播放阶段在球稳定到达 `Pre_Goal` 后切换到硬编码启发式：释放并抬起机械臂、移动到 `P_kick`、下降到场地、再朝球门方向踢球。`Pre_Goal` 和 `P_kick` 会在 play 与 EE 冒烟测试中以标记点显示。

## Gym ID

| 任务 | 训练 ID | 播放 ID |
| --- | --- | --- |
| Lift | `Isaac-VLA-Lift-Cube-OpenArm-v0` | `Isaac-VLA-Lift-Cube-OpenArm-Play-v0` |
| Soccer | `Isaac-VLA-Soccer-OpenArm-v0` | `Isaac-VLA-Soccer-OpenArm-Play-v0` |

## 冒烟测试

```bash
python openarm_vla/EE_API_Test/test_ee_dims_video.py
python openarm_vla/EE_API_Test/test_soccer_ee_dims_video.py
```

生成的视频、日志、checkpoint、导出模型和数据集 shard 都已被 `.gitignore` 忽略。
