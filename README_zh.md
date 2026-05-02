# OpenArm VLA RL Deployment

中文文档。English documentation: [README.md](README.md).

这个仓库用于 OpenArm Isaac Lab 的 VLA/RL 实验部署。基础 Isaac Lab 扩展放在 `openarm_isaac_lab` submodule 中；主仓库额外提供 `openarm_vla`、DiffIK 任务变体、PPO 训练/播放脚本和轻量数据采集工具。

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

## Submodule Push 顺序

只要改到了 `openarm_isaac_lab`，必须先提交并 push submodule：

```bash
cd openarm_isaac_lab
git status
git add <changed files>
git commit -m "..."
git push origin main
cd ..
```

然后回到主仓库提交，其中会包含更新后的 submodule 指针：

```bash
git status
git add .gitignore README.md README_zh.md openarm_vla openarm_isaac_lab
git commit -m "..."
git push origin main
```
