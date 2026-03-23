# OpenArm

[openarm_isaac_lab](https://github.com/enactic/openarm_isaac_lab) 的扩展，提供 VLA（视觉-语言-动作）接口和基于差分 IK 的环境。

**English docs:** [README.md](README.md)

## 目录结构

```
openarm/
├── openarm_isaac_lab/   # 基础 Isaac Lab 扩展（上游 Fork）
└── openarm_vla/         # VLA 接口与 DiffIK 环境
    ├── EE_API_Test/     # 冒烟测试脚本与接口说明（README）
    ├── docs/
    └── source/openarm_vla/
```

## 安装

在 Isaac Lab 的 conda 环境或 Docker 镜像中执行：

```bash
pip install -e openarm_isaac_lab/source/openarm --no-build-isolation
pip install -e openarm_vla/source/openarm_vla --no-build-isolation
```

完整环境搭建见 `openarm_vla/docs/conda_isaac_lab_setup.md`。

## 已注册的 Gym 环境


| Gym ID                                  | 任务       | 模式  |
| --------------------------------------- | -------- | --- |
| `Isaac-VLA-Lift-Cube-OpenArm-Play-v0`   | 抬起方块     | 单臂  |
| `Isaac-VLA-Reach-OpenArm-Play-v0`       | 末端到达     | 单臂  |
| `Isaac-VLA-Open-Drawer-OpenArm-Play-v0` | 开抽屉      | 单臂  |
| `Isaac-VLA-Reach-OpenArm-Bi-Play-v0`    | 双臂到达     | 双臂  |
| `Isaac-VLA-OpenArm-Bi-Play-v0`          | 自由（桌面场景） | 双臂  |


### 任务名称


| `task`        | 模式  | 动作维度 |
| ------------- | --- | ---- |
| `"lift"`      | 单臂  | 7    |
| `"reach_uni"` | 单臂  | 7    |
| `"cabinet"`   | 单臂  | 7    |
| `"reach_bi"`  | 双臂  | 14   |


**单臂（7 维）：** `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip]`，机器人基座系下的增量（m / rad）。  
**双臂（14 维）：** `[左臂 7 维, 右臂 7 维]`，每臂格式相同。  
**夹爪：** > 0.5 → 张开，≤ 0.5 → 闭合。

## 冒烟测试

录制 7 段 MP4（每个末端自由度一段），使用抬块场景：

```bash
python openarm_vla/EE_API_Test/test_ee_dims_video.py          # 无界面
python openarm_vla/EE_API_Test/test_ee_dims_video.py --gui    # 带视口
```

输出目录：`openarm_vla/EE_API_Test/videos/`。接口说明：[openarm_vla/EE_API_Test/README.md](openarm_vla/EE_API_Test/README.md)。