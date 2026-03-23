# OpenArm VLA 接入与分轴录像

## 1. 把「VLA 动作」送进仿真（输入侧）

在 **Isaac Sim 已通过 `AppLauncher` 启动** 之后（与训练脚本相同顺序），再导入 `openarm`。


| 目的      | API                                                                                                                                 |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 创建环境    | `from openarm_vla.tasks.utils.vla_env_wrapper import make_vla_env` `env = make_vla_env(task, num_envs=1, seed=0, render_mode=None)` |
| 需要相机画面  | `render_mode="rgb_array"`                                                                                                           |
| 回合开始    | `obs, info = env.reset()` 或 `env.reset(seed=…)`                                                                                     |
| 每步控制    | `obs, rew, terminated, truncated, info = env.step(action)`                                                                          |
| 查询动作维度  | `env.expected_action_dim`（单臂 7，双臂 14）                                                                                               |
| 取一帧 RGB | 在 `render_mode="rgb_array"` 下 `frame = env.render()`，形状约为 `H×W×3`、`uint8`                                                           |
| 结束      | `env.close()`                                                                                                                       |


**`action` 形状与语义（VLA 格式，经 `OpenArmVlaWrapper` 转成 Isaac 的 DiffIK + 二值夹爪）：**

- **单臂 7 维**：`[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip]`，单位 m / rad（每步增量），均在 **机器人基座系**。
- **双臂 14 维**：左臂 7 维 + 右臂 7 维，布局为 `[左 7 | 右 7]`，每一侧与上表相同。
- **夹爪**：连续标量，**> 0.5 → 张开，≤ 0.5 → 闭合**（内部再映射为 Isaac 的 ±1）。

**`task` 取值与 Play 环境 ID（`make_vla_env` 默认 `play=True`）：**


| `task`      | 模式  | Gym ID（Play）                            |
| ----------- | --- | --------------------------------------- |
| `lift`      | 单臂  | `Isaac-VLA-Lift-Cube-OpenArm-Play-v0`   |
| `reach_uni` | 单臂  | `Isaac-VLA-Reach-OpenArm-Play-v0`       |
| `reach_bi`  | 双臂  | `Isaac-VLA-Reach-OpenArm-Bi-Play-v0`    |
| `cabinet`   | 单臂  | `Isaac-VLA-Open-Drawer-OpenArm-Play-v0` |


安装包（在工作区根目录执行一次）：

```bash
pip install -e openarm_isaac_lab/source/openarm --no-build-isolation
pip install -e openarm_vla/source/openarm_vla --no-build-isolation
```

---

## 2. 从仿真读出「结果」（输出侧）


| 量    | 来源                                                                                                  |
| ---- | --------------------------------------------------------------------------------------------------- |
| 观测   | `env.reset()` / `env.step(...)` 返回的 `obs`（Isaac Lab 字典，具体键与 shape 以环境启动时的 Observation Manager 打印为准） |
| 标量反馈 | `rew`、`terminated`、`truncated`、`info`                                                               |
| 图像   | 同上，在 `render_mode="rgb_array"` 时用 `env.render()`                                                    |


（语言、图像编码等 **VLA 策略网络本身的输入** 不在本仓库内；这里只描述与 **OpenArm + Isaac Lab 环境** 的接口。）

---

## 3. `test_ee_dims_video.py`：七个 MP4 在做什么

脚本 **不加载任何 VLA 模型**，只是在 **Lift-Cube VLA 场景**里，用固定规则构造 7 维 `action`，等价于「无策略、只探每个输出通道」的冒烟测试。

**流程概要：**

1. `AppLauncher` 启动 Kit（默认无头、开相机），`make_vla_env("lift", render_mode="rgb_array")`。
2. 对 `dim_index = 0 … 6` 依次：
  - `env.reset`；
  - 连发若干步 **全零平移 + 零转动 + 夹爪闭合**（`--grip-settle-steps`），让手指从默认张开收到可见的闭合；
  - 再一步 baseline + 首帧 `render`；
  - 随后 `steps` 步：在 baseline 上 **仅第 `dim_index` 个分量非零**（平移用 `--amplitude`，转动用 `--amplitude-rot`，第 6 维为夹爪张开演示用 `1.0`），每步 `render` 追加到帧列表；
  - 将帧写成 `ee_dim_XX_*.mp4`。

**七个文件与 VLA 通道对应：**


| 文件                     | 通道       |
| ---------------------- | -------- |
| `ee_dim_00_dx.mp4`     | Δx       |
| `ee_dim_01_dy.mp4`     | Δy       |
| `ee_dim_02_dz.mp4`     | Δz       |
| `ee_dim_03_droll.mp4`  | Δroll    |
| `ee_dim_04_dpitch.mp4` | Δpitch   |
| `ee_dim_05_dyaw.mp4`   | Δyaw     |
| `ee_dim_06_grip.mp4`   | grip（张开） |


默认输出目录：与脚本同级的 `videos/`（即 `openarm_vla/EE_API_Test/videos/`），可用 `--output-dir` 修改。

**常用命令：**

```bash
# 在 openarm/ 根目录、已配置 Isaac Lab 的 Python 中
python openarm_vla/EE_API_Test/test_ee_dims_video.py
python openarm_vla/EE_API_Test/test_ee_dims_video.py --gui   # 需要本地窗口时
```


| 参数                    | 默认                   | 说明                          |
| --------------------- | -------------------- | --------------------------- |
| `--steps`             | 150                  | 每个通道录制段内的 `env.step` 次数     |
| `--amplitude`         | 1.0                  | 仅平移三通道的原始幅值                 |
| `--amplitude-rot`     | 6.0                  | 仅转动三通道的原始幅值（略大以便录像里能看出姿态变化） |
| `--grip-settle-steps` | 24                   | reset 后专用「闭合」步数，避免手指仍停在默认张开 |
| `--output-dir`        | `videos/`（脚本旁）      | MP4 输出目录                    |
| `--fps`               | 30                   | 帧率                          |
| `--gui`               | 关                    | 有界面运行                       |


---

## 4. 最小推理循环示例（单臂）

```python
from isaaclab.app import AppLauncher
import argparse
import numpy as np

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.enable_cameras = True
launcher = AppLauncher(args)
sim = launcher.app

from openarm_vla.tasks.utils.vla_env_wrapper import make_vla_env

env = make_vla_env("lift", num_envs=1, seed=0, render_mode="rgb_array")
obs, info = env.reset()

for _ in range(200):
    action = np.zeros(env.expected_action_dim, dtype=np.float32)
    action[2] = 0.5
    obs, rew, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
sim.close()
```

