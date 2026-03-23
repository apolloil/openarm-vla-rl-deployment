# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VLA 动作格式转换工具
====================
将 VLA 模型输出的末端增量

    vla_action (14 维) = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip] × 2（左 + 右）

转换为 Isaac Lab env.step() 期望的格式

    isaac_action (14 维) = [left_arm(6), right_arm(6), left_grip(1), right_grip(1)]

拼接顺序与 ActionsCfg 的属性定义顺序一致：
  left_arm_action(6) → right_arm_action(6) → left_gripper_action(1) → right_gripper_action(1)

坐标系约定（已由 Isaac Lab 源码确认）
--------------------------------------
- Isaac Lab DifferentialInverseKinematicsAction._compute_frame_pose() 通过
  subtract_frame_transforms() 将 EE 位姿转换到 **robot base frame**；
  位置 delta 直接相加（base frame 加法），旋转 delta 左乘（base frame 旋转）。
- VLA 输出的"基座坐标系增量"与此完全对齐，无需额外坐标变换。

旋转约定
--------
- Δroll / Δpitch / Δyaw：ZYX 内旋欧拉角增量（rad）。
- Isaac Lab DiffIK 期望旋转增量为 Axis-Angle 形式。
- 对 VLA 控制频率下的小步长（~0.01–0.05 rad/step），小角度近似足够精确。

夹爪约定
--------
- BinaryJointPositionActionCfg：正数 → 张开(0.044 m)，非正数 → 闭合(0.0 m)。
- VLA 输出连续值 [0,1] 时：> 0.5 映射到 +1.0（张开），≤ 0.5 映射到 -1.0（闭合）。

使用方法
--------
在推理循环里::

    import torch
    from openarm_vla.tasks.utils.action_transform import vla_action_to_isaac

    # vla_action: (batch, 14)
    #   left:  [dx, dy, dz, droll, dpitch, dyaw, grip]  → indices 0..6
    #   right: [dx, dy, dz, droll, dpitch, dyaw, grip]  → indices 7..13
    isaac_action = vla_action_to_isaac(vla_action)
    # isaac_action: (batch, 14)
    #   [left_arm(6), right_arm(6), left_grip(1), right_grip(1)]

    obs, rew, done, trunc, info = env.step(isaac_action)
"""

from __future__ import annotations

import torch


def euler_delta_to_axis_angle(
    droll: torch.Tensor,
    dpitch: torch.Tensor,
    dyaw: torch.Tensor,
) -> torch.Tensor:
    """
    将 ZYX 欧拉角增量（小角度）转换为 Axis-Angle 三维向量。

    对于**小角度增量**，Axis-Angle ≈ [Δroll, Δpitch, Δyaw]（单位：rad）。
    若增量较大（> ~0.3 rad 每步），应使用完整的旋转矩阵/四元数方式以避免顺序耦合误差。

    Parameters
    ----------
    droll, dpitch, dyaw : Tensor, shape (...,)
        ZYX 欧拉角增量，单位 rad。

    Returns
    -------
    Tensor, shape (..., 3)
        Axis-Angle 向量 [ax, ay, az]，量级 = 旋转角度 rad。
    """
    # 小角度近似：旋转矩阵增量对应的 axis-angle
    # R ≈ I + [ω]×，其中 ω ≈ [droll, dpitch, dyaw]（body frame → base frame 一阶近似）
    # 对 VLA 控制频率下的小步长（~0.01–0.05 rad/step）精度充分
    return torch.stack([droll, dpitch, dyaw], dim=-1)


def euler_delta_to_axis_angle_full(
    droll: torch.Tensor,
    dpitch: torch.Tensor,
    dyaw: torch.Tensor,
) -> torch.Tensor:
    """
    精确版：将 ZYX 欧拉角增量转换为 Axis-Angle（适用于较大角度）。

    构造 Rz(dyaw)·Ry(dpitch)·Rx(droll) 的旋转矩阵，
    再提取对应 Axis-Angle。

    Parameters
    ----------
    droll, dpitch, dyaw : Tensor, shape (...,)

    Returns
    -------
    Tensor, shape (..., 3)  Axis-Angle
    """
    cr, sr = torch.cos(droll), torch.sin(droll)
    cp, sp = torch.cos(dpitch), torch.sin(dpitch)
    cy, sy = torch.cos(dyaw), torch.sin(dyaw)

    # ZYX: R = Rz · Ry · Rx
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    # trace = r00 + r11 + r22
    trace = r00 + r11 + r22
    # angle = arccos((trace - 1) / 2)
    cos_angle = (trace - 1.0) * 0.5
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)  # (...)

    eps = 1e-6
    safe_sin = torch.sin(angle).clamp(min=eps)
    denom = 2.0 * safe_sin

    ax = (r21 - r12) / denom
    ay = (r02 - r20) / denom
    az = (r10 - r01) / denom

    axis = torch.stack([ax, ay, az], dim=-1)   # (..., 3)
    axis_angle = axis * angle.unsqueeze(-1)

    # 退化到零旋转时用小角度近似
    small = (angle < eps).unsqueeze(-1)
    fallback = torch.stack([droll, dpitch, dyaw], dim=-1)
    return torch.where(small, fallback, axis_angle)


def gripper_to_binary(grip: torch.Tensor) -> torch.Tensor:
    """
    将 VLA 输出的连续夹爪值 [0, 1] 映射为 BinaryJointPositionAction 约定的信号。

    约定：> 0.5 → +1.0（张开），≤ 0.5 → -1.0（闭合）。

    Parameters
    ----------
    grip : Tensor, shape (..., 1)  或 (...,)，范围 [0, 1]

    Returns
    -------
    Tensor, 相同 shape，值为 +1.0 或 -1.0
    """
    return torch.where(grip > 0.5, torch.ones_like(grip), -torch.ones_like(grip))


def vla_action_to_isaac(
    vla_action: torch.Tensor,
    use_approx: bool = True,
) -> torch.Tensor:
    """
    将 VLA 模型输出 (batch, 14) 转换为 Isaac Lab env.step() 期望的动作格式。

    VLA 输出格式（14 维）：
        [left_dx(0), left_dy(1), left_dz(2),
         left_droll(3), left_dpitch(4), left_dyaw(5), left_grip(6),
         right_dx(7), right_dy(8), right_dz(9),
         right_droll(10), right_dpitch(11), right_dyaw(12), right_grip(13)]

    Isaac Lab 动作格式（14 维，与 ActionsCfg 属性定义顺序一致）：
        [left_arm(0:6), right_arm(6:12), left_grip(12), right_grip(13)]
         ↑ left_dpos(3) + left_axis_angle(3)
                          ↑ right_dpos(3) + right_axis_angle(3)

    坐标系：base frame 增量，与 VLA 输出一致，无需额外变换。
    夹爪：连续 [0,1] → 二值 +1.0/−1.0（threshold=0.5）。

    Parameters
    ----------
    vla_action : Tensor, shape (..., 14)
    use_approx : bool
        True  = 小角度近似（默认，快速，适合 < 0.1 rad/step）
        False = 精确旋转矩阵转换（较大角度增量时更准）

    Returns
    -------
    Tensor, shape (..., 14)
        [left_arm(6), right_arm(6), left_grip(1), right_grip(1)]
    """
    assert vla_action.shape[-1] == 14, f"期望 14 维动作，实际 {vla_action.shape[-1]}"

    convert = euler_delta_to_axis_angle if use_approx else euler_delta_to_axis_angle_full

    # 左臂（VLA indices 0..6）
    left = vla_action[..., :7]
    left_dpos = left[..., :3]                                          # (..., 3)
    left_aa   = convert(left[..., 3], left[..., 4], left[..., 5])     # (..., 3)
    left_grip = gripper_to_binary(left[..., 6:7])                      # (..., 1)

    # 右臂（VLA indices 7..13）
    right = vla_action[..., 7:]
    right_dpos = right[..., :3]                                        # (..., 3)
    right_aa   = convert(right[..., 3], right[..., 4], right[..., 5]) # (..., 3)
    right_grip = gripper_to_binary(right[..., 6:7])                    # (..., 1)

    # 拼接顺序与 ActionsCfg 属性顺序一致：
    #   left_arm_action(6) | right_arm_action(6) | left_gripper_action(1) | right_gripper_action(1)
    return torch.cat([left_dpos, left_aa, right_dpos, right_aa, left_grip, right_grip], dim=-1)
