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
Bimanual OpenArm VLA 环境配置
==============================
面向「VLA 端到端推理」的双臂 Isaac Lab 环境，动作空间与 VLA 模型输出对齐：

    VLA 输出（14 维）= [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip]×2（左臂+右臂）

    env.step() 期望的 isaac_action（14 维）格式：
        [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]

    用 action_transform.vla_action_to_isaac() 完成格式转换后再传入 env.step()。

动作约定
--------
- **末端位姿增量**（6 维/臂）：基座坐标系（robot root frame）下的增量，
  格式 [Δpos(3), Δaxis-angle(3)]。

  坐标系已由 Isaac Lab 源码确认：DifferentialInverseKinematicsAction._compute_frame_pose()
  通过 subtract_frame_transforms() 将 EE 位姿转换到 base frame，DiffIK 的 delta 因此
  直接叠加在 base frame 上（位置直接相加，旋转左乘即 base frame 旋转）。
  VLA 输出"基座坐标系增量"与此完全对齐，无需额外坐标变换。

- **夹爪**（1 维/臂）：二值控制，正数 (+1) = 张开，非正数 (-1) = 闭合。
  内部使用 BinaryJointPositionActionCfg，open=0.044 m，close=0.0 m。
  若 VLA 输出连续值 [0,1]，请用 vla_action_to_isaac() 中的阈值映射（>0.5 → +1，≤0.5 → -1）。

- 每臂内部使用 HIGH_PD 配置（stiffness=400, damping=80），适合任务空间控制。

场景
----
- 与 **Lift a cube**（`unimanual/lift`）同款：SeattleLabTable、DexCube、`/World/GroundPlane` 与 dome light，便于可视化 / 录屏；VLA 仍是双臂 DiffIK + 夹爪，未接入 Lift 的 command / 抬升奖励逻辑。

注册的 Gym ID
-------------
- Isaac-VLA-OpenArm-Bi-v0        （完整：4096 并行）
- Isaac-VLA-OpenArm-Bi-Play-v0   （评估：50 并行，无随机扰动）
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)

# ─────────────────────────────────────────────
# 关节名称常量
# ─────────────────────────────────────────────
_LEFT_ARM_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
_RIGHT_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]
_LEFT_FINGER = "openarm_left_finger_joint.*"
_RIGHT_FINGER = "openarm_right_finger_joint.*"


##
# 场景
##


@configclass
class VlaSceneCfg(InteractiveSceneCfg):
    """双臂 VLA 场景：与 Unimanual Lift-a-cube 相同的桌面 + DexCube + 地面 + 灯光 + 机器人。"""

    # Table: procedural cuboid (no network download required)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.525)),
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 1.05),
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.9), roughness=0.5
            ),
        ),
    )

    # 地面网格（physics_material=None 避免 5.1 下 Plane 绑定失败）
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(physics_material=None),
    )

    robot: ArticulationCfg = MISSING

    # Cube object: procedural rigid body (no network download required)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.022), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.044, 0.044, 0.044),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.1),
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.1),  # red cube
                roughness=0.5,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# 动作
##


@configclass
class ActionsCfg:
    """
    双臂差分 IK 动作 + 二值夹爪。

    拼接顺序（action_manager 按属性定义顺序拼接）：
      [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)] = 14 维

    其中每臂 6 维为 [Δpos(3), Δaxis-angle(3)]，基座坐标系下的增量。
    夹爪 1 维：正数 = 张开，非正数 = 闭合。

    使用 action_transform.vla_action_to_isaac() 可将 VLA 的 14 维输出转换为此格式。
    """

    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    left_gripper_action: ActionTerm = MISSING
    right_gripper_action: ActionTerm = MISSING


##
# 观测
##


@configclass
class ObservationsCfg:
    """Policy 观测：本体状态（关节位置/速度）+ 末端位姿。"""

    @configclass
    class PolicyCfg(ObsGroup):
        # 左臂关节
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEFT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEFT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 上一时刻动作（policy 自回归参考）
        left_last_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        right_last_action = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# 事件
##


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    # 与 Lift 任务一致：重置方块位姿，避免长时间滚动测试后 cube 飞出桌面
    reset_object_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


##
# 奖励（VLA 推理时可设为 0；此处保留结构供微调 RL 使用）
##


@configclass
class RewardsCfg:
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


##
# 终止
##


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# 主环境配置
##


@configclass
class OpenArmBiVlaEnvCfg(ManagerBasedRLEnvCfg):
    """
    双臂 OpenArm VLA 基础配置。
    动作空间（14 维）= 差分 IK（6维/臂）+ 二值夹爪（1维/臂）。
    env.step() 期望格式：[left_arm(6), right_arm(6), left_grip(1), right_grip(1)]。
    """

    scene: VlaSceneCfg = VlaSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        # 俯视桌面 + DexCube（原 (3.5,3.5,3.5) 易显「空背景」）
        self.viewer.eye = (2.2, 2.2, 1.35)
        self.viewer.lookat = (0.45, 0.0, 0.25)

        # 使用高刚度 PD 配置（适合任务空间 / 差分 IK 控制）
        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── 左臂：差分 IK ──────────────────────────────────────────────
        self.actions.left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=_LEFT_ARM_JOINTS,
            body_name="openarm_left_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",       # 接收 [Δpos(3) + Δaxis-angle(3)]
                use_relative_mode=True,    # 增量模式
                ik_method="dls",           # 阻尼最小二乘，数值稳定
            ),
            # 位置增量缩放：1 cm/step 量级；可按需调整
            scale=0.01,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )

        # ── 右臂：差分 IK ──────────────────────────────────────────────
        self.actions.right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=_RIGHT_ARM_JOINTS,
            body_name="openarm_right_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.01,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )

        # ── 夹爪：二值控制，正数 → 张开(0.044 m)，非正数 → 闭合(0.0 m) ──
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[_LEFT_FINGER],
            open_command_expr={_LEFT_FINGER: 0.044},
            close_command_expr={_LEFT_FINGER: 0.0},
        )
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[_RIGHT_FINGER],
            open_command_expr={_RIGHT_FINGER: 0.044},
            close_command_expr={_RIGHT_FINGER: 0.0},
        )


@configclass
class OpenArmBiVlaEnvCfg_PLAY(OpenArmBiVlaEnvCfg):
    """评估模式：少量并行环境，关闭观测噪声。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
