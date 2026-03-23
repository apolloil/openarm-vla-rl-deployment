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

"""Bimanual Reach — VLA variant.

Keeps **all** original rewards, commands, events, and curriculum from
``OpenArmReachEnvCfg`` but replaces the joint-position actions with
Differential-IK so that VLA-format EE-delta commands can be applied directly.

VLA action format (14-dim):
  [left_arm(6), right_arm(6), left_grip(1), right_grip(1)]
  Each arm: [Δpos(3) + Δaxis-angle(3)] in robot base frame.
  Grippers: +1 = open, ≤ 0 = close  (BinaryJointPosition)

Gripper action does not appear in the original bimanual reach MDP rewards;
it is included here only to keep the VLA action vector layout uniform.
"""

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation.bimanual.reach import mdp
from openarm.tasks.manager_based.openarm_manipulation.bimanual.reach.reach_env_cfg import ReachEnvCfg

from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)

_LEFT_ARM_JOINTS  = [f"openarm_left_joint{i}"  for i in range(1, 8)]
_RIGHT_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]
_LEFT_FINGER  = "openarm_left_finger_joint.*"
_RIGHT_FINGER = "openarm_right_finger_joint.*"


@configclass
class OpenArmBiReachVlaEnvCfg(ReachEnvCfg):
    """Bimanual reach with DiffIK actions; keeps all original reach rewards."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot ──────────────────────────────────────────────────────────
        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "openarm_left_joint1": 0.0,
                    "openarm_left_joint2": 0.0,
                    "openarm_left_joint3": 0.0,
                    "openarm_left_joint4": 0.0,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": 0.0,
                    "openarm_right_joint1": 0.0,
                    "openarm_right_joint2": 0.0,
                    "openarm_right_joint3": 0.0,
                    "openarm_right_joint4": 0.0,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 0.0,
                    "openarm_left_finger_joint.*": 0.0,
                    "openarm_right_finger_joint.*": 0.0,
                },
            ),
        )

        # ── Commands body (same as joint_pos cfg) ─────────────────────────
        self.commands.left_ee_pose.body_name  = "openarm_left_hand"
        self.commands.right_ee_pose.body_name = "openarm_right_hand"
        self.commands.left_ee_pose.debug_vis  = False   # frame_prim.usd needs network
        self.commands.right_ee_pose.debug_vis = False

        # ── Rewards — body names (same as joint_pos cfg) ──────────────────
        for attr in (
            "left_end_effector_position_tracking",
            "left_end_effector_position_tracking_fine_grained",
            "left_end_effector_orientation_tracking",
        ):
            getattr(self.rewards, attr).params["asset_cfg"].body_names = ["openarm_left_hand"]

        for attr in (
            "right_end_effector_position_tracking",
            "right_end_effector_position_tracking_fine_grained",
            "right_end_effector_orientation_tracking",
        ):
            getattr(self.rewards, attr).params["asset_cfg"].body_names = ["openarm_right_hand"]

        # ── Actions: Differential IK (replaces joint-position control) ────
        # scale=0.01 → 1 cm / rad per raw VLA unit, adequate for bimanual reach
        _ik_ctrl = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        )

        self.actions.left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=_LEFT_ARM_JOINTS,
            body_name="openarm_left_hand",
            controller=_ik_ctrl,
            scale=0.01,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )
        self.actions.right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=_RIGHT_ARM_JOINTS,
            body_name="openarm_right_hand",
            controller=_ik_ctrl,
            scale=0.01,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )

        # Gripper: binary, not used by reach rewards but needed for 14-dim VLA layout
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
class OpenArmBiReachVlaEnvCfg_PLAY(OpenArmBiReachVlaEnvCfg):
    """Evaluation variant: 50 envs, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
