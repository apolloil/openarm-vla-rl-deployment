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

"""Unimanual Reach — VLA variant.

Keeps **all** original rewards, commands, events, and curriculum from
``OpenArmReachEnvCfg`` but replaces the joint-position action with a
Differential-IK action so that VLA-format EE-delta commands can be applied
directly.

VLA action format (7-dim):
  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip]
  Position/rotation deltas are in the robot **base** frame.
  grip > 0.5 → open, ≤ 0.5 → close  (BinaryJointPosition threshold).

The gripper action is not used by reach rewards but is included to keep the
7-dim VLA action vector layout consistent.
"""

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation.unimanual.reach import mdp
from openarm.tasks.manager_based.openarm_manipulation.unimanual.reach.reach_env_cfg import ReachEnvCfg

from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_unimanual import (
    OPEN_ARM_CFG,
    OPEN_ARM_HIGH_PD_CFG,
)

_ARM_JOINTS = [f"openarm_joint{i}" for i in range(1, 8)]
_FINGER_JOINT = "openarm_finger_joint.*"


@configclass
class OpenArmReachVlaEnvCfg(ReachEnvCfg):
    """Unimanual reach with DiffIK actions; keeps all original reach rewards."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot: HIGH_PD required for DiffIK (disable_gravity + stiff PD) ─
        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "openarm_joint1": 1.57,
                    "openarm_joint2": 0.0,
                    "openarm_joint3": -1.57,
                    "openarm_joint4": 1.57,
                    "openarm_joint5": 0.0,
                    "openarm_joint6": 0.0,
                    "openarm_joint7": 0.0,
                    "openarm_finger_joint.*": 0.0,
                },
            ),
        )

        # ── Commands body ──────────────────────────────────────────────────
        self.commands.ee_pose.body_name = "openarm_hand"
        self.commands.ee_pose.debug_vis = False   # frame_prim.usd needs network

        # ── Rewards — body names ───────────────────────────────────────────
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["openarm_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["openarm_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["openarm_hand"]

        # ── Actions: Differential IK ───────────────────────────────────────
        _ik_ctrl = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        )
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINTS,
            body_name="openarm_hand",
            controller=_ik_ctrl,
            scale=0.01,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )

        # Gripper: binary, not used by reach rewards but needed for 7-dim VLA layout
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[_FINGER_JOINT],
            open_command_expr={_FINGER_JOINT: 0.044},
            close_command_expr={_FINGER_JOINT: 0.0},
        )


@configclass
class OpenArmReachVlaEnvCfg_PLAY(OpenArmReachVlaEnvCfg):
    """Evaluation variant: 50 envs, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
