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

"""Unimanual Cabinet (Open Drawer) — VLA variant.

Inherits **all** of ``OpenArmCabinetEnvCfg`` (robot, cabinet, ee_frame,
all rewards, events, terminations) and replaces only the arm action with
Differential-IK.  The gripper binary action is kept unchanged.

VLA action format (7-dim):
  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip]
  Position/rotation deltas in the robot **base** frame.
  grip > 0.5 → open, ≤ 0.5 → close  (BinaryJointPosition threshold).

Rewards (unchanged from original Cabinet):
  approach_ee_handle      (+2.0)   – EE approaches drawer handle
  align_ee_handle         (+0.5)   – EE aligns with handle
  approach_gripper_handle (+5.0)   – gripper fingers approach handle
  align_grasp_around_handle (+0.125) – gripper aligned around handle
  grasp_handle            (+0.5)   – gripper grasps handle
  open_drawer_bonus       (+7.5)   – drawer opened
  multi_stage_open_drawer (+1.0)   – multi-stage progress bonus
  action_rate_l2          (-1e-2)
  joint_vel               (-1e-4)
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation.unimanual.cabinet import mdp
from openarm.tasks.manager_based.openarm_manipulation.unimanual.cabinet.config.joint_pos_env_cfg import OpenArmCabinetEnvCfg

from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_unimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)

_ARM_JOINTS = [f"openarm_joint{i}" for i in range(1, 8)]


@configclass
class OpenArmCabinetVlaEnvCfg(OpenArmCabinetEnvCfg):
    """Cabinet open-drawer with DiffIK arm action; all original rewards retained."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot: HIGH_PD required for DiffIK (disable_gravity + stiff PD) ─
        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=self.scene.robot.init_state,   # keep parent init joints
        )

        # ── Replace arm action with DiffIK ─────────────────────────────────
        _ik_ctrl = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        )
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINTS,
            body_name="openarm_ee_tcp",
            controller=_ik_ctrl,
            scale=0.01,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, -0.003)),
        )

        # gripper action is already BinaryJointPosition in parent; keep it


@configclass
class OpenArmCabinetVlaEnvCfg_PLAY(OpenArmCabinetVlaEnvCfg):
    """Evaluation variant: 50 envs, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
