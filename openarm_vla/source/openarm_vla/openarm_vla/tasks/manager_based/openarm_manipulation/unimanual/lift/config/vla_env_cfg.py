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

"""Unimanual Lift — VLA variant.

Inherits the **complete** ``OpenArmCubeLiftEnvCfg`` (robot, ee_frame,
all rewards, events, terminations, curriculum) and replaces only the arm
action with Differential-IK.  The gripper binary action is kept unchanged.

The table and cube are replaced with procedural primitives so this variant
works **offline** without downloading Omniverse Nucleus / S3 USD assets.

VLA action format (7-dim):
  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip]
  Position/rotation deltas in the robot **base** frame.
  grip > 0.5 → open, ≤ 0.5 → close  (BinaryJointPosition threshold).

Rewards (unchanged from original Lift):
  reaching_object         (+1.1)  – EE-to-object distance
  lifting_object          (+15.0) – object above minimal_height=0.04 m
  object_goal_tracking    (+16.0) – object approaching goal (std=0.3)
  object_goal_tracking_fine_grained (+5.0) – fine-grained goal (std=0.05)
  action_rate             (-1e-4)
  joint_vel               (-1e-4)
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from isaaclab.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift import mdp
from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.config.joint_pos_env_cfg import OpenArmCubeLiftEnvCfg

from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_unimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

_ARM_JOINTS   = [f"openarm_joint{i}" for i in range(1, 8)]
_FINGER_JOINT = "openarm_finger_joint.*"

# Cube size: DexCube scaled at 0.8 → roughly 4.4 cm side
_CUBE_HALF = 0.022   # half-extent for CuboidCfg (full size = 0.044 m)


@configclass
class OpenArmLiftVlaEnvCfg(OpenArmCubeLiftEnvCfg):
    """Lift-Cube with DiffIK arm action, offline-safe procedural assets."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot: HIGH_PD config is required for DiffIK ──────────────────
        # OPEN_ARM_CFG (parent): stiffness=80, damping=4, disable_gravity=False
        # OPEN_ARM_HIGH_PD_CFG:  stiffness=400, damping=80, disable_gravity=True
        # Without HIGH_PD the arm collapses under gravity between DiffIK steps.
        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=self.scene.robot.init_state,   # keep parent init joints
        )

        # ── Camera: close-up view of robot + table + cube ─────────────────
        # eye: 斜前方 1.2m 高，距桌面约 1.5m；lookat: 桌上抓取中心
        _eye = (1.5, 1.2, 1.0)
        _look = (0.4, 0.0, 0.1)
        self.viewer.eye = _eye
        self.viewer.lookat = _look

        # Headless rgb_array uses Replicator on a real USD Camera. The default
        # ``/OmniverseKit_Persp`` exists only with a GUI viewport.
        # Match the same eye / lookat as the GUI viewer (Isaac Lab uses this
        # view-matrix helper for camera world poses; quaternion is USD OpenGL).
        _eyes_t = torch.tensor([_eye], dtype=torch.float32)
        _look_t = torch.tensor([_look], dtype=torch.float32)
        _R = create_rotation_matrix_from_view(_eyes_t, _look_t, up_axis="Z", device="cpu")
        _quat_wxyz = tuple(float(x) for x in quat_from_matrix(_R)[0].tolist())

        self.scene.scene_rgb_cam = CameraCfg(
            prim_path="/World/scene_rgb_cam",
            update_period=0.0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 50.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=_eye,
                rot=_quat_wxyz,
                convention="opengl",
            ),
        )
        self.viewer.cam_prim_path = "/World/scene_rgb_cam"
        self.viewer.resolution = (1280, 720)
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "Off"

        # ── Disable debug visualisation (needs frame_prim.usd from network) ─
        self.commands.object_pose.debug_vis = False

        # ── Replace arm action with DiffIK ─────────────────────────────────
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

        # gripper action is already BinaryJointPosition in parent; keep it

        # ── Replace USD assets with offline procedural primitives ──────────

        # Table: static white cuboid approximating a lab table surface.
        # Original table is at (0.5, 0, 0), top surface roughly at z=0
        # when ground-plane is at z=-1.05.  We match those dims here.
        self.scene.table = AssetBaseCfg(
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

        # Cube object: procedural rigid body replacing DexCube USD.
        # Placed at the same position as the original DexCube.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.4, 0.0, 0.022),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(2 * _CUBE_HALF, 2 * _CUBE_HALF, 2 * _CUBE_HALF),
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
                    diffuse_color=(0.8, 0.2, 0.1),   # red cube, easy to spot
                    roughness=0.5,
                ),
            ),
        )

        # Ground plane: use physics_material=None to avoid the prim-lookup crash
        self.scene.plane = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
            spawn=sim_utils.GroundPlaneCfg(physics_material=None),
        )


@configclass
class OpenArmLiftVlaEnvCfg_PLAY(OpenArmLiftVlaEnvCfg):
    """Evaluation variant: 50 envs, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
