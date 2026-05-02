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

"""Unimanual Soccer — VLA variant.

Mirror of ``OpenArmLiftVlaEnvCfg`` for the soccer scene:

  * Replaces the arm action with Differential-IK so the policy issues
    base-frame Cartesian deltas (same as Lift VLA).
  * Bakes the gripper-down ``_VLA_INIT_JOINT_POS`` into ``init_state`` so
    every reset (manual + auto) starts from the same wrist pose used by
    Lift VLA. This is convenient for cross-task warm-starting and keeps the
    EE close to the ball at episode start.
  * Replaces the soccer scene's ``GroundPlane`` with a procedural cuboid
    slab (offline-safe; avoids the headless-RTX white-floor failure mode
    we hit on Lift).
  * Adds a static ``CameraCfg`` so headless ``render_mode='rgb_array'``
    works (mirrors Lift VLA's ``scene_rgb_cam``).
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_unimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)
from openarm.tasks.manager_based.openarm_manipulation.unimanual.soccer import mdp
from openarm.tasks.manager_based.openarm_manipulation.unimanual.soccer.config.joint_pos_env_cfg import (
    OpenArmSoccerEnvCfg,
)
from openarm.soccer import SOCCER_FIELD_POS

_ARM_JOINTS = [f"openarm_joint{i}" for i in range(1, 8)]

# DiffIK-friendly arm dynamics (identical to Lift VLA).
_VLA_ARM_DISABLE_GRAVITY = True
_VLA_ARM_STIFFNESS = 400.0
_VLA_ARM_DAMPING = 80.0

# "Gripper pointing (mostly) down" init pose — same captured values used by
# OpenArmLiftVlaEnvCfg (see lift/config/vla_env_cfg.py for full provenance).
# joint3 / joint7 keep a ~1e-3 rad buffer below the soft limits so the
# articulation validator's strict <=/>= checks always pass.
_VLA_INIT_JOINT_POS = {
    "openarm_joint1":        1.5700,
    "openarm_joint2":       -0.3610,
    "openarm_joint3":       -1.5690,
    "openarm_joint4":        0.7196,
    "openarm_joint5":       -0.0004,
    "openarm_joint6":        0.0003,
    "openarm_joint7":       -1.5690,
    "openarm_finger_joint1": 0.0440,
    "openarm_finger_joint2": 0.0440,
}

# Procedural floor slab matching Lift VLA — replaces the soccer scene's
# default GroundPlane (which trips on a NoneType collision_prim_path under
# this Isaac Lab build).
_FLOOR_SIZE_XY = (48.0, 48.0)
_FLOOR_THICKNESS = 0.08
_FLOOR_TOP_Z = -1.05
_FLOOR_CENTER_Z = _FLOOR_TOP_Z - _FLOOR_THICKNESS / 2.0


@configclass
class OpenArmSoccerVlaEnvCfg(OpenArmSoccerEnvCfg):
    """Soccer with DiffIK arm action, offline-safe procedural ground + camera."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot: HIGH_PD for DiffIK with the baked gripper-down init pose ──
        robot_cfg = OPEN_ARM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot_cfg.spawn.rigid_props.disable_gravity = _VLA_ARM_DISABLE_GRAVITY
        robot_cfg.actuators["openarm_arm"].stiffness = _VLA_ARM_STIFFNESS
        robot_cfg.actuators["openarm_arm"].damping = _VLA_ARM_DAMPING

        _default_init = robot_cfg.init_state
        robot_cfg.init_state = ArticulationCfg.InitialStateCfg(
            pos=_default_init.pos,
            rot=_default_init.rot,
            lin_vel=_default_init.lin_vel,
            ang_vel=_default_init.ang_vel,
            joint_pos=dict(_VLA_INIT_JOINT_POS),
            joint_vel=_default_init.joint_vel,
        )
        self.scene.robot = robot_cfg

        # ── Camera: cam_08b_closer_mid — left-side elevated, ~1.39 m from scene ─
        # Selected from an offline camera pilot (left-side elevated, ~1.39 m).
        # Provides a clean left-side view with robot, ball, field, and goal all
        # visible without the arm occluding the ball.
        _eye = (0.01, 1.21, 0.63)
        _look = (0.45, 0.0, 0.10)
        self.viewer.eye = _eye
        self.viewer.lookat = _look

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

        # ── DiffIK arm action (identical to Lift VLA) ───────────────────────
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
        # gripper action is BinaryJointPosition from parent; keep it.

        # ── Add EE Cartesian position to policy observations ─────────────────
        # Appended 3 dims: [x, y, z] in robot root frame.
        self.observations.policy.ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)

        # ── Replace ground plane with procedural cuboid slab ────────────────
        self.scene.plane = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, _FLOOR_CENTER_Z)),
            spawn=sim_utils.CuboidCfg(
                size=(_FLOOR_SIZE_XY[0], _FLOOR_SIZE_XY[1], _FLOOR_THICKNESS),
                collision_props=CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.22, 0.22, 0.24),
                    roughness=0.92,
                ),
            ),
        )

        # ── Episode horizon ────────────────────────────────────────────────
        # MetaWorld SawyerSoccerV2 uses max_path_length=500.  With Isaac's
        # 50 Hz env step (0.02 s), 10 s gives the same 500 policy decisions.
        self.episode_length_s = 10.0


@configclass
class OpenArmSoccerVlaEnvCfg_PLAY(OpenArmSoccerVlaEnvCfg):
    """Evaluation variant: 50 envs, no observation noise, shorter horizon."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # Override at runtime via OPENARM_PLAY_EPISODE_LENGTH_S (mirrors Lift VLA).
        import os as _os
        _eps_raw = _os.environ.get("OPENARM_PLAY_EPISODE_LENGTH_S", "").strip()
        self.episode_length_s = float(_eps_raw) if _eps_raw else 10.0
