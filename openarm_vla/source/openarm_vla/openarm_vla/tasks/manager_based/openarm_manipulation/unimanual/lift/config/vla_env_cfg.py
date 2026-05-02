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

Inherits the **complete** ``OpenArmCubeLiftEnvCfg`` (ee_frame, all rewards,
events, terminations, curriculum) and replaces the arm action with
Differential-IK. The gripper binary action is kept unchanged.

The table, cube, and **floor** are procedural primitives so this variant
works **offline** without Omniverse Nucleus grid-ground USD (which can fail to
tint / render correctly in some headless + RTX paths, yielding a flat white
background). The floor is a large dark slab; the table uses a simple wood-like
albedo (state-based RL is unaffected).

Robot initial posture
---------------------
We override ``init_state.joint_pos`` with a **pre-pitched** pose captured from
a replayed dpitch warmup (see ``EE_API_Test/test_ee_dims_video.py`` for the
original observation that a constant Δpitch rotates the wrist downward).

Concretely, the numbers below are the ``best.joint_pos`` recorded at peak
EE-down alignment (≈ 0.88) from
``capture_dpitch_amp8_300_2026-04-21_16-18-34.json`` (step 143). The old
``EulerEEActionWrapper`` tried to achieve the same thing at runtime but its
``reset()`` only fired on manual ``env.reset()`` (eval) and *not* on Isaac
Lab's internal auto-reset, leaving the first eval episode out-of-distribution.
Baking the pose directly into ``init_state`` makes every reset (manual + auto)
land on the same starting configuration, so train / eval distributions match.

The gripper is pointing "most of the way" down (not perfectly vertical —
joint7 saturates at ±π/2 during the warmup before the gripper can rotate
fully) but this is a much better starting pose for pick-place than the
original ``OPEN_ARM_HIGH_PD_CFG`` default (wrist horizontal).

VLA action format (7-dim):
  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip]
  Position/rotation deltas in the robot **base** frame.
  grip > 0.5 → open, ≤ 0.5 → close  (BinaryJointPosition threshold).

PPO policy interface is reduced to 4-dim ``[Δx, Δy, Δz, grip]`` by
``RL/lift_ee_action_wrapper.py``; rotation deltas are sent as zeros.

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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
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
from openarm_vla.tasks.manager_based.openarm_manipulation.unimanual.lift.mdp.events import (
    sync_target_marker_to_command,
)

from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_unimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

_ARM_JOINTS   = [f"openarm_joint{i}" for i in range(1, 8)]
_FINGER_JOINT = "openarm_finger_joint.*"

# Keep the original DiffIK-friendly robot dynamics. Earlier experiments showed
# that softening these gains hurt the existing reaching behavior before we had
# isolated the true grasping issue.
_VLA_ARM_DISABLE_GRAVITY = True
_VLA_ARM_STIFFNESS = 400.0
_VLA_ARM_DAMPING = 80.0

# Baked "gripper pointing (mostly) down" init_state, captured from a dpitch
# warmup replay (source: ``capture_dpitch_amp8_300_2026-04-21_16-18-34.json``,
# best step=143, align_down≈0.88). Fingers kept open at 0.044 (matches the
# pose the original 2026-03-27 checkpoint was trained under).
#
# ``openarm_joint7`` soft limit is [-π/2, +π/2] ≈ [-1.5707963, +1.5707963].
# The raw capture was -1.5707959 (safely inside), but rounding to 4 decimals
# overshoots to -1.5708 — which Isaac Lab's articulation validator rejects
# with a strict ``<=/>=`` check. We keep a tiny buffer (~1e-3 rad ≈ 0.06°)
# on joint3/joint7 so the pose stays valid under any precision round-trip.
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

# Cube size: DexCube scaled at 0.8 → roughly 4.4 cm side
_CUBE_HALF = 0.022   # half-extent for CuboidCfg (full size = 0.044 m)

# World-floor slab (replaces Grid/default_environment.usd ground plane).
_FLOOR_SIZE_XY = (48.0, 48.0)
_FLOOR_THICKNESS = 0.08
# Top surface of the slab matches the old GroundPlane placement (table frame z≈0).
_FLOOR_TOP_Z = -1.05
_FLOOR_CENTER_Z = _FLOOR_TOP_Z - _FLOOR_THICKNESS / 2.0


@configclass
class OpenArmLiftVlaEnvCfg(OpenArmCubeLiftEnvCfg):
    """Lift-Cube with DiffIK arm action, offline-safe procedural assets."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot: HIGH_PD for DiffIK, with a pre-pitched init_state baked
        # in (see `_VLA_INIT_JOINT_POS` above). Every reset (manual + auto)
        # lands on this same gripper-mostly-down pose so train and eval
        # distributions match and the first eval episode is not OOD.
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

        # ── Freeze goal per-episode + longer horizon for VLA collection ─
        # Base Lift env resamples ``object_pose`` every 5 s; for VLA data
        # collection we want a single stable goal per episode so the visual
        # marker stays put and the policy has an unambiguous target to reach.
        # A 9 s horizon gives comfortable margin for reach + lift + hold.
        self.commands.object_pose.resampling_time_range = (1e6, 1e6)
        self.episode_length_s = 9.0

        # ── Visual-only target marker (kinematic sphere, no collisions) ─
        # A kinematic rigid body (disable_gravity=True, kinematic_enabled=True,
        # no collision_props, negligible mass) lets us move the sphere each
        # step via ``write_root_pose_to_sim`` without it interacting with the
        # arm or cube.
        self.scene.target_marker = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetMarker",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.05)),
            spawn=sim_utils.SphereCfg(
                radius=0.02,
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                mass_props=MassPropertiesCfg(mass=1e-4),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.70, 0.20, 0.90),
                    roughness=0.35,
                    metallic=0.0,
                ),
            ),
        )

        # Fire every env step so the marker tracks ``object_pose`` even though
        # the command is frozen (cheap no-op writes after the first sync).
        self.events.sync_target_marker = EventTerm(
            func=sync_target_marker_to_command,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            is_global_time=True,
            params={
                "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg("robot"),
                "marker_cfg": SceneEntityCfg("target_marker"),
            },
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
            body_name="openarm_hand",
            controller=_ik_ctrl,
            scale=0.01,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )

        # gripper action is already BinaryJointPosition in parent; keep it

        # ── Add EE Cartesian pose to policy observations ─────────────────────
        # Appended 7 dims: [x, y, z, qw, qx, qy, qz] in robot root frame.
        self.observations.policy.ee_pose = ObsTerm(func=mdp.ee_pose_in_robot_root_frame)

        # ── Replace USD assets with offline procedural primitives ──────────

        # Table: static cuboid — wood-like albedo (easy to see vs floor; RL obs are state-only).
        # Original table is at (0.5, 0, 0), top surface roughly at z=0
        # when ground-plane is at z=-1.05.  We match those dims here.
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.525)),
            spawn=sim_utils.CuboidCfg(
                size=(1.2, 0.8, 1.05),
                collision_props=CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.52, 0.34, 0.18),
                    roughness=0.72,
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

        # Procedural floor slab instead of Nucleus ``default_environment.usd`` grid plane.
        # That USD path + ``ChangePropertyCommand`` tinting is fragile in headless RTX /
        # in-memory stages and often reads as missing floor (all white with dome light).
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


@configclass
class OpenArmLiftVlaEnvCfg_PLAY(OpenArmLiftVlaEnvCfg):
    """Evaluation variant: 50 envs, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # The trained expert reaches + holds the cube in ≈ 2 s. The training
        # horizon stays at 9 s (set in the base class) so reward shaping /
        # `model_3999.pt` remain reproducible, but for ``play_lift.py`` and
        # ``collect_dataset.py`` we shorten the PLAY horizon so that:
        #   * eval videos don't have a 7 s idle "hold" tail per episode,
        #   * failed-to-reach episodes time-out and get discarded faster,
        #   * successful episodes still have comfortable margin over the
        #     2 s expert runtime.
        # Override at runtime via ``OPENARM_PLAY_EPISODE_LENGTH_S`` (float,
        # seconds) if you want to A/B different horizons without editing.
        import os as _os
        _eps_raw = _os.environ.get("OPENARM_PLAY_EPISODE_LENGTH_S", "").strip()
        self.episode_length_s = float(_eps_raw) if _eps_raw else 3.0
