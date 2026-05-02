# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""VLA-only event terms.

``sync_target_marker_to_command`` snaps a visual-only kinematic rigid body
(``target_marker``) to the current goal position produced by the pose command
manager. Wire it as an ``interval`` event (``interval_range_s=(0.0, 0.0)``,
``is_global_time=True``) so it fires every env step — the goal is frozen
per-episode by setting ``resampling_time_range=(1e6, 1e6)`` in the env cfg,
so in practice the marker only moves once per reset.
"""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

__all__ = ["sync_target_marker_to_command"]


def sync_target_marker_to_command(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    marker_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
) -> None:
    """Write the current goal position into the target_marker prim, world frame.

    The command is stored in the robot **root** frame; we transform to world
    via ``combine_frame_transforms`` before writing. Orientation is identity
    (pure translation) — the marker sphere is rotationally symmetric.

    ``env_ids`` is declared **without a default** on purpose: Isaac Lab's
    ``_resolve_common_term_cfg`` validator treats any default-valued arg as
    "optional / must appear in ``params``" and will raise
    ``ValueError: ... but received: [...]`` otherwise. Standard isaaclab
    mdp events (``reset_root_state_uniform`` etc.) follow the same pattern.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    marker: RigidObject = env.scene[marker_cfg.name]

    command = env.command_manager.get_command(command_name)  # (num_envs, >=3) in root frame
    des_pos_b = command[:, :3]

    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        des_pos_b,
    )

    num_envs = des_pos_w.shape[0]
    ident_quat = torch.zeros((num_envs, 4), device=des_pos_w.device, dtype=des_pos_w.dtype)
    ident_quat[:, 0] = 1.0  # wxyz: (1, 0, 0, 0)

    pose = torch.cat([des_pos_w, ident_quat], dim=-1)  # (num_envs, 7)

    # env_ids=None → update all envs (matches event manager's global_time path).
    if env_ids is None or (isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0):
        marker.write_root_pose_to_sim(pose)
    else:
        marker.write_root_pose_to_sim(pose[env_ids], env_ids=env_ids)
