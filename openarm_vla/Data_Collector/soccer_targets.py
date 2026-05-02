# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared soccer target geometry and USD markers for play/debug scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom

from openarm.soccer import SCENE_SCALE, SOCCER_GOAL_TARGET_POS
from openarm.tasks.manager_based.openarm_manipulation.assets.local_soccer import SOCCER_FIELD_TOP_Z
from openarm.tasks.manager_based.openarm_manipulation.assets.soccer_settings import (
    KICK_GOAL_DEPTH,
    KICK_PREP_DISTANCE,
    SOCCER_PKICK_BACKOFF,
    SOCCER_PKICK_MARKER_RADIUS,
    SOCCER_PRE_GOAL_Z,
)


@dataclass(frozen=True)
class SoccerTargets:
    pre_goal: np.ndarray
    goal: np.ndarray
    p_kick: np.ndarray
    goal_dir: np.ndarray


def compute_soccer_targets(env) -> SoccerTargets:
    """Compute Pre_Goal, deep Goal, and P_kick for env 0 in world frame."""
    base_env = getattr(env, "unwrapped", env)
    env_origin = base_env.scene.env_origins[0].detach().cpu().numpy()

    goal_pos = env_origin + np.asarray(SOCCER_GOAL_TARGET_POS, dtype=np.float64)
    if hasattr(base_env, "_soccer_goal_offsets"):
        goal_pos = goal_pos + base_env._soccer_goal_offsets[0, :3].detach().cpu().numpy()

    goal_dir = goal_pos[:2] - env_origin[:2]
    norm = float(np.linalg.norm(goal_dir))
    if norm < 1e-6:
        goal_dir = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        goal_dir = goal_dir / norm

    pre_goal = np.asarray(goal_pos, dtype=np.float64).copy()
    pre_goal[:2] = env_origin[:2] + goal_dir * (float(KICK_PREP_DISTANCE) * float(SCENE_SCALE))
    pre_goal[2] = float(env_origin[2] + float(SOCCER_PRE_GOAL_Z) * float(SCENE_SCALE))

    deep_goal = np.asarray(goal_pos, dtype=np.float64).copy()
    deep_goal[:2] = goal_pos[:2] + goal_dir * (float(KICK_GOAL_DEPTH) * float(SCENE_SCALE))
    deep_goal[2] = float(env_origin[2] + SOCCER_FIELD_TOP_Z)

    p_kick = np.asarray(pre_goal, dtype=np.float64).copy()
    p_kick[:2] = pre_goal[:2] - goal_dir * (float(SOCCER_PKICK_BACKOFF) * float(SCENE_SCALE))
    p_kick[2] = float(env_origin[2] + SOCCER_FIELD_TOP_Z)

    return SoccerTargets(pre_goal=pre_goal, goal=deep_goal, p_kick=p_kick, goal_dir=goal_dir)


def _set_or_add_translate(xform: UsdGeom.Xformable, xyz: tuple[float, float, float]) -> None:
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*xyz))
            return
    xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))


def _set_marker(stage, path: str, xyz: np.ndarray, color: tuple[float, float, float]) -> None:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.GetRadiusAttr().Set(float(SOCCER_PKICK_MARKER_RADIUS))
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    marker_xyz = np.asarray(xyz, dtype=np.float64).copy()
    marker_xyz[2] += float(SOCCER_PKICK_MARKER_RADIUS) * float(SCENE_SCALE)
    _set_or_add_translate(UsdGeom.Xformable(sphere.GetPrim()), tuple(float(v) for v in marker_xyz))


def update_soccer_target_markers(env, prefix: str = "/World/Play") -> SoccerTargets | None:
    """Draw red Pre_Goal, blue Goal, and purple P_kick markers for env 0."""
    base_env = getattr(env, "unwrapped", env)
    if getattr(base_env, "num_envs", 1) != 1 or not hasattr(base_env, "scene"):
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    targets = compute_soccer_targets(base_env)
    _set_marker(stage, f"{prefix}PreGoalMarker", targets.pre_goal, color=(1.0, 0.0, 0.0))
    _set_marker(stage, f"{prefix}GoalMarker", targets.goal, color=(0.0, 0.25, 1.0))
    _set_marker(stage, f"{prefix}PKickMarker", targets.p_kick, color=(0.55, 0.0, 0.85))
    return targets
