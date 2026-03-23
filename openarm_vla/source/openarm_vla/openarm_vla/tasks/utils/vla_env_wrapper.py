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

"""Unified VLA environment wrapper for OpenArm Isaac Lab tasks.

Usage::

    from openarm_vla.tasks.utils.vla_env_wrapper import make_vla_env

    # Build a Lift-Cube environment ready for VLA inference
    env = make_vla_env("lift", num_envs=1, seed=0, render_mode="rgb_array")

    obs, info = env.reset()
    # vla_action: numpy or torch Tensor, shape (7,) or (1, 7) for unimanual
    obs, rew, terminated, truncated, info = env.step(vla_action)

    env.close()

Supported task names
--------------------
Task name       Mode        Gym ID (play variant)
-----------     ---------   -------------------------------------------------
"reach_bi"      bimanual    Isaac-VLA-Reach-OpenArm-Bi-Play-v0
"reach_uni"     unimanual   Isaac-VLA-Reach-OpenArm-Play-v0
"lift"          unimanual   Isaac-VLA-Lift-Cube-OpenArm-Play-v0
"cabinet"       unimanual   Isaac-VLA-Open-Drawer-OpenArm-Play-v0

Action dimensions
-----------------
Mode        Dims    Layout
---------   ----    -------------------------------------------------------
unimanual    7      [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip]
bimanual    14      [left_arm(6), right_arm(6), left_grip(1), right_grip(1)]
             both arms: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]

All position/rotation deltas are in the robot **base frame** (ZYX Euler
increments, radians).  The wrapper converts Euler deltas → axis-angle before
passing to Isaac Lab's DiffIK.  Gripper values are thresholded at 0.5.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import gymnasium as gym

from .action_transform import euler_delta_to_axis_angle, gripper_to_binary

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, tuple[str, str]] = {
    # task_name → (play_gym_id, mode)
    "reach_bi":  ("Isaac-VLA-Reach-OpenArm-Bi-Play-v0",       "bimanual"),
    "reach_uni": ("Isaac-VLA-Reach-OpenArm-Play-v0",           "unimanual"),
    "lift":      ("Isaac-VLA-Lift-Cube-OpenArm-Play-v0",       "unimanual"),
    "cabinet":   ("Isaac-VLA-Open-Drawer-OpenArm-Play-v0",     "unimanual"),
}

_UNIMANUAL_DIM = 7
_BIMANUAL_DIM  = 14


# ---------------------------------------------------------------------------
# Action conversion helpers
# ---------------------------------------------------------------------------

def _convert_unimanual(action: torch.Tensor) -> torch.Tensor:
    """Convert a (batch, 7) VLA action to Isaac Lab format (batch, 8).

    VLA  : [Δx(0), Δy(1), Δz(2), Δroll(3), Δpitch(4), Δyaw(5), grip(6)]
    Isaac: [arm_action(6), gripper_action(1)]

    The Isaac action vector size is ``6 + 1 = 7`` for the DiffIK arm + binary
    gripper.  Note that the Gym env concatenates the action terms in the order
    they are declared in ActionsCfg: ``arm_action`` then ``gripper_action``.
    """
    assert action.shape[-1] == _UNIMANUAL_DIM
    dpos = action[..., :3]
    aa   = euler_delta_to_axis_angle(action[..., 3], action[..., 4], action[..., 5])
    grip = gripper_to_binary(action[..., 6:7])
    return torch.cat([dpos, aa, grip], dim=-1)   # (batch, 7)


def _convert_bimanual(action: torch.Tensor) -> torch.Tensor:
    """Convert a (batch, 14) VLA action to Isaac Lab format (batch, 14).

    VLA layout  : [left_arm(7), right_arm(7)]
                   each 7: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip]
    Isaac layout: [left_arm_diffik(6), right_arm_diffik(6), left_grip(1), right_grip(1)]
    """
    assert action.shape[-1] == _BIMANUAL_DIM
    left  = action[..., :7]
    right = action[..., 7:]

    left_arm  = torch.cat([left[..., :3],  euler_delta_to_axis_angle(left[...,  3], left[...,  4], left[...,  5])], dim=-1)
    right_arm = torch.cat([right[..., :3], euler_delta_to_axis_angle(right[..., 3], right[..., 4], right[..., 5])], dim=-1)
    left_grip  = gripper_to_binary(left[...,  6:7])
    right_grip = gripper_to_binary(right[..., 6:7])

    return torch.cat([left_arm, right_arm, left_grip, right_grip], dim=-1)  # (batch, 14)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class OpenArmVlaWrapper(gym.Wrapper):
    """Gymnasium wrapper that accepts VLA-format actions and converts them
    to Isaac Lab's DiffIK + binary-gripper format before calling ``env.step``.

    Parameters
    ----------
    env : gym.Env
        The wrapped Isaac Lab environment.
    mode : {"unimanual", "bimanual"}
        Determines the expected action dimensionality (7 or 14) and which
        conversion function to apply.
    device : str | None
        Torch device for internal tensor operations.  Defaults to ``"cpu"``.
    """

    def __init__(self, env: gym.Env, *, mode: str, device: str | None = None):
        super().__init__(env)
        if mode not in ("unimanual", "bimanual"):
            raise ValueError(f"mode must be 'unimanual' or 'bimanual', got '{mode}'")
        self._mode   = mode
        self._device = device or "cpu"
        self._expected_dim = _UNIMANUAL_DIM if mode == "unimanual" else _BIMANUAL_DIM
        self._convert = _convert_unimanual if mode == "unimanual" else _convert_bimanual

    # ── public properties ──────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def expected_action_dim(self) -> int:
        return self._expected_dim

    # ── step / reset passthrough ───────────────────────────────────────────

    def step(self, action: np.ndarray | torch.Tensor) -> tuple:
        """Accept a VLA-format action, convert, and step the underlying env.

        Parameters
        ----------
        action : np.ndarray or torch.Tensor
            Shape ``(dim,)`` or ``(1, dim)`` or ``(batch, dim)``.

        Returns
        -------
        obs, reward, terminated, truncated, info  (standard Gym 5-tuple)
        """
        isaac_action = self._to_isaac(action)
        return self.env.step(isaac_action)

    def reset(self, **kwargs) -> tuple:
        return self.env.reset(**kwargs)

    # ── internal helpers ───────────────────────────────────────────────────

    def _to_isaac(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert VLA action to Isaac Lab format."""
        if isinstance(action, np.ndarray):
            t = torch.from_numpy(action).float().to(self._device)
        else:
            t = action.float().to(self._device)

        if t.dim() == 1:
            t = t.unsqueeze(0)   # add batch dim

        if t.shape[-1] != self._expected_dim:
            raise ValueError(
                f"[OpenArmVlaWrapper] Expected action dim {self._expected_dim} "
                f"for mode='{self._mode}', got {t.shape[-1]}."
            )

        return self._convert(t)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_vla_env(
    task: str,
    *,
    num_envs: int = 1,
    seed: int = 0,
    render_mode: str | None = None,
    device: str | None = None,
    play: bool = True,
) -> OpenArmVlaWrapper:
    """Create an ``OpenArmVlaWrapper`` for the specified task.

    This function:
    1. Imports ``openarm.tasks`` to register all Gym IDs.
    2. Looks up the Gym ID and mode from :data:`TASK_REGISTRY`.
    3. Instantiates the environment via ``gym.make`` (using the built-in
       env-cfg entry point).
    4. Wraps it in :class:`OpenArmVlaWrapper`.

    Parameters
    ----------
    task : str
        One of ``"reach_bi"``, ``"reach_uni"``, ``"lift"``, ``"cabinet"``.
    num_envs : int
        Number of parallel environments.
    seed : int
        Random seed passed to ``env.reset()``.
    render_mode : str | None
        e.g. ``"rgb_array"`` to enable camera rendering.
    device : str | None
        Torch device for VLA action conversion.  Defaults to ``"cpu"``.
    play : bool
        If True (default) use the ``*-Play-v0`` (smaller, no noise) variant.

    Returns
    -------
    OpenArmVlaWrapper

    Examples
    --------
    >>> env = make_vla_env("lift", num_envs=1, seed=42, render_mode="rgb_array")
    >>> obs, _ = env.reset()
    >>> action = np.zeros(7); action[2] = 0.5  # move up
    >>> obs, rew, term, trunc, info = env.step(action)
    >>> env.close()
    """
    if task not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{task}'. Available tasks: {list(TASK_REGISTRY.keys())}"
        )

    # Register openarm_vla gym environments (also triggers openarm base env registration)
    import openarm_vla.tasks  # noqa: F401

    gym_id, mode = TASK_REGISTRY[task]

    # Override num_envs by building the cfg from its entry point
    cfg = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    # Parse "module:Class" string
    if isinstance(cfg, str) and ":" in cfg:
        mod_name, cls_name = cfg.rsplit(":", 1)
        import importlib
        mod = importlib.import_module(mod_name)
        env_cfg = getattr(mod, cls_name)()
    else:
        # cfg is already a callable
        env_cfg = cfg()

    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed

    raw_env = gym.make(gym_id, cfg=env_cfg, render_mode=render_mode)

    return OpenArmVlaWrapper(raw_env, mode=mode, device=device)
