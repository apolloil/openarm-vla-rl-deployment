"""Gym wrapper for a fixed-orientation EE policy interface.

The underlying Isaac Lab environment expects a 7-dim DiffIK action::

    [dx, dy, dz, axis_angle_x, axis_angle_y, axis_angle_z, binary_grip]

For PPO training and playback we intentionally expose only a 4-dim policy
action::

    [dx, dy, dz, grip]

Rotation deltas are always sent as zero, so the policy controls translation
and gripper only. The initial wrist orientation is baked into the env
``init_state.joint_pos`` (see ``lift/config/vla_env_cfg.py``) so **every**
reset -- both manual ``env.reset()`` and Isaac Lab's internal auto-reset --
starts from the same downward-facing grasp posture.
"""

from __future__ import annotations

import gymnasium as gym
import torch


class EulerEEActionWrapper(gym.Wrapper):
    """Expose a 4D policy action while keeping the env's 7D DiffIK backend.

    Policy action layout::

        [dx, dy, dz, grip]

    Converted env action layout::

        [dx, dy, dz, 0, 0, 0, binary_grip]
    """

    POLICY_ACTION_DIM = 4

    def __init__(self, env: gym.Env):
        super().__init__(env)

        base_single = getattr(env.unwrapped, "single_action_space", None)
        if not isinstance(base_single, gym.spaces.Box):
            raise TypeError("EulerEEActionWrapper expects the base env to expose a Box single_action_space.")

        low = base_single.low[: self.POLICY_ACTION_DIM]
        high = base_single.high[: self.POLICY_ACTION_DIM]
        self.single_action_space = gym.spaces.Box(low=low, high=high, dtype=base_single.dtype)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, env.unwrapped.num_envs)

        self.single_observation_space = getattr(env.unwrapped, "single_observation_space", None)
        self.observation_space = env.observation_space

    def step(self, action: torch.Tensor):
        return self.env.step(self._convert_policy_action(action))

    @classmethod
    def _convert_policy_action(cls, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != cls.POLICY_ACTION_DIM:
            raise ValueError(f"Expected {cls.POLICY_ACTION_DIM} policy actions, got shape {tuple(action.shape)}")

        dpos = action[..., :3]
        grip_logit = action[..., 3:4]
        zeros = torch.zeros_like(dpos)
        binary_grip = torch.where(grip_logit > 0.0, torch.ones_like(grip_logit), -torch.ones_like(grip_logit))
        return torch.cat([dpos, zeros, binary_grip], dim=-1)
