"""Gym wrapper for a fixed-orientation 4D EE soccer action interface.

The underlying Isaac Lab environment expects a 7-dim DiffIK action::

    [dx, dy, dz, axis_angle_x, axis_angle_y, axis_angle_z, binary_grip]

For soccer PPO training we expose a 4-dim policy action::

    [dx, dy, dz, grip]

The policy still controls only translation and gripper.  Rotation is filled by
the wrapper as a small correction back to the reset-time wrist orientation, so
contacts and IK numerical drift do not accumulate into a visibly tilted gripper.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


class Push3DEEActionWrapper(gym.Wrapper):
    """Expose a 4D policy action while keeping the env's 7D DiffIK backend.

    Policy action layout::

        [dx, dy, dz, grip]

    Converted env action layout::

        [dx, dy, dz, rot_correction_xyz, grip]

    ``rot_correction_xyz`` is computed from the current EE quaternion to the
    reset-time EE quaternion and clipped to a conservative per-step magnitude.
    """

    POLICY_ACTION_DIM = 4
    ORIENTATION_CORRECTION_GAIN = 0.35
    ORIENTATION_CORRECTION_MAX_NORM = 0.15

    def __init__(self, env: gym.Env):
        super().__init__(env)

        base_single = getattr(env.unwrapped, "single_action_space", None)
        if not isinstance(base_single, gym.spaces.Box):
            raise TypeError(
                "Push3DEEActionWrapper expects the base env to expose a Box "
                "single_action_space."
            )

        # Expose first 3 dims (pos) + last dim (grip) from the 7D space.
        low = np.concatenate([base_single.low[:3], base_single.low[-1:]])
        high = np.concatenate([base_single.high[:3], base_single.high[-1:]])
        self.single_action_space = gym.spaces.Box(low=low, high=high, dtype=base_single.dtype)
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, env.unwrapped.num_envs
        )

        self.single_observation_space = getattr(env.unwrapped, "single_observation_space", None)
        self.observation_space = env.observation_space
        self._target_ee_quat_w: torch.Tensor | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._capture_target_orientation()
        return obs, info

    def step(self, action: torch.Tensor | np.ndarray):
        device = getattr(self.env.unwrapped, "device", None)
        return self.env.step(self._convert_policy_action(action, device=device))

    @staticmethod
    def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
        return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dim=-1,
        )

    @staticmethod
    def _quat_to_axis_angle(quat: torch.Tensor) -> torch.Tensor:
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
        vec = quat[..., 1:]
        vec_norm = vec.norm(dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(vec_norm, quat[..., :1].clamp(min=1e-9))
        axis = vec / vec_norm.clamp(min=1e-9)
        return axis * angle

    def _capture_target_orientation(self) -> None:
        base_env = self.env.unwrapped
        if not hasattr(base_env, "scene") or "ee_frame" not in base_env.scene.keys():
            self._target_ee_quat_w = None
            return
        quat = base_env.scene["ee_frame"].data.target_quat_w[:, 0, :4]
        self._target_ee_quat_w = quat.detach().clone()

    def _orientation_correction(self, batch_shape: torch.Size, device: str | torch.device | None) -> torch.Tensor:
        base_env = self.env.unwrapped
        if not hasattr(base_env, "scene") or "ee_frame" not in base_env.scene.keys():
            return torch.zeros((*batch_shape, 3), dtype=torch.float32, device=device)
        current_quat = base_env.scene["ee_frame"].data.target_quat_w[:, 0, :4]
        if self._target_ee_quat_w is None or self._target_ee_quat_w.shape != current_quat.shape:
            self._target_ee_quat_w = current_quat.detach().clone()

        target_quat = self._target_ee_quat_w.to(device=current_quat.device, dtype=current_quat.dtype)
        q_err = self._quat_mul(target_quat, self._quat_conjugate(current_quat))
        correction = self._quat_to_axis_angle(q_err) * float(self.ORIENTATION_CORRECTION_GAIN)
        norm = correction.norm(dim=-1, keepdim=True)
        max_norm = float(self.ORIENTATION_CORRECTION_MAX_NORM)
        correction = correction * (max_norm / norm.clamp(min=max_norm))
        if device is not None:
            correction = correction.to(device)
        return correction

    def _convert_policy_action(
        self,
        action: torch.Tensor | np.ndarray,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        if device is not None:
            action = action.to(device)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] != self.POLICY_ACTION_DIM:
            raise ValueError(
                f"Expected {self.POLICY_ACTION_DIM} policy actions, got shape "
                f"{tuple(action.shape)}"
            )
        dpos       = action[..., :3]                           # [dx, dy, dz]
        grip_logit = action[..., 3:4]                         # raw policy grip output
        rot        = self._orientation_correction(dpos.shape[:-1], device=device)
        # Binarise the grip so action_manager.action always stores ±1.
        # This prevents the raw unbounded logit from inflating action_rate_l2.
        binary_grip = torch.where(
            grip_logit > 0.0,
            torch.ones_like(grip_logit),
            -torch.ones_like(grip_logit),
        )
        return torch.cat([dpos, rot, binary_grip], dim=-1)
