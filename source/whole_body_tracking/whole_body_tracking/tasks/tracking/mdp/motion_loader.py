from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch


def validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value}.")
    return value


def validate_int_at_least(name: str, value: int, minimum: int) -> int:
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


class MotionLoader:
    """Load and validate a single reference-motion NPZ for tracking."""

    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        critic_frame_index: int,
        device: str = "cpu",
    ):
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        data = np.load(motion_file)
        required_keys = (
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise KeyError(f"Motion file '{motion_file}' is missing required array(s): {missing_keys}.")
        fps = np.asarray(data["fps"], dtype=np.float32)
        if fps.size == 0 or not np.isfinite(fps).all() or np.any(fps <= 0.0):
            raise ValueError(f"Motion file '{motion_file}' must contain a positive finite fps value.")
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        for name, tensor in (
            ("joint_pos", self.joint_pos),
            ("joint_vel", self.joint_vel),
            ("body_pos_w", self._body_pos_w),
            ("body_quat_w", self._body_quat_w),
            ("body_lin_vel_w", self._body_lin_vel_w),
            ("body_ang_vel_w", self._body_ang_vel_w),
        ):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"Motion array '{name}' in '{motion_file}' contains NaN or Inf.")
        body_indexes_tensor = torch.as_tensor(body_indexes, dtype=torch.long)
        if body_indexes_tensor.numel() == 0:
            raise ValueError("At least one tracked body index is required.")
        self._body_indexes = body_indexes_tensor.to(device=device)
        self.time_step_total = self.joint_pos.shape[0]
        if self.time_step_total <= 1:
            raise ValueError(f"Motion file '{motion_file}' must contain at least two frames.")
        if self.joint_pos.ndim != 2 or self.joint_vel.ndim != 2:
            raise ValueError(f"Motion file '{motion_file}' joint_pos and joint_vel must be rank-2 arrays.")
        for name, tensor in (
            ("joint_vel", self.joint_vel),
            ("body_pos_w", self._body_pos_w),
            ("body_quat_w", self._body_quat_w),
            ("body_lin_vel_w", self._body_lin_vel_w),
            ("body_ang_vel_w", self._body_ang_vel_w),
        ):
            if tensor.shape[0] != self.time_step_total:
                raise ValueError(
                    f"Motion array '{name}' has {tensor.shape[0]} frames, expected {self.time_step_total}."
                )
        for name, tensor in (
            ("body_pos_w", self._body_pos_w),
            ("body_quat_w", self._body_quat_w),
            ("body_lin_vel_w", self._body_lin_vel_w),
            ("body_ang_vel_w", self._body_ang_vel_w),
        ):
            if tensor.ndim != 3:
                raise ValueError(f"Motion array '{name}' in '{motion_file}' must be rank-3.")
        min_body_index = int(body_indexes_tensor.min().item())
        max_body_index = int(body_indexes_tensor.max().item())
        if min_body_index < 0:
            raise ValueError(f"Tracked body indexes must be non-negative, got {min_body_index}.")
        if max_body_index >= self._body_pos_w.shape[1]:
            raise ValueError(
                f"Motion file '{motion_file}' has {self._body_pos_w.shape[1]} bodies, "
                f"but the task requested body index {max_body_index}."
            )
        critic_frame_index = int(critic_frame_index)
        if not 0 <= critic_frame_index < self.time_step_total:
            raise ValueError(
                f"critic_frame_index {critic_frame_index} is outside motion frame range "
                f"[0, {self.time_step_total - 1}] for '{motion_file}'."
            )
        self.critic_frame_index = critic_frame_index

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


def quat_from_x_axis_to_vector(direction: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(direction).all():
        raise FloatingPointError("Direction vector for command orientation contains NaN or Inf.")
    norm = direction.norm(dim=-1, keepdim=True)
    if torch.any(norm <= 1e-6):
        raise ValueError("Direction vector for command orientation has near-zero length.")
    target = direction / norm
    x_axis = torch.zeros_like(target)
    x_axis[..., 0] = 1.0

    cross = torch.cross(x_axis, target, dim=-1)
    dot = torch.sum(x_axis * target, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    quat = torch.cat([1.0 + dot, cross], dim=-1)
    opposite = dot.squeeze(-1) < -0.999999
    if opposite.any():
        quat = quat.clone()
        quat[opposite] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=quat.device, dtype=quat.dtype)

    quat_norm = quat.norm(dim=-1, keepdim=True)
    if torch.any(quat_norm <= 1e-6):
        raise ValueError("Command orientation quaternion has near-zero length.")
    quat = quat / quat_norm
    if not torch.isfinite(quat).all():
        raise FloatingPointError("Command orientation quaternion contains NaN or Inf.")
    return quat
