from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def _broadcast_threshold_like_error(threshold: float | torch.Tensor, error: torch.Tensor) -> float | torch.Tensor:
    if not torch.is_tensor(threshold):
        return threshold

    threshold_tensor = cast(torch.Tensor, threshold).to(device=error.device, dtype=error.dtype)
    while threshold_tensor.ndim < error.ndim:
        threshold_tensor = threshold_tensor.unsqueeze(-1)
    return threshold_tensor


def _body_thresholds_like_error(
    threshold: float | dict[str, float] | torch.Tensor, body_names: list[str], error: torch.Tensor
) -> float | torch.Tensor:
    if not isinstance(threshold, dict):
        return _broadcast_threshold_like_error(threshold, error)

    default_threshold = threshold["default"] if "default" in threshold else None
    values = []
    for body_name in body_names:
        body_threshold = threshold[body_name] if body_name in threshold else default_threshold
        if body_threshold is None:
            raise ValueError(f"Missing threshold for body '{body_name}' and no default threshold was provided.")
        values.append(float(body_threshold))
    return torch.tensor(values, device=error.device, dtype=error.dtype).unsqueeze(0)


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float | dict[str, float],
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    selected_body_names = [command.cfg.body_names[idx] for idx in body_indexes]
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    threshold = _body_thresholds_like_error(threshold, selected_body_names, error)
    terminated = torch.any(error > threshold, dim=-1)
    return terminated


def fail_on_non_finite_obs_or_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Raise immediately when policy observations or actions contain NaN or Inf."""
    action = env.action_manager.action  # (num_envs, num_actions)
    if not torch.isfinite(action).all():
        raise FloatingPointError("Policy action contains NaN or Inf.")

    obs_dict = env.observation_manager.compute()
    if "policy" not in obs_dict:
        raise RuntimeError("Policy observation group is missing.")
    policy_obs = obs_dict["policy"]
    if not torch.isfinite(policy_obs).all():
        raise FloatingPointError("Policy observation contains NaN or Inf.")

    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def self_collision(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    body_pairs: list[tuple[str, str, float]],
) -> torch.Tensor:
    """Terminate when any specified non-adjacent body pair comes closer than its threshold.

    Args:
        asset_cfg: Scene entity config for the robot articulation.
        body_pairs: List of (body_a_name, body_b_name, min_distance_m) tuples.
                    Each pair is checked independently with its own distance threshold.

    Returns:
        Boolean tensor of shape (num_envs,), True where self-collision is detected.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_pos = robot.data.body_pos_w  # (E, num_bodies, 3)
    body_names: list[str] = robot.body_names

    colliding = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name_a, name_b, thresh in body_pairs:
        idx_a = body_names.index(name_a)
        idx_b = body_names.index(name_b)
        dist = torch.norm(body_pos[:, idx_a] - body_pos[:, idx_b], dim=-1)
        colliding |= dist < thresh

    return colliding
