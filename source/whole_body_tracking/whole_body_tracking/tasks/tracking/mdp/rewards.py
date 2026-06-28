from __future__ import annotations

import inspect
from functools import wraps
from typing import TYPE_CHECKING, cast

import torch
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _validate_reward(reward: torch.Tensor | float, name: str = "") -> torch.Tensor | float:
    """Raise immediately if a reward term produces non-finite values."""
    if not isinstance(reward, torch.Tensor):
        return reward
    if not torch.isfinite(reward).all():
        raise FloatingPointError(f"Reward term '{name}' produced NaN or Inf values.")
    return reward


def _reward_checked(fn):
    """Decorator: validate the return value of a reward term.
    Preserves the original signature so RewardManager sees the real parameters.
    """

    @wraps(fn)
    def _checked_fn(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return _validate_reward(out, name=fn.__name__)
        return out

    setattr(_checked_fn, "__signature__", inspect.signature(fn))
    return _checked_fn


def _get_body_indexes(command: MotionCommand, body_names: str | list[str] | tuple[str, ...] | None) -> list[int]:
    if body_names is None:
        return list(range(len(command.cfg.body_names)))
    if isinstance(body_names, str):
        body_names = [body_names]
    if len(body_names) == 0:
        raise ValueError("At least one body name must be provided.")
    missing = [name for name in body_names if name not in command.cfg.body_names]
    if missing:
        raise ValueError(f"Unknown body name(s) for motion command: {missing}")
    return [i for i, name in enumerate(command.cfg.body_names) if name in body_names]


def _require_finite_tensor(name: str, tensor: torch.Tensor | None) -> torch.Tensor:
    if tensor is None:
        raise RuntimeError(f"Required tensor '{name}' is missing.")
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Tensor '{name}' contains NaN or Inf.")
    return tensor


def _loco_phase_without_warmup(command: MotionCommand) -> torch.Tensor:
    return (command.time_steps < command.motion.critic_frame_index) & (~command.is_warmup)


def _stand_still_mask(command: MotionCommand) -> torch.Tensor:
    return command.stand_still_env_mask & command.stand_still_start_valid


def _zero_stand_still_envs(reward: torch.Tensor | float, command: MotionCommand) -> torch.Tensor | float:
    if not isinstance(reward, torch.Tensor):
        return reward
    reward_tensor = cast(torch.Tensor, reward)
    if reward_tensor.ndim == 0:
        return reward_tensor
    if reward_tensor.shape[0] != command.stand_still_env_mask.shape[0]:
        return reward_tensor
    if command.stand_still_env_mask.any():
        reward_tensor = reward_tensor.clone()
        reward_tensor[command.stand_still_env_mask] = 0.0
    return reward_tensor


def _body_thresholds_like_error(
    threshold: float | dict[str, float] | torch.Tensor, body_names: list[str], error: torch.Tensor
) -> float | torch.Tensor:
    if not isinstance(threshold, dict):
        if not torch.is_tensor(threshold):
            return threshold
        threshold_tensor = cast(torch.Tensor, threshold).to(device=error.device, dtype=error.dtype)
        while threshold_tensor.ndim < error.ndim:
            threshold_tensor = threshold_tensor.unsqueeze(-1)
        return threshold_tensor

    default_threshold = threshold["default"] if "default" in threshold else None
    values = []
    for body_name in body_names:
        body_threshold = threshold[body_name] if body_name in threshold else default_threshold
        if body_threshold is None:
            raise ValueError(f"Missing threshold for body '{body_name}' and no default threshold was provided.")
        values.append(float(body_threshold))
    return torch.tensor(values, device=error.device, dtype=error.dtype).unsqueeze(0)


@_reward_checked
def my_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large changes between the current and previous action.

    This discourages jerky commands and helps the policy produce smoother actuator targets.
    """
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return torch.clamp(reward, max=50.0)


@_reward_checked
def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float) -> torch.Tensor:
    """Reward the robot anchor for matching the reference motion anchor position in world frame.

    Small errors inside the configured threshold are ignored, then the remaining squared position error
    is converted to an exponential tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    error = torch.clamp(error - (2*std*error_threshold)**2, min=0.0)
    return torch.exp(-error / std**2)


@_reward_checked
def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float) -> torch.Tensor:
    """Reward the robot anchor for matching the reference anchor orientation in world frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    error = torch.clamp(error - (std*error_threshold)**2, min=0.0)
    
    return torch.exp(-error / std**2)


@_reward_checked
def cmd_delta_com_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float) -> torch.Tensor:
    """Reward the robot CoM for staying close to the adapted command target position."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_com_pos_w = env.scene["robot"].data.root_state_w[:, :3]
    error = torch.sum(torch.square(command.adapt_motion_pos - robot_com_pos_w), dim=-1)
    error = torch.clamp(error - (std*error_threshold)**2, min=0.0)
    return torch.exp(-error / std**2)


@_reward_checked
def stand_still_com_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float = 0.0
) -> torch.Tensor:
    """Reward stand-still envs for keeping the global anchor near its reset position."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    mask = _stand_still_mask(command)
    if not torch.any(mask):
        return torch.zeros(env.num_envs, device=env.device)
    error = torch.sum(torch.square(command.stand_still_start_anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    error = torch.clamp(error - (std * error_threshold) ** 2, min=0.0)
    reward = torch.exp(-error / std**2)
    reward[~mask] = 0.0
    return reward


@_reward_checked
def stand_still_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float = 0.0
) -> torch.Tensor:
    """Reward stand-still envs for keeping the anchor orientation near its reset orientation."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    mask = _stand_still_mask(command)
    if not torch.any(mask):
        return torch.zeros(env.num_envs, device=env.device)
    error = quat_error_magnitude(command.stand_still_start_anchor_quat_w, command.robot_anchor_quat_w) ** 2
    error = torch.clamp(error - (std * error_threshold) ** 2, min=0.0)
    reward = torch.exp(-error / std**2)
    reward[~mask] = 0.0
    return reward


@_reward_checked
def stand_still_base_anchor_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    base_lin_weight: float = 1.0,
    base_ang_weight: float = 0.5,
    anchor_lin_weight: float = 1.0,
    anchor_ang_weight: float = 0.5,
    body_names: list[str] | None = None,
    body_lin_weight: float = 0.0,
    body_ang_weight: float = 0.0,
) -> torch.Tensor:
    """Penalize torso/base and anchor velocity in stand-still envs."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    mask = _stand_still_mask(command)
    if not torch.any(mask):
        return torch.zeros(env.num_envs, device=env.device)

    root_state = env.scene["robot"].data.root_state_w
    base_lin_vel = root_state[:, 7:10]
    base_ang_vel = root_state[:, 10:13]
    anchor_lin_vel = command.robot_anchor_lin_vel_w
    anchor_ang_vel = command.robot_anchor_ang_vel_w

    penalty = base_lin_weight * torch.sum(torch.square(base_lin_vel), dim=-1)
    penalty += base_ang_weight * torch.sum(torch.square(base_ang_vel), dim=-1)
    penalty += anchor_lin_weight * torch.sum(torch.square(anchor_lin_vel), dim=-1)
    penalty += anchor_ang_weight * torch.sum(torch.square(anchor_ang_vel), dim=-1)

    if body_names is not None and len(body_names) > 0 and (body_lin_weight != 0.0 or body_ang_weight != 0.0):
        body_indexes = _get_body_indexes(command, body_names)
        lin_vel = command.robot_body_lin_vel_w[:, body_indexes]
        ang_vel = command.robot_body_ang_vel_w[:, body_indexes]
        penalty += body_lin_weight * torch.sum(torch.square(lin_vel), dim=-1).mean(-1)
        penalty += body_ang_weight * torch.sum(torch.square(ang_vel), dim=-1).mean(-1)

    penalty[~mask] = 0.0
    return penalty


@_reward_checked
def cmd_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float) -> torch.Tensor:
    """Reward the robot anchor orientation for following the adapted command orientation."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.adapt_motion_ori, command.robot_anchor_quat_w) ** 2
    error = torch.clamp(error - (std*error_threshold)**2, min=0.0)
    return torch.exp(-error / std**2)

@_reward_checked
def cmd_global_com_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float) -> torch.Tensor:
    """Reward the robot CoM for moving toward the commanded global target position."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_com_pos_w = env.scene["robot"].data.root_state_w[:, :3]
    error = torch.sum(torch.square(command.cmd_target_pos - robot_com_pos_w), dim=-1)
    error = torch.clamp(error - (std*error_threshold)**2, min=0.0)
    return torch.exp(-error / std**2)

@_reward_checked
def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float, body_names: list[str] | None = None, 
) -> torch.Tensor:
    """Reward tracked bodies for matching the reference body positions relative to the anchor.

    This is the main whole-body pose tracking term for positions. During jump-triggered locomotion,
    the term is weakened before the kick frame so the locomotion controller has more freedom.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    error = torch.clamp(error - (2*std*error_threshold)**2, min=0.0)
    reward = torch.exp(-error.mean(-1) / std**2)
    if command.cfg.jump_flag:
        loco_phase = _loco_phase_without_warmup(command)
        reward[loco_phase] /= 10.0
    return reward


@_reward_checked
def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Reward tracked bodies for matching the reference body orientations relative to the anchor.

    This complements the body-position term and uses quaternion angular error for each selected body.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    error = torch.clamp(error - (std*error_threshold)**2, min=0.0)
    reward = torch.exp(-error.mean(-1) / std**2)
    if command.cfg.jump_flag:
        loco_phase = _loco_phase_without_warmup(command)
        reward[loco_phase] /= 10.0
    return reward


@_reward_checked
def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Reward selected bodies for matching reference linear velocities in world frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    if command.cfg.jump_flag:
        loco_phase = _loco_phase_without_warmup(command)
        reward[loco_phase] = 0.0
    return reward


@_reward_checked
def motion_global_feet_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Reward selected feet for matching reference linear velocities in world frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    if command.cfg.jump_flag:
        loco_phase = _loco_phase_without_warmup(command)
        reward[loco_phase] = 0.0
    return reward

@_reward_checked
def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, error_threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Reward selected bodies for matching reference angular velocities in world frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    if command.cfg.jump_flag:
        loco_phase = (command.time_steps < 50) & (~command.is_warmup)
        reward[loco_phase] = 0.0
    return reward


@_reward_checked
def feet_contact_time(env: ManagerBasedRLEnv, command_name: str, body_names: list[str], threshold: float = 0.5) -> torch.Tensor:
    """Reward feet that leave contact after a short recent-contact window.

    This encourages stepping behavior by scoring first-air events when the previous contact duration
    was below the threshold.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    sensor_cfg = SceneEntityCfg(name="contact_forces", body_names=body_names)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


@_reward_checked
def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_names: list[str],
    threshold: float = 0.25,
    saturate_at: float | None = None,
    contact_force_threshold: float = 1.0,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward feet that stayed in the air long enough before touchdown.

    Tracks per-foot swing duration, rewards on first touchdown, and uses a simple
    filtered contact state (`contact OR last_contact`) to reduce mesh-contact noise.
    """
    command = env.command_manager.get_term(command_name)
    sensor_cfg = SceneEntityCfg(name="contact_forces", body_names=body_names)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    fmat = _require_finite_tensor("contact_forces.net_forces_w", contact_sensor.data.net_forces_w)[
        :, sensor_cfg.body_ids
    ]

    # Isaac Lab exposes world-frame net contact forces; this mirrors the Isaac Gym
    # implementation's vertical-force contact test.
    contact = fmat[..., 2] > contact_force_threshold
    num_feet = contact.shape[1]

    body_key = tuple(body_names)
    state = command._feet_air_time_state.get(body_key)
    buffer_shape = (env.num_envs, num_feet)
    if (
        state is None
        or state["air_time_buf"].shape != buffer_shape
        or state["last_contacts"].shape != buffer_shape
    ):
        state = {
            "air_time_buf": torch.zeros(buffer_shape, dtype=torch.float32, device=env.device),
            "last_contacts": torch.zeros(buffer_shape, dtype=torch.bool, device=env.device),
        }
        command._feet_air_time_state[body_key] = state
    last_contacts = state["last_contacts"]
    feet_air_time_buf = state["air_time_buf"]

    # Reset the temporal buffers for freshly reset envs.
    just_reset = command.real_time_steps <= 1
    if torch.any(just_reset):
        last_contacts = last_contacts.clone()
        feet_air_time_buf = feet_air_time_buf.clone()
        last_contacts[just_reset] = False
        feet_air_time_buf[just_reset] = 0.0

    contact_filt = torch.logical_or(contact, last_contacts)
    first_contact = (feet_air_time_buf > 0.0) & contact_filt

    feet_air_time_buf = feet_air_time_buf + env.step_dt
    reward = torch.sum((feet_air_time_buf - threshold) * first_contact.float(), dim=-1)

    command_mag = torch.norm(command.adapt_motion_pos[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)
    stable_phase = command.time_steps > (command.motion.critic_frame_index + command.cfg.kick_hold_steps)
    reward = reward * (command_mag > command_threshold).float() * (1-stable_phase.float())

    feet_air_time_buf = feet_air_time_buf * (~contact_filt).float()
    state["last_contacts"] = contact
    state["air_time_buf"] = feet_air_time_buf

    return reward



@_reward_checked
def action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the second derivative of actions.

    This discourages rapid changes in action velocity, which produces smoother motion than a
    first-order action-rate penalty alone.
    """
    command: MotionCommand = env.command_manager.get_term("motion")
    action_diff = env.action_manager.action - env.action_manager.prev_action
    action_diff2 = action_diff - (env.action_manager.prev_action - command.prev_prev_action)
    
    rew = torch.clamp(torch.sum(torch.square(action_diff2), dim=-1), 0, 10)
    return rew


@_reward_checked
def feet_slip(env: ManagerBasedRLEnv, command_name: str, body_names: list[str], sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:       #  -0.2
    """Penalize horizontal foot motion while the foot is in contact with the ground.

    Only feet with contact force above the threshold contribute, so swing feet are not punished.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fmat = _require_finite_tensor("contact_forces.net_forces_w", contact_sensor.data.net_forces_w)[
        :, sensor_cfg.body_ids
    ]

    # Penalize each foot's tangential speed only while that foot is supporting weight.
    feet_indices = _get_body_indexes(env.command_manager.get_term(command_name), body_names)
    foot_speed_xy = torch.norm(command.robot_body_lin_vel_w[:, feet_indices, :2], dim=-1)
    foot_contact = fmat.norm(dim=-1) > threshold

    if foot_speed_xy.shape[1] != foot_contact.shape[1]:
        raise RuntimeError(
            f"feet_slip body/contact mismatch: {foot_speed_xy.shape[1]} body velocities vs "
            f"{foot_contact.shape[1]} contact-force entries."
        )

    slip_speed = torch.where(foot_contact, foot_speed_xy, torch.zeros_like(foot_speed_xy))
    rew = torch.sum(torch.square(slip_speed), dim=-1)
    return rew


@_reward_checked
def feet_contact_force_over_threshold(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 350.0,
    contact_force_scale: float = 0.05,
    max_penalty: float = 2.0,
) -> torch.Tensor:
    """Penalize per-foot contact force magnitudes above a threshold.

    This mirrors the Isaac Gym-style feet contact-force penalty using Isaac Lab's
    ContactSensor net forces for the selected foot bodies.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fmat = _require_finite_tensor("contact_forces.net_forces_w", contact_sensor.data.net_forces_w)[
        :, sensor_cfg.body_ids
    ]

    excess_force = torch.clamp(torch.norm(fmat, dim=-1) - threshold, min=0.0)
    penalty = torch.sum(torch.square(contact_force_scale * excess_force), dim=1)
    return torch.clamp(penalty, max=max_penalty)


@_reward_checked
def locomotion_phase_feet_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_names: list[str],
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.08,
    threshold: float = 1.0,
    max_penalty: float = 0.2,
) -> torch.Tensor:
    """Penalize swing feet that do not clear a target height during locomotion.

    This adapts the phase/contact-based clearance idea to this environment:
    feet in contact are ignored, feet not in contact are expected to be at least
    ``target_height`` above the env origin. The term is active only before the
    kick frame so it does not fight the kicking motion.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    locomotion_phase = command.time_steps < command.motion.critic_frame_index
    if not torch.any(locomotion_phase):
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fmat = _require_finite_tensor("contact_forces.net_forces_w", contact_sensor.data.net_forces_w)[
        :, sensor_cfg.body_ids
    ]

    feet_indices = _get_body_indexes(command, body_names)
    foot_height = command.robot_body_pos_w[:, feet_indices, 2] - env.scene.env_origins[:, 2:3]
    foot_contact = fmat.norm(dim=-1) > threshold

    if foot_height.shape[1] != foot_contact.shape[1]:
        raise RuntimeError(
            f"locomotion_phase_feet_clearance body/contact mismatch: {foot_height.shape[1]} body heights vs "
            f"{foot_contact.shape[1]} contact-force entries."
        )
    swing_mask = ~foot_contact

    clearance_error = torch.clamp(target_height - foot_height, min=0.0)
    penalty = torch.sum(torch.square(clearance_error) * swing_mask.float(), dim=-1)
    penalty = torch.clamp(penalty, max=max_penalty)
    return penalty

@_reward_checked
def robot_alive(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Provide a survival reward, with a larger value after the kick recovery window."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    post_mask = command.time_steps > (command.motion.critic_frame_index + 50)
    rew = torch.ones(env.num_envs, device=env.device)
    rew[post_mask] = 10.0
    return rew


@_reward_checked
def no_fly(env: ManagerBasedRLEnv, command_name: str, body_names: list[str]) -> torch.Tensor:
    """Penalize states where all selected feet are simultaneously above the ground threshold."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_index = _get_body_indexes(command, body_names)
    feet_height = command.robot_body_pos_w[:, feet_index, 2]
    fly_mask = (feet_height > 0.05).float()
    both_fly_mask = (torch.sum(fly_mask, dim=-1) == len(body_names))
    return both_fly_mask.float()


@_reward_checked
def locomotion_phase_orientation_l2(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Penalize non-flat base orientation."""
    _ = command_name
    robot_quat = env.scene["robot"].data.root_state_w[:, 3:7]
    gravity_vec = torch.tensor([0.0, 0.0, -9.81], device=env.device).repeat(env.num_envs, 1)
    projected_gravity = math_utils.quat_apply_inverse(robot_quat, gravity_vec)
    reward = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    return reward


@_reward_checked
def locomotion_phase_lin_vel_z_l2(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Penalize base vertical velocity."""
    _ = command_name
    base_lin_vel_z = env.scene["robot"].data.root_state_w[:, 7:10][:, 2]
    reward = torch.square(base_lin_vel_z)
    return reward


@_reward_checked
def locomotion_phase_torso_orientation_l2(
    env: ManagerBasedRLEnv, command_name: str, torso_body_name: str = "torso_link"
) -> torch.Tensor:
    """Penalize non-flat torso orientation."""
    _ = command_name
    torso_body_idx = env.scene["robot"].body_names.index(torso_body_name)
    torso_quat = env.scene["robot"].data.body_quat_w[:, torso_body_idx, :]
    gravity_vec = torch.tensor([0.0, 0.0, -9.81], device=env.device).repeat(env.num_envs, 1)
    torso_projected_gravity = math_utils.quat_apply_inverse(torso_quat, gravity_vec)
    reward = torch.sum(torch.square(torso_projected_gravity[:, :2]), dim=1)
    return reward


@_reward_checked
def unstable_penalty(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Penalize robot instability after the kick phase has finished.

    The term activates after the kick hold window and penalizes:
    - torso tilt away from upright,
    - base planar velocity,
    - base angular velocity.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    stable_phase = (command.time_steps > (command.motion.critic_frame_index + command.cfg.kick_hold_steps)) | _stand_still_mask(command)
    if not torch.any(stable_phase):
        return torch.zeros(env.num_envs, device=env.device)
    root_state = env.scene["robot"].data.root_state_w
    base_lin_vel_xy = root_state[:, 7:9]
    base_ang_vel = root_state[:, 10:13]
    base_lin_penalty = torch.sum(torch.square(base_lin_vel_xy), dim=1)
    base_ang_penalty = torch.sum(torch.square(base_ang_vel), dim=1)

    penalty = base_lin_penalty + 0.5 * base_ang_penalty
    penalty[~stable_phase] = 0.0
    return penalty


@_reward_checked
def stable_anchor_pos_tracking(
    env: ManagerBasedRLEnv, command_name: str, xy_weight: float = 1.0, z_weight: float = 0.25, std: float = 0.1
) -> torch.Tensor:
    """Penalize anchor displacement from the recorded kick-frame stabilization pose."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    stable_phase = command.time_steps > (command.motion.critic_frame_index+20)
    if not torch.any(stable_phase):
        return torch.zeros(env.num_envs, device=env.device)

    pos_delta = command.robot_anchor_pos_w - command.stabilize_anchor_pos_w
    pos_penalty_xy = torch.sum(torch.square(pos_delta[:, :2]), dim=1)
    pos_penalty_z = torch.square(pos_delta[:, 2])
    penalty = xy_weight * pos_penalty_xy + z_weight * pos_penalty_z
    
    exp_penalty = torch.exp(-penalty / (std ** 2))
    exp_penalty[~stable_phase] = 0.0
    return exp_penalty


@_reward_checked
def dof_vel(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Penalize mechanical power by summing joint velocity times applied torque."""
    robot_dof_vel = env.scene["robot"].data.joint_vel
    robot_torque = env.scene["robot"].data.computed_torque
    error = torch.sum(robot_dof_vel*robot_torque, dim=-1)
    return error / std


@_reward_checked
def torque(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Penalize large actuator torques."""
    robot_torque = env.scene["robot"].data.computed_torque
    error = torch.sum(torch.square(robot_torque), dim=-1)
    return error / std

@_reward_checked
def error_ball_to_target(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward episodes where the ball's best distance to the target is small."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error_min = command.round_min_distance_to_target
    reward = torch.exp(-(error_min)**2 / (std**2))
    return _zero_stand_still_envs(reward, command)


@_reward_checked
def shot_target_success(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Persistent reward equal to the number of successful shots so far in the episode."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _zero_stand_still_envs(command.episode_shot_success_count.float(), command)


@_reward_checked
def ee_body_pos_termination_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float | dict[str, float],
    warmup_threshold: float = 0.05,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Return 1.0 on the same step where the EE body-position termination fires."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    selected_body_names = [command.cfg.body_names[idx] for idx in body_indexes]
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    threshold = _body_thresholds_like_error(threshold, selected_body_names, error)
    terminated = torch.any(error > threshold, dim=-1)

    warmup_error = torch.norm(
        command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1
    )
    warmup_threshold = _body_thresholds_like_error(warmup_threshold, selected_body_names, warmup_error)
    warmup_terminated = torch.any(warmup_error > warmup_threshold, dim=-1) & command.is_warmup
    return (terminated | warmup_terminated).float()


@_reward_checked
def goal_reward_burst(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Short reward burst triggered by a newly scored goal."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    failed = env.termination_manager.terminated
    if torch.any(failed & command.goal_reward_active_flag):
        command.clear_goal_reward_burst(failed)
    return _zero_stand_still_envs(command.goal_reward_active_flag.float(), command)


@_reward_checked
def cmd_velocity_tracking(
    env: ManagerBasedRLEnv, command_name: str, max_vel: float = 0.5, std: float = 1.0
) -> torch.Tensor:
    """Reward base velocity that points from the current anchor toward the adapted target.

    The desired speed scales with remaining distance and is capped by ``max_vel``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_vel = env.scene["robot"].data.root_state_w[:, 7:9]
    to_target = command.adapt_motion_pos[:, :2] - command.robot_anchor_pos_w[:, :2]
    dist = to_target.norm(dim=-1, keepdim=True).clamp(min=1e-6, max=1.0)
    direction = to_target / dist
    nominal_vel = direction * max_vel
    track_vel = dist / 1.0 * nominal_vel
    reward = torch.exp(-(robot_vel - track_vel).norm(dim=-1) ** 2 / (std ** 2))
    return reward


@_reward_checked
def penalize_weak_foot_contact(env: ManagerBasedRLEnv, command_name: str, threshold: float, std: float) -> torch.Tensor:
    """Penalize the non-kicking foot for getting close enough to contact the ball.

    This keeps the shot focused on the configured main foot.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    main_foot_name = command.cfg.main_foot_name
    weak_foot_name = "left_ankle_roll_link" if main_foot_name == "right_ankle_roll_link" else "right_ankle_roll_link"
    body_ids = _get_body_indexes(command, [weak_foot_name])
    foot_ball_distance = torch.norm(command.robot_body_pos_w[:, body_ids, :3].squeeze(1) - command.ball_pos, dim=-1)
    reward = torch.exp(-torch.square(foot_ball_distance - threshold) / (std**2))

    return _zero_stand_still_envs(reward, command)

@_reward_checked
def penalize_self_contact_feet(env: ManagerBasedRLEnv, command_name: str, body_names: list[str], threshold: float, std: float) -> torch.Tensor:
    """Penalize the feet for coming too close to each other."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_ids = _get_body_indexes(command, body_names)
    if len(body_ids) != 2:
        raise ValueError(f"penalize_self_contact_feet expects exactly 2 bodies, got {len(body_ids)}.")
    foot_poss = command.robot_body_pos_w[:, body_ids, :3]
    left_right_diff = torch.norm(foot_poss[:, 0] - foot_poss[:, 1], dim=-1)
    start_penalizing = left_right_diff < threshold
    reward = torch.zeros_like(left_right_diff)
    reward[start_penalizing] = 10.0 * (1.0 - torch.exp(-torch.square(left_right_diff[start_penalizing] - threshold) / (std**2)))
    return reward


@_reward_checked
def robot_ball_contact(env: ManagerBasedRLEnv, goal_sigma, feet_sigma, force_threshold, vel_threshold, command_name: str,
                       sensor_cfg_name: str) -> torch.Tensor:
    """Reward useful foot-ball interaction using contact proximity, shot accuracy, and ball speed.

    The contact component encourages the main foot to reach the ball before impact. After contact,
    the goal-distance and velocity components encourage the ball to travel quickly toward the target.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)

    if force_threshold <= 0.0:
        raise ValueError("force_threshold must be positive.")

    body_ids = _get_body_indexes(command, command.cfg.main_foot_name)
    right_ankle_roll_pos = command.robot_body_pos_w[:, body_ids, :3].squeeze(1)
    ball_pos = command.ball_pos.clone()

    r_contact = torch.exp(-(torch.clamp(torch.norm(right_ankle_roll_pos - ball_pos, dim=-1)-0.115, min=1e-8, max=10)) / (feet_sigma))
    contact_mask = command.contacted_flag
    if contact_mask.any(): r_contact[contact_mask] = 1.0
    goal_mask = command.round_min_distance_to_target < 0.5*goal_sigma
    min_distance_to_goal = torch.clamp(command.round_min_distance_to_target-0.5*goal_sigma, min=0, max=10)
    r_goal = torch.exp(-min_distance_to_goal**2 / (2*goal_sigma)**2)
    if goal_mask.any(): r_goal[goal_mask] = 1.0
    
    ball_velocity = torch.clamp(command.ball_velocity.norm(dim=-1), min=vel_threshold, max=10.0)

    r_vel = 1 - torch.clamp(torch.exp( -(ball_velocity - vel_threshold)**2/(10.0)), max=1.0, min = 0.0)

    force_sensor: ContactSensor = env.scene.sensors[sensor_cfg_name]
    data = force_sensor.data
    fm_hist = data.force_matrix_w_history
    if fm_hist is not None:
        fmh = _require_finite_tensor(f"{sensor_cfg_name}.force_matrix_w_history", fm_hist)
        pair_norm = torch.norm(fmh, dim=-1)
        f_norm = pair_norm.amax(dim=(1, 2, 3))
    elif data.force_matrix_w is not None:
        fm = _require_finite_tensor(f"{sensor_cfg_name}.force_matrix_w", data.force_matrix_w)
        f_norm = torch.norm(fm, dim=-1).amax(dim=(1, 2))
    else:
        raise RuntimeError(
            f"Contact sensor '{sensor_cfg_name}' must provide force_matrix_w_history or force_matrix_w."
        )
    r_force = torch.clamp(f_norm / float(force_threshold), max=1.0)

    reward = (r_contact + r_goal) * (r_vel + r_force) / 4.0
    return _zero_stand_still_envs(reward, command)


@_reward_checked
def robot_ball_contact_count(env: ManagerBasedRLEnv, command_name: str, sensor_cfg_name: str) -> torch.Tensor:
    """Reward environments that have registered any robot-ball contact."""

    command = env.command_manager.get_term(command_name)
    return _zero_stand_still_envs(command.contacted_flag.float(), command)


@_reward_checked
def ball_over_line(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward the ball crossing the goal line and penalize it going behind the robot."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ball_root_pos = env.scene["soccer"].data.root_state_w[:, :3]
    env_origins = env.scene.env_origins[:, :3]
    over_line = ball_root_pos[:, 1]>(7.0+env_origins[:,1])
    over_back_line = ball_root_pos[:, 1]<( -1.0 + env_origins[:,1])
    reward = 2*over_line.float() - over_back_line.float()
    return _zero_stand_still_envs(reward, command)


@_reward_checked
def ball_velocity(env: ManagerBasedRLEnv, command_name: str, sensor_cfg_name: str, force_threshold: float, std: float) -> torch.Tensor:
    """Reward ball speed after contact, ignoring speeds below the configured minimum."""
    _ = sensor_cfg_name, force_threshold
    command: MotionCommand = env.command_manager.get_term(command_name)
    ball_root_vel = env.scene["soccer"].data.root_state_w[:, 7:10]
    min_vel_threshold = abs(command.cfg.ball_velocity_range[1])
    if command.contacted_flag.any() == False:
        return torch.zeros(env.num_envs, device=env.device)
    speeds = torch.norm(ball_root_vel, dim=-1)
    speeds = torch.clamp(speeds - min_vel_threshold, min=0.0, max=10.0)
    # should give a threshold for max speed
    speed_reward = 1 - 1.0 / (1 + speeds / std)
    return _zero_stand_still_envs(speed_reward, command)


@_reward_checked
def ball_contact_orientation(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg_name: str, contact_window_steps: int = 5
) -> torch.Tensor:
    """Reward ball velocity toward the target only for a short window after foot-ball contact."""
    _ = sensor_cfg_name, contact_window_steps
    command: MotionCommand = env.command_manager.get_term(command_name)

    contact_window = command.contacted_flag
    if not contact_window.any():
        return torch.zeros(env.num_envs, device=env.device)

    delta_xy = command.target_pos_r[:, :2] - command.contact_ball_pos_r[:, :2]
    shoot_orientation_xy = delta_xy / (torch.norm(delta_xy, dim=-1, keepdim=True) + 1e-6)

    ball_root_vel = env.scene["soccer"].data.root_state_w[:, 7:10]
    ball_vel_proj_on_shoot_ori = torch.sum(ball_root_vel[:, :2] * shoot_orientation_xy, dim=-1)
    ball_vel_proj = torch.clamp(ball_vel_proj_on_shoot_ori, min=-10.0, max=10.0)
    vel_reward_xy = ball_vel_proj / 10.0
    
    delta_xyz = command.target_pos_r[:, :3] - command.contact_ball_pos_r[:, :3]
    shoot_orientation_xyz = delta_xyz / (torch.norm(delta_xyz, dim=-1, keepdim=True) + 1e-6)
    ball_vel_proj_on_shoot_ori_xyz = torch.sum(ball_root_vel[:, :3] * shoot_orientation_xyz, dim=-1)
    ball_vel_proj_xyz = torch.clamp(ball_vel_proj_on_shoot_ori_xyz, min=-15.0, max=15.0)
    vel_reward_xyz = ball_vel_proj_xyz / 15.0
    
    vel_reward = 0.5*vel_reward_xy + 0.5*vel_reward_xyz

    reward = torch.where(contact_window, vel_reward, torch.zeros_like(vel_reward))

    fail = torch.where(reward < 0.0)
    reward[fail] = 0.1 * reward[fail]
    return _zero_stand_still_envs(reward, command)


# before contact

@_reward_checked
def robot_feet_ball_distance(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward the main foot for approaching the ball before the post-kick stabilization phase.

    Distances inside the ball radius are treated as contact-level proximity. Once contact is detected,
    the distance is clamped to the best value so the approach reward stays saturated.
    """
    
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    robot: Articulation = env.scene["robot"]
    ball_root_pos = env.scene["soccer"].data.root_state_w[:, :3]
    body_names = command.cfg.main_foot_name
    body_ids = _get_body_indexes(command, body_names)
    body_feet_pos = command.robot_body_pos_w[:, body_ids, :3].squeeze(1)
    dist = torch.norm(body_feet_pos - ball_root_pos, dim=-1)

    contacted_flag = torch.where(command.contacted_flag)
    dist[contacted_flag] = 0.115
    clamped_dist = torch.clamp(dist, min=0.115)-0.115
    reward = 1./(1. + torch.square(clamped_dist / std))
    
    stable_phase = command.time_steps > (command.motion.critic_frame_index + command.cfg.kick_hold_steps)
    if stable_phase.any():
        reward[stable_phase] = 0.0
    return _zero_stand_still_envs(reward, command)


@_reward_checked
def robot_com_ball_distance(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward the robot CoM for staying close to the ball in the horizontal plane before recovery.

    This encourages the whole body to move into a useful striking position, not only the foot.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    robot: Articulation = env.scene["robot"]
    ball_root_pos = env.scene["soccer"].data.root_state_w[:, :3]
    robot_com_pos = robot.data.root_state_w[:, :3]
    dist = torch.norm(robot_com_pos[:, :2] - ball_root_pos[:, :2], dim=-1)

    contacted_flag = command.contacted_flag
    if contacted_flag.any():
        dist[contacted_flag] = 0.25

    clamped_dist = torch.clamp(dist, min=0.25)-0.25
    reward = 1./(1. + torch.square(clamped_dist / std))

    stable_phase = command.time_steps > (command.motion.critic_frame_index + command.cfg.kick_hold_steps)
    if stable_phase.any():
        reward[stable_phase] = 0.0
    return _zero_stand_still_envs(reward, command)


@_reward_checked
def robot_torso_ball_distance(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Reward the torso or selected upper-body links for staying close to the ball before recovery.

    This regularizes body placement around the ball and can reduce awkward reaches from the kicking leg.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    robot: Articulation = env.scene["robot"]
    ball_root_pos = env.scene["soccer"].data.root_state_w[:, :3]
    if body_names is None:
        body_names = ["torso_link"]
    body_ids = [robot.body_names.index(body_name) for body_name in body_names]
    torso_pos = robot.data.body_state_w[:, body_ids, :3].mean(dim=1)
    dist = torch.norm(torso_pos[:, :2] - ball_root_pos[:, :2], dim=-1)
    
    contacted_flag = command.contacted_flag
    if contacted_flag.any():
        dist[contacted_flag] = 0.3

    clamped_dist = torch.clamp(dist, min=0.3) - 0.3
    reward = 1.0 / (1.0 + torch.square(clamped_dist / std))

    stable_phase = command.time_steps > (command.motion.critic_frame_index + command.cfg.kick_hold_steps)
    if stable_phase.any():
        reward[stable_phase] = 0.0
    return _zero_stand_still_envs(reward, command)
# ---------------------------------------------------------------------------
# Arm default-pose regularisation
# ---------------------------------------------------------------------------

@_reward_checked
def arm_default_pose_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_joint_names: list[str] | None = None,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward arm joints for staying near their default positions.

    Elbow joints use a task-specific 1.0 rad target instead of the URDF default.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    if arm_joint_names is None:
        arm_joint_names = [
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ]
    joint_ids, joint_names = robot.find_joints(arm_joint_names)
    cur = robot.data.joint_pos[:, joint_ids]
    default = robot.data.default_joint_pos[:, joint_ids].clone()
    elbow_joint_ids = [idx for idx, joint_name in enumerate(joint_names) if joint_name.endswith("_elbow_joint")]
    
    error_sq = (cur - default).pow(2)
    reward = -error_sq  # (E, J)
    reward[:, elbow_joint_ids] *= 5.0
    return reward.mean(dim=-1)                    # (E,)


@_reward_checked
def hand_height_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_names: list[str] | None = None,
    max_height: float = 1.15,
) -> torch.Tensor:
    """Penalize hands that rise too high during locomotion."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if body_names is None:
        body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    body_indexes = _get_body_indexes(command, body_names)

    hand_height = command.robot_body_pos_w[:, body_indexes, 2] - env.scene.env_origins[:, 2:3]
    excess_height = torch.clamp(hand_height - float(max_height), min=0.0)
    penalty = torch.sum(torch.square(excess_height), dim=-1)
    return _zero_stand_still_envs(penalty, command)


@_reward_checked
def arm_pitch_same_sign_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_joint_name: str = "left_shoulder_pitch_joint",
    right_joint_name: str = "right_shoulder_pitch_joint",
    deadband: float = 0.03,
) -> torch.Tensor:
    """Penalize left/right shoulder pitch offsets when they move in the same direction from default."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints([left_joint_name, right_joint_name], preserve_order=True)
    if len(joint_ids) != 2:
        raise ValueError(
            "arm_pitch_same_sign_penalty expects both shoulder pitch joints "
            f"({left_joint_name!r}, {right_joint_name!r}); found {len(joint_ids)}."
        )

    offsets = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
    active_offsets = torch.where(
        offsets.abs() > float(deadband),
        offsets,
        torch.zeros_like(offsets),
    )
    same_sign = active_offsets[:, 0] * active_offsets[:, 1] > 0.0
    penalty = torch.minimum(active_offsets[:, 0].abs(), active_offsets[:, 1].abs()).pow(2)
    return _zero_stand_still_envs(torch.where(same_sign, penalty, torch.zeros_like(penalty)), command)
