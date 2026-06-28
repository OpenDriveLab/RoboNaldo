from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.adapt_motion_pos,
        command.adapt_motion_ori,
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori_rel_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.adapt_motion_pos,
        command.adapt_motion_ori,
    )
    mat = matrix_from_quat(ori_rel_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def soccer_robot_relative_pos_w(env: ManagerBasedEnv) -> torch.Tensor:
    robot_pos = env.scene["robot"].data.root_state_w[:, :3]
    soccer_pos = env.scene["soccer"].data.root_state_w[:, :3]
    rel_pos = soccer_pos - robot_pos
    rel_pos = torch.clamp(rel_pos, -8.0, 8.0)
    return rel_pos.view(env.num_envs, -1)

def target_robot_relative_pos_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    target_pos = command.target_pos
    robot_pos = env.scene["robot"].data.root_state_w[:, :3]
    rel_pos = target_pos - robot_pos
    rel_pos = torch.clamp(rel_pos, -8.0, 8.0)
    return rel_pos.view(env.num_envs, -1)

def soccer_robot_relative_pos_b(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    rel_pos_b = command.apply_lidar_stale_ball_pos_b(command.ball_pos_b)
    rel_pos_b = torch.clamp(rel_pos_b, -8.0, 8.0)

    return rel_pos_b.view(env.num_envs, -1)

def target_robot_relative_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    target_pos = command.target_pos
    robot_pos = env.scene["robot"].data.root_state_w[:, :3]
    rel_pos = target_pos - robot_pos
    robot_quat = env.scene["robot"].data.root_state_w[:, 3:7]
    rel_pos_b = math_utils.quat_apply_inverse(robot_quat, rel_pos)
    rel_pos = torch.clamp(rel_pos_b, -18.0, 18.0)
    return rel_pos.view(env.num_envs, -1)

_buf = None
_step = None
_inited = None

def soccer_robot_relative_pos_b_strided_hist(env: ManagerBasedEnv) -> torch.Tensor:
    global _buf, _step, _inited

    stride = 5
    H = 5
    n_min, n_max = -0.03, 0.03

    fresh = soccer_robot_relative_pos_b(env)
    N, D = fresh.shape
    device = fresh.device

    if (
        _buf is None
        or _step is None
        or _inited is None
        or tuple(_buf.shape) != (N, H, D)
        or _buf.device != device
        or _buf.dtype != fresh.dtype
        or tuple(_step.shape) != (N,)
        or _step.device != device
        or tuple(_inited.shape) != (N,)
        or _inited.device != device
    ):
        _buf = fresh.unsqueeze(1).repeat(1, H, 1).clone()   # [N,H,3]
        _step = torch.zeros(N, device=device, dtype=torch.long)
        _inited = torch.ones(N, device=device, dtype=torch.bool)

    # reset
    if hasattr(env, "reset_buf"):
        ids = torch.nonzero(env.reset_buf, as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            _step[ids] = 0
            _buf[ids] = fresh[ids].unsqueeze(1).repeat(1, H, 1)
            _inited[ids] = True

    do_sample = (_step % stride) == 0
    do_sample = do_sample | (~_inited)

    if do_sample.any():
        ids = torch.nonzero(do_sample, as_tuple=False).squeeze(-1)
        v = fresh[ids]
        # Add noise only at sampled frames.
        v = v + (n_max - n_min) * torch.rand_like(v) + n_min

        _buf[ids, :-1] = _buf[ids, 1:].clone()
        _buf[ids, -1] = v
        _inited[ids] = True

    _step += 1
    return _buf.reshape(N, H * D)  # [N,15]
