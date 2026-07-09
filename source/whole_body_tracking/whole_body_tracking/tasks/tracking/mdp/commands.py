from __future__ import annotations

import math
import torch
import isaaclab.sim as sim_utils
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
from whole_body_tracking.tasks.tracking.mdp.motion_loader import (
    MotionLoader,
    quat_from_x_axis_to_vector as _quat_from_x_axis_to_vector,
    validate_int_at_least as _validate_int_at_least,
    validate_probability as _validate_probability,
)
from whole_body_tracking.tasks.tracking.mdp.trajectory import (
    ball_circle_min_distance,
    ball_enters_circle,
)
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cuboid_marker_cfg(
    prim_path: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )


def _arrow_x_marker_cfg(
    prim_path: str,
    color: tuple[float, float, float] = (1.0, 0.4, 0.8),
) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.num_dofs = 29
        self.step_dt = env.step_dt

        self.motion = MotionLoader(
            self.cfg.motion_file,
            self.body_indexes,
            device=self.device,
            critic_frame_index=self.cfg.critic_frame_index,
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.real_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.warmup_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        self.adapt_motion_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.adapt_motion_ori = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self.adapt_motion_ori[:, 0] = 1.0
        self.prev_adapt_motion_pos = torch.zeros_like(self.adapt_motion_pos)
        self.prev_adapt_motion_ori = torch.zeros_like(self.adapt_motion_ori)
        self.prev_adapt_motion_ori[:, 0] = 1.0
        self.desired_anchor_pos_b = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.desired_anchor_ori_b = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        self.desired_anchor_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.prev_desired_anchor_pos_w = torch.zeros_like(self.desired_anchor_pos_w)
        stand_still_ratio = _validate_probability("stand_still_env_ratio", self.cfg.stand_still_env_ratio)
        self._shot_valid_steps = _validate_int_at_least("shot_valid_steps", self.cfg.shot_valid_steps, 1)
        self._goal_reward_burst_steps = _validate_int_at_least(
            "goal_reward_burst_steps", self.cfg.goal_reward_burst_steps, 1
        )
        self._goal_reset_delay_steps = _validate_int_at_least(
            "goal_reset_delay_steps", self.cfg.goal_reset_delay_steps, 0
        )
        stand_still_count = int(round(self.num_envs * stand_still_ratio))
        self.stand_still_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if stand_still_count > 0:
            stand_still_ids = torch.randperm(self.num_envs, device=self.device)[:stand_still_count]
            self.stand_still_env_mask[stand_still_ids] = True
        self._stand_still_meshes_painted = False
        self.stand_still_start_com_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.stand_still_start_anchor_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.stand_still_start_anchor_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self.stand_still_start_anchor_quat_w[:, 0] = 1.0
        self.stand_still_start_body_pos_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, dtype=torch.float32, device=self.device
        )
        self.stand_still_start_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Feet-air-time reward temporal buffers keyed by body-name tuples.
        # This avoids hard-coding IsaacGym-style `foot_name` in Isaac Sim configs.
        self._feet_air_time_state: dict[tuple[str, ...], dict[str, torch.Tensor]] = {}
        self.stabilize_anchor_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.stabilize_anchor_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self.stabilize_anchor_quat_w[:, 0] = 1.0
        self.stabilize_target_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stabilize_ori_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stable_phase_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.adapt_body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.adapt_body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.adapt_body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )

        self.kernel = self.kernel / self.kernel.sum()

        self.soccer = env.scene[cfg.soccer_asset_name]
        self.ball_contact_forces = env.scene.sensors["ball_contact_forces"]
        self.prev_ball_velocity = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._foot_ball_history_len = 10
        self._foot_ball_distance_history = torch.full(
            (self.num_envs, self._foot_ball_history_len), float("inf"), dtype=torch.float32, device=self.device
        )
        self.origin = env.scene.env_origins
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.overline_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contacted_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Ring buffer of recent contact force norms (per env) to catch brief kick contacts
        self._contact_force_history_len = 6
        self._contact_force_history = torch.zeros(
            self.num_envs, self._contact_force_history_len, dtype=torch.float32, device=self.device
        )

        self.round_min_distance_to_target = 10.0 * torch.ones(self.num_envs, device=self.device)
        self.shot_window_min_distance_to_goal = 10.0 * torch.ones(self.num_envs, device=self.device)
        self.shot_target_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.contact_ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_init_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.prev_prev_action = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.estimated_ball_hit = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.cmd_target_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.last_contact_step = -torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.shot_valid_until_step = -torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.shot_active_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.shot_success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.shot_success_reward_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.shot_success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_max_shot_success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_shot_success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_had_shot = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_shot_error = 10.0 * torch.ones(self.num_envs, device=self.device)
        self.last_episode_had_shot = torch.zeros(self.num_envs, device=self.device)
        self.last_episode_shot_error = 10.0 * torch.ones(self.num_envs, device=self.device)
        self.goal_reward_steps_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_reward_active_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_reset_steps_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_reset_pending_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lidar_ball_pos_b_prev = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.lidar_ball_pos_b_cached = torch.zeros_like(self.lidar_ball_pos_b_prev)
        self.lidar_ball_pos_b_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lidar_ball_pos_b_cache_step = -1

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_ball_velocity"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_shot_success_streak"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["shot_success_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["last_episode_had_shot"] = self.last_episode_had_shot
        self.metrics["last_episode_shot_error"] = self.last_episode_shot_error
        self.metrics["stand_still_env"] = self.stand_still_env_mask.float()

    @property
    def command(self) -> torch.Tensor:
        """Expose joint position and velocity observations for command consumers."""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def is_warmup(self) -> torch.Tensor:
        return self.real_time_steps < self.warmup_time_steps + int(self.cfg.warmup_steps)

    @property
    def motion_time_steps(self) -> torch.Tensor:
        return torch.where(self.stand_still_env_mask, torch.zeros_like(self.time_steps), self.time_steps)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.motion_time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.motion_time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    @property
    def ball_pos(self) -> torch.Tensor:
        return self.soccer.data.root_state_w[:, :3]
    
    @property
    def ball_pos_r(self) -> torch.Tensor:
        return self.ball_pos - self.origin

    @property
    def contact_ball_pos_r(self) -> torch.Tensor:
        return self.contact_ball_pos - self.origin
    
    @property
    def ball_pos_b(self) -> torch.Tensor:
        robot_quat = self.robot_anchor_quat_w
        rel_pos = self.ball_pos - self.robot_anchor_pos_w
        rel_pos_b = math_utils.quat_apply_inverse(robot_quat, rel_pos)
        return rel_pos_b

    def apply_lidar_stale_ball_pos_b(self, fresh_ball_pos_b: torch.Tensor) -> torch.Tensor:
        """Return the lidar ball observation, occasionally reusing the last valid frame."""
        stale_probability = _validate_probability("lidar_stale_probability", self.cfg.lidar_stale_probability)
        if stale_probability == 0.0:
            return fresh_ball_pos_b
        fresh_ball_pos_b = fresh_ball_pos_b.view(self.num_envs, -1)
        if fresh_ball_pos_b.shape != self.lidar_ball_pos_b_prev.shape:
            self.lidar_ball_pos_b_prev = torch.zeros_like(fresh_ball_pos_b)
            self.lidar_ball_pos_b_cached = torch.zeros_like(fresh_ball_pos_b)
            self.lidar_ball_pos_b_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=fresh_ball_pos_b.device)
            self.lidar_ball_pos_b_cache_step = -1

        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=fresh_ball_pos_b.device)
        if hasattr(self._env, "reset_buf"):
            reset_mask = self._env.reset_buf.to(device=fresh_ball_pos_b.device, dtype=torch.bool)
        init_mask = reset_mask | (~self.lidar_ball_pos_b_valid)

        current_step = int(getattr(self._env, "common_step_counter", self.lidar_ball_pos_b_cache_step + 1))
        if current_step == self.lidar_ball_pos_b_cache_step:
            if init_mask.any():
                self.lidar_ball_pos_b_prev[init_mask] = fresh_ball_pos_b[init_mask]
                self.lidar_ball_pos_b_cached[init_mask] = fresh_ball_pos_b[init_mask]
                self.lidar_ball_pos_b_valid[init_mask] = True
            return self.lidar_ball_pos_b_cached

        stale_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=fresh_ball_pos_b.device)
        if stale_probability > 0.0:
            stale_mask = torch.rand(self.num_envs, device=fresh_ball_pos_b.device) < stale_probability
            stale_mask = stale_mask & self.lidar_ball_pos_b_valid & (~init_mask)

        output = fresh_ball_pos_b.clone()
        output[stale_mask] = self.lidar_ball_pos_b_prev[stale_mask]

        fresh_mask = ~stale_mask
        self.lidar_ball_pos_b_prev[fresh_mask] = fresh_ball_pos_b[fresh_mask]
        self.lidar_ball_pos_b_cached = output
        self.lidar_ball_pos_b_valid[:] = True
        self.lidar_ball_pos_b_cache_step = current_step
        return self.lidar_ball_pos_b_cached
    
    @property
    def target_pos_r(self) -> torch.Tensor:
        return self.target_pos - self.origin
    
    @property
    def target_pos_b(self) -> torch.Tensor:
        robot_quat = self.robot_anchor_quat_w
        rel_pos = self.target_pos - self.robot_anchor_pos_w
        rel_pos_b = math_utils.quat_apply_inverse(robot_quat, rel_pos)
        return rel_pos_b
    
    @property
    def ball_velocity(self) -> torch.Tensor:
        return self.soccer.data.root_state_w[:,7:10]
    
    @property
    def ball_to_target_dist(self) -> torch.Tensor:
        return torch.norm(self.target_pos_r - self.ball_pos_r, dim=-1)

    @property
    def ball_to_target_dist_current(self) -> torch.Tensor:
        return self.ball_to_target_dist

    @property
    def ball_init_pos_r(self) -> torch.Tensor:
        return self.ball_init_pos - self.origin

    def _update_target_distance_metrics(self) -> None:
        current_distance = self.ball_to_target_dist_current
        self.round_min_distance_to_target = torch.minimum(self.round_min_distance_to_target, current_distance)

        shot_active = self.shot_active_flag
        if shot_active.any():
            self.shot_window_min_distance_to_goal[shot_active] = torch.minimum(
                self.shot_window_min_distance_to_goal[shot_active], current_distance[shot_active]
            )
            self.episode_had_shot[shot_active] = True
            self.episode_shot_error[shot_active] = torch.minimum(
                self.episode_shot_error[shot_active], current_distance[shot_active]
            )

    def _update_shot_state(self) -> torch.Tensor:
        self._advance_goal_reward_state()

        shot_timed_out = self.shot_active_flag & (self.real_time_steps > self.shot_valid_until_step)
        if shot_timed_out.any():
            self.shot_active_flag[shot_timed_out] = False
            failed_shots = shot_timed_out & (~self.shot_success_flag)
            if failed_shots.any():
                self.shot_success_streak[failed_shots] = 0

        shot_min_distance = self.shot_window_min_distance_to_goal
        successful_shots = (
            self.shot_active_flag
            & (~self.shot_success_flag)
            & (shot_min_distance < float(self.cfg.shot_success_threshold))
        )
        if successful_shots.any():
            self.shot_success_flag[successful_shots] = True
            self.shot_success_reward_ready[successful_shots] = True
            self.shot_success_streak[successful_shots] += 1
            self.episode_max_shot_success_streak[successful_shots] = torch.maximum(
                self.episode_max_shot_success_streak[successful_shots],
                self.shot_success_streak[successful_shots],
            )
            self.episode_shot_success_count[successful_shots] += 1
            self.goal_reward_active_flag[successful_shots] = True
            self.goal_reward_steps_left[successful_shots] = self._goal_reward_burst_steps - 1
            self.goal_reset_pending_flag[successful_shots] = True
            self.goal_reset_steps_left[successful_shots] = self._goal_reset_delay_steps

        self.metrics["max_shot_success_streak"] = self.episode_max_shot_success_streak.float()
        self.metrics["shot_success_count"] = self.episode_shot_success_count.float()
        self.metrics["last_episode_had_shot"] = self.last_episode_had_shot
        self.metrics["last_episode_shot_error"] = self.last_episode_shot_error
        return successful_shots

    def _update_shot_projection(self) -> None:
        over_line_idx = (self.ball_pos[:, 1] > self.target_pos[:, 1]) & (self.ball_velocity[:, 1] > 0.05)
        update_idx = torch.where(~self.overline_flag & over_line_idx)[0]
        if len(update_idx) > 0:
            self.shot_target_pos[update_idx] = self.ball_pos[update_idx]
        self.overline_flag = over_line_idx

    def _reset_round_shooting_state(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        self.prev_ball_velocity = self.prev_ball_velocity.clone()
        self._contact_force_history = self._contact_force_history.clone()
        self._foot_ball_distance_history = self._foot_ball_distance_history.clone()
        self.round_min_distance_to_target = self.round_min_distance_to_target.clone()
        self.shot_window_min_distance_to_goal = self.shot_window_min_distance_to_goal.clone()
        self.shot_target_pos = self.shot_target_pos.clone()
        self.contact_ball_pos = self.contact_ball_pos.clone()
        self.last_contact_step = self.last_contact_step.clone()
        self.shot_valid_until_step = self.shot_valid_until_step.clone()
        self.shot_active_flag = self.shot_active_flag.clone()
        self.shot_success_flag = self.shot_success_flag.clone()
        self.shot_success_reward_ready = self.shot_success_reward_ready.clone()
        self.overline_flag[env_ids] = False
        self.contacted_flag[env_ids] = False
        self.prev_ball_velocity[env_ids] = 0.0
        self._contact_force_history[env_ids, :] = 0.0
        self._foot_ball_distance_history[env_ids, :] = float("inf")
        self.round_min_distance_to_target[env_ids] = 10.0
        self.shot_window_min_distance_to_goal[env_ids] = 10.0
        self.shot_target_pos[env_ids] = 0.0
        self.contact_ball_pos[env_ids] = 0.0
        self.last_contact_step[env_ids] = -1
        self.shot_valid_until_step[env_ids] = -1
        self.shot_active_flag[env_ids] = False
        self.shot_success_flag[env_ids] = False
        self.shot_success_reward_ready[env_ids] = False
        self.goal_reset_pending_flag[env_ids] = False
        self.goal_reset_steps_left[env_ids] = 0
        self.desired_anchor_pos_w[env_ids] = self.robot_anchor_pos_w[env_ids]
        self.prev_desired_anchor_pos_w[env_ids] = self.robot_anchor_pos_w[env_ids]
        for state in self._feet_air_time_state.values():
            state["air_time_buf"][env_ids] = 0.0
            state["last_contacts"][env_ids] = False
        self.stabilize_target_valid[env_ids] = False
        self.stabilize_ori_valid[env_ids] = False
        self._stable_phase_prev[env_ids] = False
        self.stabilize_anchor_pos_w[env_ids] = 0.0
        self.stabilize_anchor_quat_w[env_ids] = 0.0
        self.stabilize_anchor_quat_w[env_ids, 0] = 1.0

    def _reset_episode_shooting_state(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        self.last_episode_had_shot = self.last_episode_had_shot.clone()
        self.last_episode_shot_error = self.last_episode_shot_error.clone()
        self.metrics["last_episode_had_shot"] = self.metrics["last_episode_had_shot"].clone()
        self.metrics["last_episode_shot_error"] = self.metrics["last_episode_shot_error"].clone()

        had_shot = self.episode_had_shot[env_ids]
        self.last_episode_had_shot[env_ids] = had_shot.float()
        self.last_episode_shot_error[env_ids] = torch.where(
            had_shot,
            self.episode_shot_error[env_ids],
            torch.full_like(self.episode_shot_error[env_ids], 10.0),
        )
        self.metrics["last_episode_had_shot"][env_ids] = self.last_episode_had_shot[env_ids]
        self.metrics["last_episode_shot_error"][env_ids] = self.last_episode_shot_error[env_ids]

        self._reset_round_shooting_state(env_ids)
        self.shot_success_streak[env_ids] = 0
        self.episode_max_shot_success_streak[env_ids] = 0
        self.episode_shot_success_count[env_ids] = 0
        self.episode_had_shot[env_ids] = False
        self.episode_shot_error[env_ids] = 10.0
        self.goal_reward_steps_left[env_ids] = 0
        self.goal_reward_active_flag[env_ids] = False
        self.goal_reset_steps_left[env_ids] = 0
        self.goal_reset_pending_flag[env_ids] = False
        self.metrics["max_shot_success_streak"][env_ids] = 0.0
        self.metrics["shot_success_count"][env_ids] = 0.0

    def _advance_goal_reward_state(self) -> None:
        active_mask = self.goal_reward_steps_left > 0
        self.goal_reward_active_flag = active_mask
        if active_mask.any():
            self.goal_reward_steps_left[active_mask] -= 1

    def clear_goal_reward_burst(self, env_mask: torch.Tensor) -> None:
        """Disable any pending goal-reward burst for envs that have failed."""
        if not torch.any(env_mask):
            return
        self.goal_reward_steps_left[env_mask] = 0
        self.goal_reward_active_flag[env_mask] = False

    def _advance_goal_reset_state(self) -> torch.Tensor:
        pending_mask = self.goal_reset_pending_flag
        if pending_mask.any():
            self.goal_reset_steps_left[pending_mask] -= 1
        ready_to_reset = self.goal_reset_pending_flag & (self.goal_reset_steps_left <= 0)
        return ready_to_reset

    def _handle_goal_success(self, successful_shots: torch.Tensor) -> None:
        reset_mask = self._advance_goal_reset_state()
        if not reset_mask.any():
            return
        env_ids = torch.where(reset_mask)[0]
        if self.cfg.use_ontime_ball_reset:
            self._resample_command(env_ids, reset_episode_state=False)
            self.reset_ball(
                env_ids,
                position_range_x=self.cfg.ball_position_range_x,
                position_range_y=self.cfg.ball_position_range_y,
                position_range_z=self.cfg.ball_position_range_z,
                velocity_range=self.cfg.ball_velocity_range,
                in_episode_reset=True,
            )
        self._reset_round_shooting_state(env_ids)
    
    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        self.metrics["max_ball_velocity"] = torch.max(
            self.metrics["max_ball_velocity"], torch.norm(self.ball_velocity, dim=-1)
        )
        self._update_shot_projection()
        self._update_target_distance_metrics()
        successful_shot_mask = self._update_shot_state()
        self._handle_goal_success(successful_shot_mask)

    def _sample_critical_frame_steps(self, count: int, critic_frame: int, max_frame: int) -> torch.Tensor:
        """Sample the kick-entry frame near the critical frame when enabled."""
        if count <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        window = int(self.cfg.critical_frame_sampling_window)
        if not self.cfg.critical_frame_adaptive_sampling or window <= 0:
            return torch.full((count,), critic_frame, dtype=torch.long, device=self.device)

        end_frame = min(critic_frame + window, max_frame)
        candidate_frames = torch.arange(critic_frame, end_frame + 1, dtype=torch.long, device=self.device)
        if candidate_frames.numel() == 1:
            return candidate_frames.repeat(count)

        candidate_bins = torch.clamp(
            (candidate_frames * self.bin_count) // self.motion.time_step_total, 0, self.bin_count - 1
        )
        sampling_probabilities = self.bin_failed_count[candidate_bins].float()
        sampling_probabilities += self.cfg.adaptive_uniform_ratio / float(candidate_frames.numel())
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        sampled_offsets = torch.multinomial(sampling_probabilities, count, replacement=True)
        return candidate_frames[sampled_offsets]

    def _record_stabilization_target(self, env_mask: torch.Tensor) -> None:
        """Remember the robot support pose at kick entry for post-kick stabilization."""
        if not torch.any(env_mask):
            return
        self.stabilize_anchor_pos_w[env_mask] = self.robot_anchor_pos_w[env_mask]
        self.stabilize_anchor_quat_w[env_mask] = self.robot_anchor_quat_w[env_mask]
        self.stabilize_target_valid[env_mask] = True
        self.stabilize_ori_valid[env_mask] = True

    def _compute_rule_based_kick_trigger(self, critic_frame: int, require_before_critic: bool = True) -> torch.Tensor:
        # Use the same body-frame relative ball signal as the observation path so the
        # trigger logic is better aligned with the real-world setup, where absolute
        # world-frame ball position is not the primary input.
        ball_rel_pos_b = self.ball_pos_b
        ball_rel_vel_b = math_utils.quat_apply_inverse(
            self.robot_anchor_quat_w,
            self.ball_velocity - self.robot_anchor_lin_vel_w,
        )
        anchor_offset_xy_body = torch.tensor(
            self.cfg.anchor_xy_offset, device=self.device, dtype=self.robot_anchor_pos_w.dtype
        ).unsqueeze(0).expand(self.num_envs, -1)
        anchor_vel_xy_body = torch.zeros_like(anchor_offset_xy_body)
        ball_incoming = self._ball_enters_circle(
            ball_rel_pos_b[:, :2],
            ball_rel_vel_b[:, :2],
            anchor_offset_xy_body,
            anchor_vel_xy_body,
            radius=self.cfg.threshold_r,
            horizon=self.cfg.threshold_t,
        )
        trigger = (self.ball_pos[:, 2] < 0.75) & ball_incoming
        if require_before_critic:
            trigger = trigger & (self.time_steps < critic_frame)
        return trigger

    def _compute_rule_based_adapt_motion(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the rule-based adapted anchor command in world frame.

        This is the single source of truth for the hand-crafted command logic.
        """
        if not self.cfg.adapt_motion_flag:
            # Keep the target marker meaningful when adapted commands are disabled.
            self.cmd_target_pos = self.ball_pos.clone()
            anchor_pos = self.anchor_pos_w.clone()
            anchor_quat = self.anchor_quat_w.clone()
            if self.stand_still_env_mask.any():
                anchor_pos[self.stand_still_env_mask] = self.robot_anchor_pos_w[self.stand_still_env_mask]
                anchor_quat[self.stand_still_env_mask] = self.robot_anchor_quat_w[self.stand_still_env_mask]
                self.cmd_target_pos[self.stand_still_env_mask] = self.robot_anchor_pos_w[self.stand_still_env_mask]
            return anchor_pos, anchor_quat


        lambda_factor = self.cfg.lambda_factor
        cmd_target_pos = self.ball_pos.clone()

        stable_phase = self.time_steps > (self.motion.critic_frame_index + 50)
        entering_stable_phase = stable_phase & (~self._stable_phase_prev)
        if entering_stable_phase.any():
            # Latch the current anchor orientation exactly when stable phase starts.
            self.stabilize_anchor_quat_w[entering_stable_phase] = self.anchor_quat_w[entering_stable_phase]
            self.stabilize_ori_valid[entering_stable_phase] = True
        self._stable_phase_prev = stable_phase.clone()

        stabilize_phase_pos = stable_phase & self.stabilize_target_valid
        if stabilize_phase_pos.any():
            cmd_target_pos[stabilize_phase_pos] = self.stabilize_anchor_pos_w[stabilize_phase_pos]

        to_cmd_pos_w = (cmd_target_pos - self.robot_anchor_pos_w).clone()
        to_cmd_pos_w[:, 2] = self.anchor_pos_w[:, 2] - self.robot_anchor_pos_w[:, 2]

        to_cmd_dis_w = to_cmd_pos_w[:, :2].norm(dim=-1, keepdim=True)
        to_cmd_dir_w = to_cmd_pos_w / (to_cmd_pos_w.norm(dim=-1, keepdim=True) + 1e-6)
        command_amount = torch.clamp(to_cmd_dis_w, min=0.1, max=2) * 0.5

        estimated_ball_hit = self.anchor_pos_w.clone()
        anchor_pos_b_cmd = self.robot_anchor_pos_w[:, :2] + lambda_factor * (command_amount * to_cmd_dir_w[:, :2])
        estimated_ball_hit[:, :2] = anchor_pos_b_cmd
        estimated_ball_hit[:, 2] = self.anchor_pos_w[:, 2]

        to_ball_pos_w = self.ball_pos - self.robot_anchor_pos_w
        to_ball_pos_w[:, 2] = self.anchor_pos_w[:, 2] - self.robot_anchor_pos_w[:, 2]
        modified_ori = _quat_from_x_axis_to_vector(to_ball_pos_w)
        if stabilize_phase_pos.any():
            estimated_ball_hit[stabilize_phase_pos] = self.stabilize_anchor_pos_w[stabilize_phase_pos]
        stabilize_phase_ori = stable_phase & self.stabilize_ori_valid
        if stabilize_phase_ori.any():
            modified_ori[stabilize_phase_ori] = self.stabilize_anchor_quat_w[stabilize_phase_ori]

        if self.stand_still_env_mask.any():
            estimated_ball_hit[self.stand_still_env_mask] = self.robot_anchor_pos_w[self.stand_still_env_mask]
            modified_ori[self.stand_still_env_mask] = self.robot_anchor_quat_w[self.stand_still_env_mask]
            cmd_target_pos[self.stand_still_env_mask] = self.robot_anchor_pos_w[self.stand_still_env_mask]

        self.cmd_target_pos = cmd_target_pos.clone()

        return estimated_ball_hit, modified_ori

    def reset_ball(
        self,
        env_ids:  Sequence[int],
        position_range_x: tuple[float, float] = (0.1 , 0.2 ),
        position_range_y: tuple[float, float] = (0.1 , 0.2 ),
        position_range_z: tuple[float, float] = (0.0 , 0.0 ),
        velocity_range: tuple[float, float] = (-0.0 , 0.0 ),
        in_episode_reset: bool = False,
    ):
        """Reset the ball position and velocity."""

        asset = self.soccer
        if in_episode_reset:
            cur_anchor_pos_xy = self.robot_anchor_pos_w[env_ids, :2] - self.origin[env_ids, :2]
            cur_anchor_pos_xy = cur_anchor_pos_xy.to(asset.device)
        else:
            cur_anchor_pos_xy = torch.zeros((len(env_ids), 2), device=asset.device)
        # sample random position and velocity
        if len(env_ids) == 0:
            return
        
        pos_range_x = torch.tensor(position_range_x, device=asset.device)
        pos_range_y = torch.tensor(position_range_y, device=asset.device)
        pos_range_z = torch.tensor(position_range_z, device=asset.device)
        vel_range = torch.tensor(velocity_range, device=asset.device)
        
        positions_x = math_utils.sample_uniform(pos_range_x[0], pos_range_x[1], (len(env_ids), 1), device=asset.device)
        positions_y = math_utils.sample_uniform(pos_range_y[0], pos_range_y[1], (len(env_ids), 1), device=asset.device)
        positions_z = math_utils.sample_uniform(pos_range_z[0], pos_range_z[1], (len(env_ids), 1), device=asset.device)
        positions = torch.cat([positions_x, positions_y, positions_z], dim=-1)
        velocities = math_utils.sample_uniform(vel_range[0], vel_range[1], (len(env_ids), 3), device=asset.device)
        if self.cfg.static_ball_probability > 0.0:
            static_mask = torch.rand(len(env_ids), device=asset.device) < self.cfg.static_ball_probability
            velocities[static_mask] = 0.0
        
        # regularization
        velocities[:,2] = 0.0  # reduce vertical velocity
        positions [:,2] /= 2.0 # reduce z range
        
        default_pos = self.origin[env_ids, :3]
        root_state = asset.data.default_root_state.clone()
        
        # set pos
        root_state[env_ids, :3] += default_pos
        root_state[env_ids, :3] += positions
        root_state[env_ids, :2] += cur_anchor_pos_xy
        root_state[env_ids, 7:10] += velocities
        root_state[env_ids, 8] = - torch.abs(root_state[env_ids, 8])  # ensure negative y velocity
        # Compensate for incoming ball motion before the kick window.
        come_t = torch.rand(len(env_ids), device=self.device)
        come_t = torch.clamp(come_t, min=0.75, max=2)+0.2
        root_state[env_ids, :2] -= come_t.unsqueeze(1) * root_state[env_ids, 7:9]
        self.ball_init_pos[env_ids] = root_state[env_ids, :3].clone()
        asset.write_root_pose_to_sim(root_state[env_ids, :7], env_ids=env_ids )
        asset.write_root_velocity_to_sim(root_state[env_ids, 7:], env_ids=env_ids )
        
        # reset buffers
        asset.reset()

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // self.motion.time_step_total, 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        sampled_time_steps = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        start_fraction = _validate_probability("start_time_sampling_fraction", self.cfg.start_time_sampling_fraction)
        if start_fraction < 1.0:
            sampled_time_steps = (
                start_fraction * sampled_bins.float() / self.bin_count * (self.motion.time_step_total - 1)
            ).long()
        self.time_steps[env_ids] = sampled_time_steps
        self.real_time_steps[env_ids] = self.time_steps[env_ids]
        self.warmup_time_steps[env_ids] = sampled_time_steps
        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int], *, reset_episode_state: bool = True):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        stand_still_env_ids = env_ids_tensor[self.stand_still_env_mask[env_ids_tensor]]
        if len(stand_still_env_ids) > 0:
            self.time_steps[stand_still_env_ids] = 0
            self.real_time_steps[stand_still_env_ids] = 0
            self.warmup_time_steps[stand_still_env_ids] = 0
        

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range[key] for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range[key] for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        if len(stand_still_env_ids) > 0:
            self.stand_still_start_com_pos_w[stand_still_env_ids] = root_pos[stand_still_env_ids]
            self.stand_still_start_anchor_pos_w[stand_still_env_ids] = self.robot_anchor_pos_w[stand_still_env_ids]
            self.stand_still_start_anchor_quat_w[stand_still_env_ids] = self.robot_anchor_quat_w[stand_still_env_ids]
            self.stand_still_start_body_pos_w[stand_still_env_ids] = self.body_pos_w[stand_still_env_ids]
            self.stand_still_start_valid[stand_still_env_ids] = True

        self.reset_ball(env_ids, position_range_x=self.cfg.ball_position_range_x, position_range_y=self.cfg.ball_position_range_y, position_range_z = self.cfg.ball_position_range_z, velocity_range=self.cfg.ball_velocity_range)
        self.lidar_ball_pos_b_valid[env_ids] = False
        self.lidar_ball_pos_b_cache_step = -1
        self.overline_flag[env_ids] = False
        # target
        range_x = self.cfg.target_pos_range['x']
        range_y = self.cfg.target_pos_range['y']
        range_z = self.cfg.target_pos_range['z']
        resample_pos = torch.rand(len(env_ids), 3, device=self.device)
        resample_pos [:,0] = torch.rand(len(env_ids), device=self.device)*(range_x[1]-range_x[0]) + range_x[0] # x
        resample_pos [:,1] = torch.rand(len(env_ids), device=self.device)*(range_y[1]-range_y[0]) + range_y[0] # y 
        resample_pos [:,2] = torch.rand(len(env_ids), device=self.device)*(range_z[1]-range_z[0]) + range_z[0] # z
        self.target_pos[env_ids] = resample_pos + self._env.scene.env_origins[env_ids]
        self.contacted_flag[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        self._contact_force_history[env_ids, :] = 0.0
        self.metrics["max_ball_velocity"][env_ids] = torch.zeros(len(env_ids), device=self.device)
        if reset_episode_state:
            self._reset_episode_shooting_state(env_ids)
        else:
            self._reset_round_shooting_state(env_ids)
        self.prev_prev_action[env_ids] = torch.zeros((len(env_ids), self.num_dofs), dtype=torch.float, device=self.device)

    @staticmethod
    def _ball_enters_circle(
        ball_pos_xy: torch.Tensor,   # (E, 2)
        ball_vel_xy: torch.Tensor,   # (E, 2)  velocity relative to anchor
        anchor_xy: torch.Tensor,     # (E, 2)
        anchor_vel_xy: torch.Tensor, # (E, 2)
        radius: float = 0.5,
        horizon: float = 0.5,
    ) -> torch.Tensor:               # (E,) bool
        """True if the ball's linear trajectory passes near the anchor."""
        return ball_enters_circle(
            ball_pos_xy,
            ball_vel_xy,
            anchor_xy,
            anchor_vel_xy,
            radius=radius,
            horizon=horizon,
        )

    @staticmethod
    def _ball_circle_min_distance(
        ball_pos_xy: torch.Tensor,   # (E, 2)
        ball_vel_xy: torch.Tensor,   # (E, 2)  velocity relative to anchor
        anchor_xy: torch.Tensor,     # (E, 2)
        anchor_vel_xy: torch.Tensor, # (E, 2)
        horizon: float = 0.5,
    ) -> torch.Tensor:               # (E,)
        """Minimum predicted ball-anchor distance over the next `horizon` seconds."""
        return ball_circle_min_distance(
            ball_pos_xy,
            ball_vel_xy,
            anchor_xy,
            anchor_vel_xy,
            horizon=horizon,
        )

    def _update_command(self):
        max_frame = self.motion.time_step_total - 1
        cf = self.motion.critic_frame_index
        warmup_phase = self.is_warmup
        if self.cfg.jump_flag:
            # Ball velocity relative to anchor (XY only)
            rel_vel_xy  = (self.ball_velocity - self.robot_anchor_lin_vel_w)[:, :2]  # (E, 2)
            ball_pos_xy = self.ball_pos[:, :2]                                        # (E, 2)
            anchor_xy_world = self.robot_anchor_pos_w[:, :2]                          # (E, 2)
            anchor_vel_xy = self.robot_anchor_lin_vel_w[:, :2]                          # (E, 2)
            # `anchor_xy_offset` is defined in the robot body frame and therefore must
            # be rotated into world coordinates before it shifts the anchor center used
            # by the jump trigger's controllable-area test.
            anchor_offset_xy_body = torch.tensor(
                self.cfg.anchor_xy_offset, device=self.device, dtype=self.robot_anchor_pos_w.dtype
            ).unsqueeze(0).expand(self.num_envs, -1)
            anchor_offset_xyz_body = torch.zeros(
                (self.num_envs, 3), device=self.device, dtype=self.robot_anchor_pos_w.dtype
            )
            anchor_offset_xyz_body[:, :2] = anchor_offset_xy_body
            anchor_offset_xy_world = quat_apply(self.robot_anchor_quat_w, anchor_offset_xyz_body)[:, :2]
            anchor_xy = anchor_xy_world + anchor_offset_xy_world
            # Boolean mask [E]: ball will enter 0.2 m circle around anchor within 0.4 s
            ball_incoming: torch.Tensor = self._ball_enters_circle(
                ball_pos_xy, rel_vel_xy, anchor_xy, anchor_vel_xy, radius=self.cfg.threshold_r, horizon=self.cfg.threshold_t
            )

            kick_idx = torch.where((self.time_steps >= cf - 1) & (~warmup_phase) & (~self.stand_still_env_mask))[0]

            # Jump to kick frame when: not yet at kick frame AND ball is incoming AND ball is low
            idx = torch.where(
                (self.time_steps < cf)
                & (self.ball_pos[:, 2] < 0.75)
                & ball_incoming
                & (~warmup_phase)
                & (~self.stand_still_env_mask)
            )[0]

            if idx.numel() > 0:
                trigger_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                trigger_mask[idx] = True
                self._record_stabilization_target(trigger_mask)
                self.time_steps[idx] = self._sample_critical_frame_steps(idx.numel(), cf, max_frame)
            self.time_steps[kick_idx] += 1
        else:
            self.time_steps[~self.stand_still_env_mask] += 1

        if warmup_phase.any():
            self.time_steps[warmup_phase] = self.warmup_time_steps[warmup_phase]
        if self.stand_still_env_mask.any():
            self.time_steps[self.stand_still_env_mask] = 0

        # motion.*[self.time_steps] requires 0 <= time_steps < time_step_total; kick_idx += 1 can overshoot T-1
        self.time_steps.clamp_(0, max_frame)
        self.real_time_steps += 1
        env_ids = torch.where(self.real_time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)
        self.time_steps.clamp_(0, max_frame)

        self.prev_adapt_motion_pos = self.adapt_motion_pos.clone()
        self.prev_adapt_motion_ori = self.adapt_motion_ori.clone()
        self.adapt_motion_pos, self.adapt_motion_ori = self._compute_rule_based_adapt_motion()
        self.estimated_ball_hit = self.adapt_motion_pos.clone()
        if self.stand_still_env_mask.any():
            self.desired_anchor_pos_b[self.stand_still_env_mask] = 0.0
            self.desired_anchor_ori_b[self.stand_still_env_mask] = 0.0

        adapt_anchor_pos_w_repeat = self.adapt_motion_pos[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        adapt_anchor_quat_w_repeat = self.adapt_motion_ori[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        adapt_delta_pos_w = robot_anchor_pos_w_repeat
        adapt_delta_pos_w[..., 2] = adapt_anchor_pos_w_repeat[..., 2]
        adapt_delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(adapt_anchor_quat_w_repeat)))
        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.adapt_body_quat_relative_w = quat_mul(adapt_delta_ori_w, self.body_quat_w)
        self.adapt_body_pos_relative_w = adapt_delta_pos_w + quat_apply(adapt_delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

        action_manager = self._env.action_manager
        self.prev_prev_action = action_manager.prev_action.clone()
        self.process_contact()
        
    def process_contact(self):
        """Update contact state using a short temporal window so brief kick contacts are detected.

        A contact is registered when a large ball-velocity change occurs AND the foot-ball
        distance was below the threshold at *any* step within the recent history window.
        """
        current_ball_velocity = self.ball_velocity.clone()
        ball_velocity_change = (current_ball_velocity - self.prev_ball_velocity).norm(dim=-1)
        ball_velocity_change_threshold = 0.3
        contact_distance_threshold = 0.2

        ball_pos = self.ball_pos
        main_foot_idx = self.cfg.body_names.index(self.cfg.main_foot_name)
        main_foot_pos = self.robot_body_pos_w[:, main_foot_idx, :]
        distance = torch.norm(main_foot_pos - ball_pos, dim=-1)

        # Shift window left by 1 and write current distance into the last slot
        self._foot_ball_distance_history = torch.roll(self._foot_ball_distance_history, -1, dims=1)
        self._foot_ball_distance_history[:, -1] = distance

        # Foot was close to ball in any step within the window
        min_dist_in_window = self._foot_ball_distance_history.min(dim=-1).values
        close_in_window = min_dist_in_window < contact_distance_threshold

        # Contact: velocity changed sharply AND foot was close (within the window)
        contact_mask = (ball_velocity_change > ball_velocity_change_threshold) & close_in_window
        self.contacted_flag[contact_mask] = True
        if contact_mask.any():
            self.last_contact_step[contact_mask] = self.real_time_steps[contact_mask]
            self.shot_valid_until_step[contact_mask] = self.real_time_steps[contact_mask] + self._shot_valid_steps
            self.shot_active_flag[contact_mask] = True
            self.shot_success_flag[contact_mask] = False
            self.shot_success_reward_ready[contact_mask] = False
            self.shot_window_min_distance_to_goal[contact_mask] = self.ball_to_target_dist_current[contact_mask]
            self.episode_had_shot[contact_mask] = True
            self.episode_shot_error[contact_mask] = torch.minimum(
                self.episode_shot_error[contact_mask], self.ball_to_target_dist_current[contact_mask]
            )
            self.contact_ball_pos[contact_mask] = ball_pos[contact_mask]
        # Maintain prev_ball_velocity for next call
        self.prev_ball_velocity = current_ball_velocity.clone()
        

    def _paint_stand_still_robot_meshes(self) -> None:
        """Bind a blue visual material to stand-still robots."""
        stand_still_env_mask = getattr(self, "stand_still_env_mask", None)
        if (
            stand_still_env_mask is None
            or getattr(self, "_stand_still_meshes_painted", False)
            or not torch.any(stand_still_env_mask)
        ):
            return
        import omni.usd
        from pxr import Sdf, UsdShade

        stage = omni.usd.get_context().get_stage()
        material_path = Sdf.Path("/World/Materials/StandStillRobotBlue")
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.15, 1.0))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        stand_still_env_ids = torch.where(stand_still_env_mask)[0].detach().cpu().tolist()
        for env_id in stand_still_env_ids:
            robot_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot")
            if not robot_prim.IsValid():
                continue
            UsdShade.MaterialBindingAPI.Apply(robot_prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
        self._stand_still_meshes_painted = True


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            self._paint_stand_still_robot_meshes()
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )
                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                self.estimate_visualizers = []
                for name in self.cfg.body_names:
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)
        
        # Soccer Ball vis
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                
                self.goal_visualizer = VisualizationMarkers(
                    self.cfg.goal_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/target")
                )
                self.shot_visualizer = VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/shot")
                )
                self.estimate_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/estimate/anchor")

                )

                self.goal_visualizers = []
                self.goal_visualizers.append(
                    VisualizationMarkers(
                        self.cfg.goal_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/target")
                    )
                )
                self.goal_visualizers.append(
                    VisualizationMarkers(
                        self.cfg.shot_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/shot")
                    )
                )
                self.estimate_visualizers.append(
                    VisualizationMarkers(
                        self.cfg.estimate_visualizer_cfg.replace(prim_path="/Visuals/Command/estimate/ball_hit")
                    )
                )
                self.ball_hit_arrow_visualizer = VisualizationMarkers(
                    self.cfg.ball_hit_arrow_cfg.replace(prim_path="/Visuals/Command/estimate/ball_hit_arrow")
                )

            self.goal_visualizer.set_visibility(True)
            self.shot_visualizer.set_visibility(True)
            self.estimate_anchor_visualizer.set_visibility(True)
            self.goal_visualizers[0].set_visibility(True)
            self.goal_visualizers[1].set_visibility(True)
            self.estimate_visualizers[0].set_visibility(True)
            self.ball_hit_arrow_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
                self.shot_visualizer.set_visibility(False)
                self.estimate_anchor_visualizer.set_visibility(False)
                self.goal_visualizers[0].set_visibility(False)
                self.goal_visualizers[1].set_visibility(False)
                self.estimate_visualizers[0].set_visibility(False)
                if hasattr(self, "ball_hit_arrow_visualizer"):
                    self.ball_hit_arrow_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_visualizer.visualize(self.target_pos)
        self.shot_visualizer.visualize(self.shot_target_pos)
        self.goal_visualizers[1].visualize(self.shot_target_pos)
        self.estimate_visualizers[0].visualize(self.estimated_ball_hit)

        _dir = self.cmd_target_pos - self.robot_anchor_pos_w              # (E, 3)
        _len = _dir.norm(dim=-1, keepdim=True)
        _q = _quat_from_x_axis_to_vector(_dir)
        # scale X to segment length (clamped so the arrow stays readable); keep Y/Z thin
        _arrow_len = _len.clamp(min=0.1, max=0.5)
        _scales = torch.cat([_arrow_len, torch.full_like(_len, 0.06), torch.full_like(_len, 0.06)], dim=-1)
        self.ball_hit_arrow_visualizer.visualize(self.robot_anchor_pos_w, _q, scales=_scales)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])

        for i in range(len(self.cfg.body_names)):
            self.goal_body_visualizers[i].visualize(
                self.adapt_body_pos_relative_w[:, i], self.adapt_body_quat_relative_w[:, i]
            )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion self."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    # task related
    soccer_asset_name: str = "soccer"    
    ball_position_range_x: tuple[float, float] = (0.2-0.5 , 0.2+0.5 )
    ball_position_range_y: tuple[float, float] = (1.0-0.25 , 1.0+0.75 )
    ball_position_range_z: tuple[float, float] = (0,0)
    ball_velocity_range: tuple[float, float] = (-0.0 , 0.0 )
    static_ball_probability: float = 0.1
    lidar_stale_probability: float = 0.0

    target_pos_range: dict[str, tuple[float, float]] = {'x': (-3.0, 3.0),'y':(5.0,5.0), 'z': (0.0, 2.0)}
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    shot_visualizer_cfg: VisualizationMarkersCfg = _cuboid_marker_cfg(
        prim_path="/Visuals/Command/shot",
        size=(0.15, 0.15, 0.15),
        color=(1.0, 0.0, 0.0, 1.0),
    )
    goal_visualizer_cfg: VisualizationMarkersCfg = _cuboid_marker_cfg(
        prim_path="/Visuals/Command/target",
        size=(0.5, 0.5, 0.5),
        color=(0.68, 0.85, 0.9, 0.8),
    )
    estimate_visualizer_cfg: VisualizationMarkersCfg = _cuboid_marker_cfg(
        prim_path="/Visuals/Command/estimate/anchor",
        size=(0.6, 0.6, 0.6),
        color=(0.2, 0.5, 1.0, 0.5),
    )
    ball_hit_arrow_cfg: VisualizationMarkersCfg = _arrow_x_marker_cfg(
        prim_path="/Visuals/Command/estimate/ball_hit_arrow",
    )
    contacted_flag: torch.Tensor = None
    overline_flag: torch.Tensor = None
    main_foot_name: str = "right_ankle_roll_link"
    critic_frame_index: int = MISSING
    stage_name: str = "tracking"
    start_time_sampling_fraction: float = 0.1
    warmup_steps: int = 20

    jump_flag: bool = False
    anchor_xy_offset: tuple[float, float] = (0.0, 0.0)
    adapt_motion_flag: bool = False
    critical_frame_adaptive_sampling: bool = False
    critical_frame_sampling_window: int = 0
    kick_hold_steps: int = 50
    use_ontime_ball_reset: bool = False
    stand_still_env_ratio: float = 0.0
    threshold_r: float = 0.2
    threshold_t: float = 0.4
    lambda_factor: float = 0.5
    shot_valid_steps: int = 100
    shot_success_threshold: float = 0.5
    goal_reward_burst_steps: int = 4
    goal_reset_delay_steps: int = 50
