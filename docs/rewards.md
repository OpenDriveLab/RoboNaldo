# Rewards

RoboNaldo combines imitation, soccer-task, and regularization rewards:

```text
r = motion_weight * r_motion + goal_weight * r_task + reg_weight * r_reg
```

The YAML presets control the three group weights and can override individual
reward terms.

## Motion Imitation

Motion rewards track the retargeted kick reference:

| Term | Code function | Purpose |
| --- | --- | --- |
| `motion_global_anchor_pos` | `motion_global_anchor_position_error_exp` | Keeps the anchor close to the reference. |
| `motion_global_anchor_ori` | `motion_global_anchor_orientation_error_exp` | Tracks anchor orientation. |
| `motion_body_pos` | `motion_relative_body_position_error_exp` | Tracks selected body positions relative to the anchor. |
| `motion_body_ori` | `motion_relative_body_orientation_error_exp` | Tracks selected body orientations. |
| `motion_body_lin_vel` | `motion_global_body_linear_velocity_error_exp` | Tracks body linear velocities. |
| `motion_body_ang_vel` | `motion_global_body_angular_velocity_error_exp` | Tracks body angular velocities. |
| `motion_feet_lin_vel` | `motion_global_feet_linear_velocity_error_exp` | Adds contact-sensitive foot velocity tracking. |

The global anchor weights are intentionally configured in each YAML preset under
`reward_overrides.motion.motion_global_anchor_pos` and
`reward_overrides.motion.motion_global_anchor_ori`. Tracking presets usually keep
these terms stronger, while task presets can lower them when the soccer objective
needs more deviation from the reference.

## Soccer Task

| Term | Code function | Purpose |
| --- | --- | --- |
| `robot_ball_contact_count` | `robot_ball_contact_count` | Rewards any detected foot-ball contact. |
| `robot_ball_contact` | `robot_ball_contact` | Instant interaction shaping for proximity, speed, force, and target quality. |
| `ball_velocity` | `ball_velocity` | Rewards high post-contact ball speed. |
| `ball_contact_orientation` | `ball_contact_orientation` | Rewards ball velocity toward the target after contact. |
| `error_ball_to_target` | `error_ball_to_target` | Densified shot-placement reward from the best ball-target distance. |
| `goal_reward_burst` | `goal_reward_burst` | Short reward burst after a new successful shot. |

The command term maintains rolling shot state: contact detection, shot-valid
windows, best distance to target, success metrics, goal bursts, and optional
in-episode reset.

## Stage 3 Locomotion and Stabilization

When `adapt_motion_flag` and `jump_flag` are enabled, the policy learns to approach
incoming balls before switching to the kick reference.

| Term | Purpose |
| --- | --- |
| `cmd_global_com_pos`, `cmd_delta_com_pos`, `cmd_global_anchor_ori` | Track the adapted anchor/command instead of only the original reference. |
| `cmd_velocity_tracking` | Encourages anchor velocity toward the adapted command. |
| `feet_air_time`, `feet_clearance` | Improve locomotion before the kick trigger. |
| `stable_anchor_pos_tracking`, `unstable_penalty` | Stabilize after the kick and reduce post-contact falls. |

## Regularization

Regularization terms penalize high action rate, rough contacts, foot slip,
joint-limit violations, excessive torque/velocity, and awkward arm poses. These
terms are intentionally conservative because they are used during sim-to-real
training.
