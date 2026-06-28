# Task Parameters

Task presets live under
`source/whole_body_tracking/whole_body_tracking/tasks/tracking/yaml/`.
They are the public configuration surface for the RoboNaldo curriculum.

## Loading Rules

- `--yaml right_kick/task_params_3.yaml` resolves relative to the tracking
  `yaml/` directory.
- Absolute YAML paths are also supported.
- `inherits` can be used by presets to inherit another preset and override a
  subset of fields.
- Every preset is schema-checked at load time. Unknown or missing top-level keys
  fail fast.
- Training archives the effective preset as `params/task_params.yaml`, and
  play/eval can reuse that archived file from a local checkpoint or WandB run.

## Curriculum Fields

| Field | Meaning |
| --- | --- |
| `stage` | Either `tracking` or `task`; selects stage-aware reward values and command behavior. |
| `motion_weight` | Global multiplier for imitation rewards. |
| `goal_weight` | Global multiplier for soccer task rewards. |
| `reg_weight` | Global multiplier for regularization terms. |
| `start_time_sampling_fraction` | Limits adaptive motion-clock sampling toward the beginning of the clip. |
| `critic_frame_index` | Contact/kick reference frame used by jump triggering and adaptive sampling. |

## Reset and Domain Randomization

| Field | Meaning |
| --- | --- |
| `init_pos_range` | Half-width of the ball XY spawn range around `ball_init_state_pos`. |
| `init_vel_range` | Half-width of sampled ball linear velocity. |
| `init_pos_z_bound` | Upper bound for ball spawn height offset. |
| `init_vel_z_range` | Vertical velocity range; normally zero for soccer. |
| `init_yaw_range` | Robot yaw perturbation at reset. |
| `static_ball_probability` | Probability of zeroing sampled ball velocity. |
| `velocity_range` | Random push velocity ranges for robot robustness. |
| `rigid_num` | PhysX GPU rigid patch budget exponent. |
| `use_rough_terrain` | Enables generated rough terrain instead of a plane. |
| `mixed_terrain` | Optional mixed-terrain generator settings used when `use_rough_terrain` is true. |

`right_kick/tracking_params.yaml` stays on a plane for the first tracking run.
`right_kick/tracking_mixed_params.yaml` inherits that preset and enables
`mixed_terrain` for robustness fine-tuning. The block controls the generated
terrain bank:

```yaml
use_rough_terrain: true
mixed_terrain:
  terrain_bank_rows: 4
  terrain_bank_cols: 8
  max_init_terrain_level: 0
  random_rough:
    proportion: 0.8
    noise_range: [0.0, 0.008]
  slope:
    proportion: 0.1
    slope_range: [0.0, 0.005]
  inverted_slope:
    proportion: 0.1
    slope_range: [0.0, 0.005]
```

Increase `noise_range`, `slope_range`, or `max_init_terrain_level` only after
the tracking prior learns reliably on the lighter preset.

## Soccer Task Fields

| Field | Meaning |
| --- | --- |
| `main_foot_name` | Contact foot used by sensors and task rewards. |
| `ball_init_state_pos` | Default ball position before randomization. |
| `ball_init_state_vel` | Default ball velocity before randomization. |
| `shot_success_threshold` | Target-distance threshold for successful shots. |
| `shot_valid_steps` | Steps after contact during which a shot can score. |
| `goal_reward_burst_steps` | Duration of the transient success reward. |
| `goal_reset_delay_steps` | Delay before in-episode ball reset after a success. |

## Stage 3 Command Fields

| Field | Meaning |
| --- | --- |
| `adapt_motion_flag` | Adapts the anchor command toward the ball or stabilization pose. |
| `jump_flag` | Jumps the motion clock near the kick frame when an incoming ball is predicted. |
| `critical_frame_adaptive_sampling` | Samples kick-entry frames around `critic_frame_index`. |
| `critical_frame_sampling_window` | Frame window used by critical-frame sampling. |
| `anchor_xy_offset` | Body-frame offset for the jump trigger's controllable-area test. |
| `threshold_r` | Incoming-ball trigger radius. |
| `threshold_t` | Incoming-ball trigger horizon in seconds. |
| `lambda_factor` | Scale for adapted anchor displacement toward the ball. |
| `use_ontime_ball_reset` | Allows in-episode ball reset after a successful shot. |

## Overrides

`reward_overrides` is grouped by category:

```yaml
reward_overrides:
  motion:
    motion_global_anchor_pos:
      weight_scale: 0.5
      params:
        error_threshold: 0.0
  goal:
    error_ball_to_target:
      weight_scale: 20.0
      params:
        std: 0.5
```

`weight_scale` is multiplied by the category group weight. `weight` sets an
absolute reward weight. `params` updates the target reward function parameters.
Reward-term-specific values, including imitation `error_threshold`, should stay
here instead of being added as top-level shortcut fields:

```yaml
reward_overrides:
  regularization:
    ee_body_pos_termination_penalty:
      params:
        warmup_threshold: 0.35
  goal:
    goal_reward_burst:
      weight: 500.0
```

Legacy flat aliases such as `*_weight`, `*_std`, and
`warmup_ee_body_pos_threshold` are intentionally unsupported. The schema fails
fast on those keys so presets do not grow multiple names for the same behavior.

`termination_overrides` is a single-level mapping from termination term name to
the override for the current preset:

```yaml
termination_overrides:
  anchor_pos:
    threshold: 0.25
  ee_body_pos:
    threshold:
      left_ankle_roll_link: 0.25
      right_ankle_roll_link: 0.25
      left_wrist_yaw_link: 0.25
      right_wrist_yaw_link: 0.25
  self_collision: false
```

Do not nest `default`, `tracking`, or `task` sections under
`termination_overrides`; each YAML preset should spell out the termination
settings it wants to use.
