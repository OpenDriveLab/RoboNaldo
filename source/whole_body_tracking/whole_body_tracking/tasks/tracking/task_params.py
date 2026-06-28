from __future__ import annotations

import yaml
from pathlib import Path

YAML_OVERRIDE_ENV = "WBT_TASK_PARAMS_YAML"

TRACKING_YAML_DIR = Path(__file__).parent / "yaml"
DEFAULT_TASK_PARAMS_PRESET = "right_kick/task_params_2.yaml"
DEFAULT_TASK_PARAMS_FILE = TRACKING_YAML_DIR / DEFAULT_TASK_PARAMS_PRESET

TRACKING_YAML_KEYS = {
    "inherits",
    "stage",
    "velocity_range",
    "motion_weight",
    "goal_weight",
    "reg_weight",
    "goal_reward_burst_steps",
    "goal_reset_delay_steps",
    "shot_success_threshold",
    "shot_valid_steps",
    "start_time_sampling_fraction",
    "init_pos_range",
    "init_vel_range",
    "init_pos_z_bound",
    "init_vel_z_range",
    "init_yaw_range",
    "static_ball_probability",
    "stand_still_env_ratio",
    "lidar_stale_probability",
    "ball_init_state_pos",
    "ball_init_state_vel",
    "std_difficulty_multiplier",
    "rigid_num",
    "use_rough_terrain",
    "mixed_terrain",
    "main_foot_name",
    "critic_frame_index",
    "critical_frame_adaptive_sampling",
    "critical_frame_sampling_window",
    "adapt_motion_flag",
    "jump_flag",
    "use_ontime_ball_reset",
    "anchor_xy_offset",
    "lambda_factor",
    "threshold_r",
    "threshold_t",
    "reward_overrides",
    "termination_overrides",
}

MIXED_TERRAIN_KEYS = {
    "terrain_bank_rows",
    "terrain_bank_cols",
    "horizontal_scale",
    "vertical_scale",
    "slope_threshold",
    "max_init_terrain_level",
    "random_rough",
    "slope",
    "inverted_slope",
}

MIXED_TERRAIN_SECTION_KEYS = {
    "random_rough": {"proportion", "noise_range", "noise_step", "downsampled_scale", "border_width"},
    "slope": {"proportion", "slope_range", "platform_width", "border_width"},
    "inverted_slope": {"proportion", "slope_range", "platform_width", "border_width"},
}

REQUIRED_TRACKING_YAML_KEYS = {
    "stage",
    "velocity_range",
    "motion_weight",
    "goal_weight",
    "reg_weight",
    "goal_reward_burst_steps",
    "goal_reset_delay_steps",
    "shot_success_threshold",
    "shot_valid_steps",
    "start_time_sampling_fraction",
    "init_pos_range",
    "init_vel_range",
    "init_pos_z_bound",
    "init_vel_z_range",
    "init_yaw_range",
    "static_ball_probability",
    "stand_still_env_ratio",
    "lidar_stale_probability",
    "ball_init_state_pos",
    "ball_init_state_vel",
    "std_difficulty_multiplier",
    "rigid_num",
    "use_rough_terrain",
    "main_foot_name",
    "critic_frame_index",
    "critical_frame_adaptive_sampling",
    "critical_frame_sampling_window",
    "adapt_motion_flag",
    "jump_flag",
    "use_ontime_ball_reset",
    "anchor_xy_offset",
    "lambda_factor",
    "threshold_r",
    "threshold_t",
}


def resolve_task_yaml_path(path_str: str | None) -> Path | None:
    """Resolve a task YAML path from CLI/env, supporting relative shorthand."""
    if path_str is None:
        return None
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = TRACKING_YAML_DIR / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Task params YAML not found: {candidate} (input: {path_str}). "
            f"Use path relative to {TRACKING_YAML_DIR} or an absolute path."
        )
    return candidate


def validate_tracking_yaml_keys(path: Path, params: dict) -> None:
    """Fail fast when a preset has stale, misspelled, or missing public keys."""
    unknown_keys = set(params) - TRACKING_YAML_KEYS
    if unknown_keys:
        raise KeyError(f"Unknown top-level key(s) in task params YAML '{path}': {sorted(unknown_keys)}.")
    missing_keys = REQUIRED_TRACKING_YAML_KEYS - set(params)
    if missing_keys:
        raise KeyError(f"Missing required top-level key(s) in task params YAML '{path}': {sorted(missing_keys)}.")
    termination_overrides = params.get("termination_overrides", {})
    if not isinstance(termination_overrides, dict):
        raise TypeError(f"termination_overrides in task params YAML '{path}' must be a mapping.")
    old_sections = {"default", "tracking", "task"} & set(termination_overrides)
    if old_sections:
        raise KeyError(
            f"Unsupported nested termination_overrides section(s) in task params YAML '{path}': "
            f"{sorted(old_sections)}. Put termination terms directly under termination_overrides."
        )
    validate_mixed_terrain_yaml(path, params.get("mixed_terrain"))


def validate_mixed_terrain_yaml(path: Path, mixed_terrain) -> None:
    """Validate optional mixed-terrain config without importing Isaac Lab."""
    if mixed_terrain is None:
        return
    if not isinstance(mixed_terrain, dict):
        raise TypeError(f"mixed_terrain in task params YAML '{path}' must be a mapping.")

    unknown_keys = set(mixed_terrain) - MIXED_TERRAIN_KEYS
    if unknown_keys:
        raise KeyError(f"Unknown mixed_terrain key(s) in task params YAML '{path}': {sorted(unknown_keys)}.")

    for section_name, allowed_keys in MIXED_TERRAIN_SECTION_KEYS.items():
        section = mixed_terrain.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise TypeError(f"mixed_terrain.{section_name} in task params YAML '{path}' must be a mapping.")
        unknown_section_keys = set(section) - allowed_keys
        if unknown_section_keys:
            raise KeyError(
                f"Unknown mixed_terrain.{section_name} key(s) in task params YAML '{path}': "
                f"{sorted(unknown_section_keys)}."
            )


def load_tracking_yaml_with_inherits(path: Path) -> dict:
    """Load a task preset, recursively applying the optional `inherits` key."""
    params = yaml.safe_load(path.read_text()) or {}
    inherits = params.pop("inherits", None)
    if inherits is not None:
        inherited_path = resolve_task_yaml_path(str(inherits))
        if inherited_path is None:
            raise FileNotFoundError(f"Task params YAML inherit target is empty for {path}.")
        inherited_params = load_tracking_yaml_with_inherits(inherited_path)
        inherited_params.update(params)
        params = inherited_params
    validate_tracking_yaml_keys(path, params)
    return params


def resolve_tracking_task_params_path(task_params_path: str | Path | None) -> Path:
    """Resolve a task-params path from either an absolute path or a path relative to yaml/."""
    if task_params_path is None:
        return DEFAULT_TASK_PARAMS_FILE

    candidate = Path(task_params_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    relative_candidate = (TRACKING_YAML_DIR / candidate).resolve()
    if relative_candidate.is_file():
        return relative_candidate

    raise FileNotFoundError(
        f"Could not find task params file '{task_params_path}'. Tried '{candidate}' and '{relative_candidate}'."
    )


def load_tracking_task_params(task_params_path: str | Path | None) -> tuple[Path, dict]:
    """Load a tracking preset YAML and normalize its value types."""
    params_path = resolve_tracking_task_params_path(task_params_path)
    params = load_tracking_yaml_with_inherits(params_path)
    params["velocity_range"] = {k: tuple(v) for k, v in params["velocity_range"].items()}
    params["critic_frame_index"] = int(params["critic_frame_index"])
    return params_path, params


def dump_effective_tracking_task_params(task_params_path: str | Path, output_path: str | Path) -> Path:
    """Write a fully-expanded task preset so resumed runs do not depend on inherited files."""
    params_path = resolve_tracking_task_params_path(task_params_path)
    params = load_tracking_yaml_with_inherits(params_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
    return destination


def resolve_tracking_training_stage(params_path: Path, params: dict) -> str:
    """Read the explicit training stage from a tracking YAML preset."""
    stage = str(params["stage"])
    if stage not in {"tracking", "task"}:
        raise ValueError(f"Unsupported training stage '{stage}' in task params YAML '{params_path}'.")
    return stage


def stage_key(training_stage: str) -> str:
    return "tracking" if training_stage == "tracking" else "task"


def select_stage_value(value, training_stage: str, context: str):
    if not isinstance(value, dict):
        return value
    selected_stage_key = stage_key(training_stage)
    valid_keys = {training_stage, selected_stage_key, "default"}
    if not any(key in value for key in valid_keys):
        raise KeyError(f"{context} has no value for stage '{training_stage}'.")
    if training_stage in value:
        return value[training_stage]
    if selected_stage_key in value:
        return value[selected_stage_key]
    return value["default"]


def select_weight_stage_value(value, training_stage: str, context: str):
    return select_stage_value(value, training_stage, context)


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
