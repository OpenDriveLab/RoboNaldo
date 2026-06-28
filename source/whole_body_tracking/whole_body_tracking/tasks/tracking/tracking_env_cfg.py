from __future__ import annotations

from copy import deepcopy
import os
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.task_overrides import (
    apply_reward_overrides as _apply_reward_overrides,
    apply_termination_overrides as _apply_termination_overrides,
    make_self_collision_termination as _make_self_collision_termination,
)
from whole_body_tracking.tasks.tracking.task_params import (
    DEFAULT_TASK_PARAMS_FILE as _DEFAULT_TASK_PARAMS_FILE,
    DEFAULT_TASK_PARAMS_PRESET as _DEFAULT_TASK_PARAMS_PRESET,
    YAML_OVERRIDE_ENV as _YAML_OVERRIDE_ENV,
    load_tracking_task_params,
    load_tracking_yaml_with_inherits as _load_tracking_yaml_with_inherits,
    resolve_task_yaml_path as _resolve_task_yaml_path,
    resolve_tracking_training_stage,
)

##
# Scene definition
##

# ---------------------------------------------------------------------------
# Default tuneable hyperparameters. These are only bootstrap defaults used to
# construct the config objects. The effective preset is loaded at runtime via
# `task_params_path`, so different runs can select different YAML files without
# editing this module.
# ---------------------------------------------------------------------------
if not _load_tracking_yaml_with_inherits(_DEFAULT_TASK_PARAMS_FILE):
    raise RuntimeError(f"Default tracking preset '{_DEFAULT_TASK_PARAMS_FILE}' is empty.")
_PARAMS_FILE = _resolve_task_yaml_path(os.environ.get(_YAML_OVERRIDE_ENV)) or _DEFAULT_TASK_PARAMS_FILE
_P = _load_tracking_yaml_with_inherits(_PARAMS_FILE)


VELOCITY_RANGE: dict = {k: tuple(v) for k, v in _P["velocity_range"].items()}
MAIN_FOOT_NAME: str = _P["main_foot_name"]
# simulation settings
rigid_num: int = int(_P["rigid_num"])
use_rough_terrain: bool = bool(_P["use_rough_terrain"])

# reward settings
motion_weight: float = float(_P["motion_weight"])
goal_weight: float = float(_P["goal_weight"])
reg_weight: float = float(_P["reg_weight"])
goal_reward_burst_steps: int = int(_P["goal_reward_burst_steps"])
goal_reset_delay_steps: int = int(_P["goal_reset_delay_steps"])

# episode initialisation ranges
init_pos_range: float = float(_P["init_pos_range"])
init_vel_range: float = float(_P["init_vel_range"])
init_pos_z_bound: float = float(_P["init_pos_z_bound"])
init_vel_z_range: float = float(_P["init_vel_z_range"])
init_yaw_range: float = float(_P["init_yaw_range"])
static_ball_probability: float = float(_P["static_ball_probability"])
ball_init_state_pos: list[float] = _P["ball_init_state_pos"]
ball_init_state_vel: list[float] = _P["ball_init_state_vel"]

std_difficulty_multiplier: float = float(_P["std_difficulty_multiplier"])
critic_frame_index: int = int(_P["critic_frame_index"])
DEFAULT_REWARD_ERROR_THRESHOLD: float = 0.0

# Adapted motion and jump trigger settings.
adapt_motion_flag: bool = bool(_P["adapt_motion_flag"])
jump_flag: bool = bool(_P["jump_flag"])
# Body-frame XY offset applied to the robot anchor before jump-trigger controllable-area checks.
anchor_xy_offset: tuple[float, float] = tuple(_P["anchor_xy_offset"])
threshold_r: float = float(_P["threshold_r"])
threshold_t: float = float(_P["threshold_t"])
lambda_factor: float = float(_P["lambda_factor"])
critical_frame_adaptive_sampling: bool = bool(_P["critical_frame_adaptive_sampling"])
critical_frame_sampling_window: int = int(_P["critical_frame_sampling_window"])
shot_valid_steps: int = int(_P["shot_valid_steps"])
shot_success_threshold: float = float(_P["shot_success_threshold"])
use_ontime_ball_reset: bool = bool(_P["use_ontime_ball_reset"])
stand_still_env_ratio: float = float(_P["stand_still_env_ratio"])
stand_still_reward_weight: float = 1.5 / stand_still_env_ratio if stand_still_env_ratio > 0.0 else 0.0


DEFAULT_MIXED_TERRAIN_CFG = {
    "terrain_bank_cols": 10,
    "terrain_bank_rows": 10,
    "horizontal_scale": 0.1,
    "vertical_scale": 0.005,
    "slope_threshold": 0.75,
    "max_init_terrain_level": 1,
    "random_rough": {
        "proportion": 0.8,
        "noise_range": (0.0, 0.035),
        "noise_step": 0.005,
        "downsampled_scale": 0.2,
        "border_width": 0.25,
    },
    "slope": {
        "proportion": 0.1,
        "slope_range": (0.0, 0.02),
        "platform_width": 0.25,
        "border_width": 0.25,
    },
    "inverted_slope": {
        "proportion": 0.1,
        "slope_range": (0.0, 0.02),
        "platform_width": 0.25,
        "border_width": 0.25,
    },
}


def resolve_mixed_terrain_cfg(raw_cfg: dict | None = None) -> dict:
    """Merge YAML mixed-terrain settings with conservative defaults."""
    cfg = deepcopy(DEFAULT_MIXED_TERRAIN_CFG)
    if raw_cfg:
        for key, value in raw_cfg.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value

    for section_name, range_key in (
        ("random_rough", "noise_range"),
        ("slope", "slope_range"),
        ("inverted_slope", "slope_range"),
    ):
        section = cfg[section_name]
        section[range_key] = tuple(section[range_key])
    return cfg


def make_mixed_terrain_importer_cfg(
    num_envs: int,
    env_spacing: float,
    mixed_terrain_cfg: dict | None = None,
) -> TerrainImporterCfg:
    """Build rough generated terrain."""
    terrain_cfg = resolve_mixed_terrain_cfg(mixed_terrain_cfg)
    # We do not need one unique terrain tile per environment. TerrainImporter can
    # reuse a smaller bank of terrain origins across all envs, which dramatically
    # reduces startup time for dense continuous meshes.
    terrain_bank_cols = int(terrain_cfg["terrain_bank_cols"])
    terrain_bank_rows = int(terrain_cfg["terrain_bank_rows"])
    num_cols = min(terrain_bank_cols, max(1, num_envs))
    num_rows = min(terrain_bank_rows, max(1, num_envs))
    random_rough_cfg = terrain_cfg["random_rough"]
    slope_cfg = terrain_cfg["slope"]
    inverted_slope_cfg = terrain_cfg["inverted_slope"]

    my_terrain_gen = terrain_gen.TerrainGeneratorCfg(
        size=(float(0.5 * env_spacing), float(0.5 * env_spacing)),
        border_width=2.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=float(terrain_cfg["horizontal_scale"]),
        vertical_scale=float(terrain_cfg["vertical_scale"]),
        slope_threshold=float(terrain_cfg["slope_threshold"]),
        seed=0,
        use_cache=True,
        sub_terrains={
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=float(random_rough_cfg["proportion"]),
                noise_range=tuple(random_rough_cfg["noise_range"]),
                noise_step=float(random_rough_cfg["noise_step"]),
                downsampled_scale=float(random_rough_cfg["downsampled_scale"]),
                border_width=float(random_rough_cfg["border_width"]),
            ),
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=float(slope_cfg["proportion"]),
                slope_range=tuple(slope_cfg["slope_range"]),
                platform_width=float(slope_cfg["platform_width"]),
                border_width=float(slope_cfg["border_width"]),
            ),
            "inverted_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=float(inverted_slope_cfg["proportion"]),
                slope_range=tuple(inverted_slope_cfg["slope_range"]),
                platform_width=float(inverted_slope_cfg["platform_width"]),
                border_width=float(inverted_slope_cfg["border_width"]),
            ),
        },
    )

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=my_terrain_gen,
        max_init_terrain_level=min(int(terrain_cfg["max_init_terrain_level"]), max(num_rows - 1, 0)),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="max",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.95,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        use_terrain_origins=True,
        env_spacing=env_spacing,
        debug_vis=False,
    )


def make_plane_terrain_importer_cfg() -> TerrainImporterCfg:
    """Build flat plane terrain importer config."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="max",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.95,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )


def configure_scene_terrain(env_cfg) -> None:
    """Apply the requested terrain config to the scene and simulator."""
    if env_cfg.use_rough_terrain:
        env_cfg.scene.terrain = make_mixed_terrain_importer_cfg(
            env_cfg.scene.num_envs,
            env_cfg.scene.env_spacing,
            getattr(env_cfg, "mixed_terrain", None),
        )
    else:
        env_cfg.scene.terrain = make_plane_terrain_importer_cfg()
    env_cfg.sim.physics_material = env_cfg.scene.terrain.physics_material

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = make_plane_terrain_importer_cfg()
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=False
    )

    soccer = RigidObjectCfg(
        init_state = RigidObjectCfg.InitialStateCfg(pos=ball_init_state_pos, lin_vel=ball_init_state_vel),
        prim_path = "{ENV_REGEX_NS}/soccer",
        spawn = sim_utils.SphereCfg(
            radius=0.115,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                linear_damping=0.03,
                angular_damping=0.01,
                max_depenetration_velocity=20,
                enable_gyroscopic_forces = True,
                max_contact_impulse=3000.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.41),
        ),
        collision_group=0,
    )

    
    # Contact reporter API is on the ball rigid body root from SphereCfg, not on refinement/mesh children.
    # Do not use soccer/.* here: Isaac Lab only binds the sensor to prims that have PhysxContactReportAPI.
    ball_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/soccer",
        history_length=10,
        track_air_time=True,
        update_period=0.0,
        max_contact_data_count_per_prim = 16,
        force_threshold=0.5,
        debug_vis=False,
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{MAIN_FOOT_NAME}",
        ],
    )
    
    sim_utils.spawn_rigid_body_material(
        prim_path = "/World/Materials/Soccer_Material", 
        cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.1,
            restitution=0.95,
            restitution_combine_mode="max",
            friction_combine_mode="max",
        )
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-init_yaw_range, init_yaw_range),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        ball_position_range_x=(-init_pos_range, init_pos_range ),
        ball_position_range_y=(-init_pos_range, init_pos_range ),
        ball_position_range_z=(0.0*init_pos_z_bound, init_pos_z_bound),
        ball_velocity_range = (-init_vel_range , init_vel_range ),
        static_ball_probability=static_ball_probability,
        target_pos_range={'x': (-5.0, 5.0), 'y':(4.0,10.0), 'z': (0.0, 2.0)},
        main_foot_name=MAIN_FOOT_NAME,
        critic_frame_index=critic_frame_index,
        adapt_motion_flag=adapt_motion_flag,
        jump_flag=jump_flag,
        anchor_xy_offset=anchor_xy_offset,
        threshold_r=threshold_r,
        threshold_t=threshold_t,
        lambda_factor=lambda_factor,
        critical_frame_adaptive_sampling=critical_frame_adaptive_sampling,
        critical_frame_sampling_window=critical_frame_sampling_window,
        shot_valid_steps=shot_valid_steps,
        shot_success_threshold=shot_success_threshold,
        use_ontime_ball_reset=use_ontime_ball_reset,
        goal_reward_burst_steps=goal_reward_burst_steps,
        goal_reset_delay_steps=goal_reset_delay_steps,
        stand_still_env_ratio=stand_still_env_ratio,
        warmup_steps=0,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
        # Clip raw policy outputs (before scale is applied).
        # resolve_matching_names_values requires non-overlapping patterns, so no ".*" catch-all.
        # Scales: legs/waist_yaw ~0.35-0.55 rad/unit, ankles ~0.44, wrists ~0.07.
        clip={
            # Legs: +/-3 -> +/-1.1-1.65 rad travel, well within URDF limits.
            ".*_hip_yaw_joint":   [-6.0, 6.0],
            ".*_hip_roll_joint":  [-6.0, 6.0],
            ".*_hip_pitch_joint": [-6.0, 6.0],
            ".*_knee_joint":      [-6.0, 6.0],
            # Ankle pitch: URDF limit -0.87/+0.52 rad, scale ~0.44 -> +/-1.3 rad before physics clipping.
            ".*_ankle_pitch_joint": [-3.0, 3.0],
            # Ankle roll: URDF limit +/-0.26 rad, scale ~0.44 -> clip +/-0.6.
            ".*_ankle_roll_joint":  [-0.6, 0.6],
            # Waist
            "waist_yaw_joint":   [-3.0, 3.0],
            # Waist roll/pitch: URDF limit +/-0.52 rad, scale ~0.44 -> clip +/-1.2.
            "waist_roll_joint":  [-1.2, 1.2],
            "waist_pitch_joint": [-1.2, 1.2],
            # Arms: wide URDF limits, +/-3 is within the intended control range.
            ".*_shoulder_pitch_joint": [-2.0, 2.0],
            ".*_shoulder_roll_joint":  [-3.0, 3.0],
            ".*_shoulder_yaw_joint":   [-3.0, 3.0],
            ".*_elbow_joint":          [-3.0, 3.0],
            ".*_wrist_roll_joint":     [-3.0, 3.0],
            ".*_wrist_pitch_joint":    [-3.0, 3.0],
            ".*_wrist_yaw_joint":      [-3.0, 3.0],
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5),history_length=5)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),history_length=5)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),history_length=5)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5),history_length=5)
        # clip=(-3,3): last_action returns the raw network output (unbounded); clipping it
        # to the same +/-3 used in ActionsCfg prevents extreme values from feeding back
        # into the next observation and creating a runaway divergence loop.
        actions = ObsTerm(func=mdp.last_action, history_length=5, clip=(-6.28,6.28))
        soccer_robot_relative_pos_w = ObsTerm(
            func=mdp.soccer_robot_relative_pos_w,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-10,10),
            history_length=5  
        )
        target_robot_relative_pos_w = ObsTerm(
            func=mdp.target_robot_relative_pos_w,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.1, n_max=0.1),history_length=5
        )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,history_length=5)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,history_length=5)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel,history_length=5)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,history_length=5)
        actions = ObsTerm(func=mdp.last_action,history_length=5)
        soccer_robot_relative_pos_w = ObsTerm(
            func=mdp.soccer_robot_relative_pos_w,
            clip=(-10,10),
            history_length=5
        )
        target_robot_relative_pos_w = ObsTerm(
            func=mdp.target_robot_relative_pos_w,
            params={"command_name": "motion"},history_length=5
        )
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

    

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""



    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=1.0 * motion_weight,
        params={"command_name": "motion", "std": 0.3, "error_threshold": DEFAULT_REWARD_ERROR_THRESHOLD},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=1.0 * motion_weight,
        params={"command_name": "motion", "std": 0.4, "error_threshold": DEFAULT_REWARD_ERROR_THRESHOLD},
    )

    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0*motion_weight,
        params={"command_name": "motion", "std": 0.3, "error_threshold": DEFAULT_REWARD_ERROR_THRESHOLD},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0*motion_weight,
        params={"command_name": "motion", "std": 0.4, "error_threshold": DEFAULT_REWARD_ERROR_THRESHOLD},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1*motion_weight,
        params={"command_name": "motion", "std": 1.0, "error_threshold": DEFAULT_REWARD_ERROR_THRESHOLD},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1*motion_weight, #1.0
        params={"command_name": "motion", "std": 3.14, "error_threshold": DEFAULT_REWARD_ERROR_THRESHOLD},
    )
    motion_feet_lin_vel = RewTerm(
        func=mdp.motion_global_feet_linear_velocity_error_exp,
        weight=1.0*motion_weight, # 1.0
        params={"command_name": "motion", 
                "std": 1.0,
                "error_threshold": DEFAULT_REWARD_ERROR_THRESHOLD,
                "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"], },
    )
    action_rate_l2 = RewTerm(func=mdp.my_action_rate_l2, weight=-2e-1*reg_weight)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    robot_alive = RewTerm(func=mdp.robot_alive, params={"command_name": "motion"}, weight=+0.5*goal_weight)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 100.0, #10
        },
    )

    # Locomotion regularization terms.
    feet_contact_time = RewTerm(
        func=mdp.feet_contact_time,
        weight=-0.5,
        params={"command_name": "motion",
        "threshold":0.25,
        "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=50,
        params={
            "command_name": "motion",
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "threshold": 0.15,
            "command_threshold": 0.01,
        },
    )
    feet_clearance = RewTerm(
        func=mdp.locomotion_phase_feet_clearance,
        weight=(-80.0 * (float(jump_flag))-20.0),
        params={
            "command_name": "motion",
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
            ),
            "target_height": 0.12,
            "threshold": 1.0,
            "max_penalty": 0.5,
        },
    )
    action_smoothness = RewTerm(
        func=mdp.action_smoothness,
        weight=-3e-2*reg_weight, # 3e-3
    )
    feet_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.5*reg_weight,
        params={"command_name": "motion", 
                "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"], 
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ],
                ),
            },
    )
    feet_contact_force = RewTerm(
        func=mdp.feet_contact_force_over_threshold,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
            ),
            "threshold": 400.0,
            "contact_force_scale": 0.01,
            "max_penalty": 2.0,
        },
    )
    no_fly = RewTerm(
        func=mdp.no_fly,
        weight=-1*reg_weight,
        params={"command_name": "motion", "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],},
    )
    loco_dof_vel = RewTerm(
        func=mdp.dof_vel,
        weight=-3e-5,
        params={"std": 1.0},
    )
    loco_torque = RewTerm(
        func=mdp.torque,
        weight=-2e-7,
        params={"std": 1.0},
    )
    locomotion_phase_orientation = RewTerm(
        func=mdp.locomotion_phase_orientation_l2,
        weight=-0.2 * reg_weight,
        params={"command_name": "motion"},
    )
    locomotion_phase_lin_vel_z = RewTerm(
        func=mdp.locomotion_phase_lin_vel_z_l2,
        weight=-0.5 * reg_weight,
        params={"command_name": "motion"},
    )
    locomotion_phase_torso_orientation = RewTerm(
        func=mdp.locomotion_phase_torso_orientation_l2,
        weight=-0.2 * reg_weight,
        params={"command_name": "motion", "torso_body_name": "torso_link"},
    )
    unstable_penalty = RewTerm(
        func=mdp.unstable_penalty,
        weight=-0.2*reg_weight,
        params={"command_name": "motion"},
    )
    stable_anchor_pos_tracking = RewTerm(
        func=mdp.stable_anchor_pos_tracking,
        weight=10*reg_weight*(float(adapt_motion_flag)),
        params={"command_name": "motion", "xy_weight": 1.0, "z_weight": 0.25, "std": 0.1},
    )
    stand_still_com_pos = RewTerm(
        func=mdp.stand_still_com_position_error_exp,
        weight=stand_still_reward_weight,
        params={"command_name": "motion", "std": 0.25, "error_threshold": 0.0},
    )
    stand_still_anchor_ori = RewTerm(
        func=mdp.stand_still_anchor_orientation_error_exp,
        weight=stand_still_reward_weight,
        params={"command_name": "motion", "std": 0.4, "error_threshold": 0.0},
    )
    stand_still_base_anchor_vel = RewTerm(
        func=mdp.stand_still_base_anchor_velocity_l2,
        weight=-1,
        params={
            "command_name": "motion",
            "base_lin_weight": 1.0,
            "base_ang_weight": 0.5,
            "anchor_lin_weight": 1.0,
            "anchor_ang_weight": 0.5,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
            "body_lin_weight": 0.1,
            "body_ang_weight": 0.05,
        },
    )
    cmd_global_com_pos = RewTerm(
        func=mdp.cmd_global_com_position_error_exp,
        weight=1*goal_weight*(float(adapt_motion_flag)), # 0.5
        params={"command_name": "motion", "std": 1.0, "error_threshold": DEFAULT_REWARD_ERROR_THRESHOLD},
    )
    cmd_delta_com_pos = RewTerm(
        func=mdp.cmd_delta_com_position_error_exp,
        weight=1.0*goal_weight*(float(adapt_motion_flag)), # 0.5
        params={"command_name": "motion", "std": 0.4, "error_threshold": DEFAULT_REWARD_ERROR_THRESHOLD},
    )
    cmd_global_anchor_ori = RewTerm(
        func=mdp.cmd_global_anchor_orientation_error_exp,
        weight=0.5*goal_weight*(float(adapt_motion_flag)), # 0.5
        params={"command_name": "motion", "std": 1.0, "error_threshold": DEFAULT_REWARD_ERROR_THRESHOLD},
    )
    cmd_velocity_tracking = RewTerm(
        func=mdp.cmd_velocity_tracking,
        weight=1 * goal_weight*(float(adapt_motion_flag)),
        params={"command_name": "motion", "max_vel": 0.7, "std": 0.5},
    )



    error_ball_to_target = RewTerm(
        func=mdp.error_ball_to_target,
        weight=20.0*goal_weight,
        params={"command_name": "motion", "std": 0.5 / std_difficulty_multiplier},
    )
    ball_over_line = RewTerm(
        func=mdp.ball_over_line,
        weight=0.05*goal_weight,
        params={"command_name": "motion"},
    )
    robot_ball_contact_count = RewTerm(
        func=mdp.robot_ball_contact_count,
        weight=1.0*goal_weight,
        params={"command_name": "motion", "sensor_cfg_name": "ball_contact_forces"},
    )
    robot_ball_contact = RewTerm(
        func=mdp.robot_ball_contact,
        weight=5.0*goal_weight,
        params={"command_name": "motion", 
                "force_threshold": 2.0,
                "vel_threshold": 2.0,
                "goal_sigma": 2.0 / std_difficulty_multiplier,
                "feet_sigma": 1.0 / std_difficulty_multiplier,
                "sensor_cfg_name": "ball_contact_forces",
                }
    )
    ball_velocity = RewTerm(
        func=mdp.ball_velocity,
        weight=0.4*goal_weight,
        params={"command_name": "motion",
                "sensor_cfg_name": "ball_contact_forces", 
            "force_threshold": 0.5,
            "std": 1.0,
        },
    )
    ball_contact_orientation = RewTerm(
        func=mdp.ball_contact_orientation,
        weight=2.0*goal_weight,
        params={"command_name": "motion",
                "sensor_cfg_name": "ball_contact_forces",
        },
    )

    robot_feet_ball_distance = RewTerm(
        func=mdp.robot_feet_ball_distance,
        weight=1.0*goal_weight,
        params={"command_name": "motion",
                "std": 0.5},
    )

    robot_com_ball_distance = RewTerm(
        func=mdp.robot_com_ball_distance,
        weight=1.0*goal_weight,
        params={"command_name": "motion",
                "std": 0.5},
    )

    robot_head_torso_ball_distance = RewTerm(
        func=mdp.robot_torso_ball_distance,
        weight=1.0*goal_weight,
        params={"command_name": "motion",
                "body_names": ["torso_link"],
                "std": 0.5},
    )

    # Task-specific penalty terms.
    penalize_weak_foot_contact = RewTerm(
        func=mdp.penalize_weak_foot_contact,
        weight=-0.5*goal_weight,
        params={"command_name": "motion",
                "threshold": 0.12,
                "std": 0.1},
    )

    penalize_self_contact_feet = RewTerm(
        func=mdp.penalize_self_contact_feet,
        weight=-1*goal_weight,
        params={"command_name": "motion",
                "body_names":["left_ankle_roll_link", "right_ankle_roll_link"],
                "threshold": 0.2,
                "std": 0.05},
    )

    arm_default_pose = RewTerm(
        func=mdp.arm_default_pose_penalty,
        weight=0.5 * reg_weight,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.4,
        },
    )
    hand_height_penalty = RewTerm(
        func=mdp.hand_height_penalty,
        weight=0.0,
        params={
            "command_name": "motion",
            "body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
            "max_height": 0.8,
        },
    )
    arm_pitch_same_sign_penalty = RewTerm(
        func=mdp.arm_pitch_same_sign_penalty,
        weight=0.0,
        params={
            "command_name": "motion",
            "asset_cfg": SceneEntityCfg("robot"),
            "left_joint_name": "left_shoulder_pitch_joint",
            "right_joint_name": "right_shoulder_pitch_joint",
            "deadband": 0.03,
        },
    )

    goal_reward_burst = RewTerm(
        func=mdp.goal_reward_burst,
        weight=0.0,
        params={"command_name": "motion"},
    )
    ee_body_pos_termination_penalty = RewTerm(
        func=mdp.ee_body_pos_termination_penalty,
        weight=-100.0,
        params={
            "command_name": "motion",
            "threshold": {
                "left_ankle_roll_link": 0.6,
                "right_ankle_roll_link": 0.6,
                "left_wrist_yaw_link": 0.45,
                "right_wrist_yaw_link": 0.45,
            },
            "warmup_threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    finite_check = DoneTerm(func=mdp.fail_on_non_finite_obs_or_action)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.5},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": {
                "left_ankle_roll_link": 0.55,
                "right_ankle_roll_link": 0.55,
                "left_wrist_yaw_link": 0.5,
                "right_wrist_yaw_link": 0.5,
            },
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )
    self_collision = _make_self_collision_termination()


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


##
# Environment configuration
##


def apply_tracking_task_params(env_cfg, task_params_path: str | Path | None = None) -> Path:
    """Apply a tracking preset to an instantiated TrackingEnvCfg (or subclass)."""
    tracking_env_cls = globals().get("TrackingEnvCfg")
    if tracking_env_cls is not None and not isinstance(env_cfg, tracking_env_cls):
        raise TypeError(f"Expected TrackingEnvCfg, got {type(env_cfg).__name__}.")
    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "motion"):
        raise AttributeError("Tracking env config must define commands.motion.")

    selected_task_params_path = task_params_path if task_params_path is not None else getattr(env_cfg, "task_params_path", None)
    params_path, params = load_tracking_task_params(selected_task_params_path)
    env_cfg.task_params_path = str(params_path)
    training_stage = resolve_tracking_training_stage(params_path, params)
    env_cfg.training_stage = training_stage

    velocity_range = params["velocity_range"]
    motion_cfg = env_cfg.commands.motion
    motion_cfg.stage_name = training_stage
    motion_cfg.start_time_sampling_fraction = float(params["start_time_sampling_fraction"])
    motion_cfg.pose_range["yaw"] = (-float(params["init_yaw_range"]), float(params["init_yaw_range"]))
    motion_cfg.velocity_range = velocity_range
    motion_cfg.ball_position_range_x = (-float(params["init_pos_range"]), float(params["init_pos_range"]))
    motion_cfg.ball_position_range_y = (-float(params["init_pos_range"]), float(params["init_pos_range"]))
    motion_cfg.ball_position_range_z = (0.0, float(params["init_pos_z_bound"]))
    motion_cfg.ball_velocity_range = (-float(params["init_vel_range"]), float(params["init_vel_range"]))
    motion_cfg.main_foot_name = str(params["main_foot_name"])
    motion_cfg.critic_frame_index = params["critic_frame_index"]
    motion_cfg.adapt_motion_flag = bool(params["adapt_motion_flag"])
    motion_cfg.jump_flag = bool(params["jump_flag"])
    motion_cfg.threshold_r = float(params["threshold_r"])
    motion_cfg.threshold_t = float(params["threshold_t"])
    motion_cfg.critical_frame_adaptive_sampling = bool(params["critical_frame_adaptive_sampling"])
    motion_cfg.critical_frame_sampling_window = int(params["critical_frame_sampling_window"])
    motion_cfg.anchor_xy_offset = tuple(params["anchor_xy_offset"])
    motion_cfg.lambda_factor = float(params["lambda_factor"])
    motion_cfg.static_ball_probability = float(params["static_ball_probability"])
    motion_cfg.use_ontime_ball_reset = bool(params["use_ontime_ball_reset"])
    motion_cfg.shot_valid_steps = int(params["shot_valid_steps"])
    motion_cfg.shot_success_threshold = float(params["shot_success_threshold"])
    motion_cfg.goal_reward_burst_steps = int(params["goal_reward_burst_steps"])
    motion_cfg.goal_reset_delay_steps = int(params["goal_reset_delay_steps"])
    motion_cfg.stand_still_env_ratio = float(params["stand_still_env_ratio"])
    motion_cfg.lidar_stale_probability = float(params["lidar_stale_probability"])
    env_cfg.use_rough_terrain = bool(params["use_rough_terrain"])
    env_cfg.mixed_terrain = (
        resolve_mixed_terrain_cfg(params.get("mixed_terrain")) if env_cfg.use_rough_terrain else {}
    )

    env_cfg.scene.soccer.init_state.pos = list(params["ball_init_state_pos"])
    env_cfg.scene.soccer.init_state.lin_vel = list(params["ball_init_state_vel"])
    env_cfg.scene.ball_contact_forces.prim_path = "{ENV_REGEX_NS}/soccer"
    env_cfg.scene.ball_contact_forces.filter_prim_paths_expr = [
        f"{{ENV_REGEX_NS}}/Robot/{params['main_foot_name']}",
    ]

    env_cfg.events.push_robot.params["velocity_range"] = velocity_range
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** int(params["rigid_num"])

    motion_weight = float(params["motion_weight"])
    goal_weight = float(params["goal_weight"])
    reg_weight = float(params["reg_weight"])

    def _configure_motion_reward(term_name: str, weight: float) -> None:
        reward_term = getattr(env_cfg.rewards, term_name, None)
        if reward_term is None:
            raise AttributeError(f"Motion reward term '{term_name}' is not configured.")
        reward_term.weight = weight
        reward_term.params["error_threshold"] = DEFAULT_REWARD_ERROR_THRESHOLD

    _configure_motion_reward("motion_global_anchor_pos", 1.0 * motion_weight)
    _configure_motion_reward("motion_global_anchor_ori", 1.0 * motion_weight)
    _configure_motion_reward("motion_body_pos", 1.0 * motion_weight)
    _configure_motion_reward("motion_body_ori", 1.0 * motion_weight)
    _configure_motion_reward("motion_body_lin_vel", 1.0 * motion_weight)
    _configure_motion_reward("motion_body_ang_vel", 1.0 * motion_weight)
    _configure_motion_reward("motion_feet_lin_vel", 1.0 * motion_weight)

    reward_category_weights = {
        "motion": motion_weight,
        "goal": goal_weight,
        "regularization": reg_weight,
        "reg": reg_weight,
    }
    _apply_reward_overrides(env_cfg, params, training_stage, reward_category_weights)

    if hasattr(env_cfg, "terminations"):
        _apply_termination_overrides(env_cfg, params)

    return params_path


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    task_params_path: str | None = os.environ.get(_YAML_OVERRIDE_ENV) or _DEFAULT_TASK_PARAMS_PRESET
    training_stage: str = "task_robustness"
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=15)
    use_rough_terrain: bool = use_rough_terrain
    mixed_terrain: dict = resolve_mixed_terrain_cfg(_P.get("mixed_terrain")) if use_rough_terrain else {}
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.bounce_threshold_velocity = 0.4
        # viewer settings
        self.viewer.eye = (8.0, 8.0, 5.0)
        self.viewer.origin_type = "env"
        self.viewer.asset_name = "robot"
        params_file = apply_tracking_task_params(self, self.task_params_path)
        configure_scene_terrain(self)
        print(f"[INFO] Task params YAML: {params_file}")
        print(f"[INFO] Stand-still env ratio: {self.commands.motion.stand_still_env_ratio}")
        print(f"[INFO] Terrain mode: {'generator(mesh rough)' if self.use_rough_terrain else 'plane'}")

@configclass
class ObservationsBodyFrameCfg(ObservationsCfg):
    """Observation specifications for the MDP with world position."""

    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = None
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),history_length=5)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),history_length=5,clip=(-12.56, 12.56))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5),history_length=5,clip=(-12.56, 12.56))
        actions = ObsTerm(func=mdp.last_action,history_length=5, clip=(-6.28,6.28))

        soccer_robot_relative_pos_b = ObsTerm(
            func=mdp.soccer_robot_relative_pos_b_strided_hist,
            clip=(-10,10),
        )
        target_robot_relative_pos_b = ObsTerm(
            func=mdp.target_robot_relative_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.1, n_max=0.1),history_length=5
        )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,history_length=5)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,history_length=5)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel,history_length=5, clip=(-12.56, 12.56))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,history_length=5, clip=(-12.56, 12.56))
        actions = ObsTerm(func=mdp.last_action,history_length=5, clip=(-12.56, 12.56))

        soccer_robot_relative_pos_b = ObsTerm(
            func=mdp.soccer_robot_relative_pos_b_strided_hist,
            clip=(-10,10),
        )
        target_robot_relative_pos_b = ObsTerm(
            func=mdp.target_robot_relative_pos_b,
            params={"command_name": "motion"},history_length=5
        )
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class TrackingWorldPosEnvCfg(TrackingEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsBodyFrameCfg = ObservationsBodyFrameCfg()
    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
