from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    TrackingEnvCfg,
    TrackingWorldPosEnvCfg,
)


G1_TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


def _apply_g1_robot_settings(env_cfg) -> None:
    env_cfg.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.actions.joint_pos.scale = G1_ACTION_SCALE
    env_cfg.commands.motion.anchor_body_name = "torso_link"
    env_cfg.commands.motion.body_names = G1_TRACKED_BODY_NAMES

@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_g1_robot_settings(self)

@configclass
class G1FlatBodyFrameEnvCfg(TrackingWorldPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_g1_robot_settings(self)
