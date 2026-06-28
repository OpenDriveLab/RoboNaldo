"""Play a trained RoboNaldo tracking policy with RSL-RL."""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

cli_args.ensure_repo_extension_on_path()

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained RoboNaldo tracking policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--no_video", dest="video", action="store_false", help="Disable video recording.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_num_episodes",
    type=int,
    default=None,
    help="Number of full episodes to record. Overrides --video_length if provided.",
)
parser.add_argument("--video_fps", type=int, default=30, help="Playback FPS of the recorded video.")
parser.add_argument(
    "--video_views",
    type=str,
    default="east,north",
    help="Comma-separated camera views for video clips (e.g., east,north).",
)
parser.add_argument(
    "--video_closeup",
    action="store_true",
    default=False,
    help="Use a near camera framing centered on env 0 (single-env style).",
)
parser.add_argument("--video_width", type=int, default=1920, help="Render width for recorded play videos.")
parser.add_argument("--video_height", type=int, default=1080, help="Render height for recorded play videos.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--disable_dr",
    action="store_true",
    default=False,
    help="Disable domain randomization terms and observation corruption during play.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--yaml",
    type=str,
    default=None,
    help="Task params YAML path (e.g. right_kick/task_params.yaml or absolute path).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if cli_args.should_export_yaml_to_tracking_env(args_cli):
    os.environ["WBT_TASK_PARAMS_YAML"] = args_cli.yaml
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.headless = True
    if hasattr(args_cli, "width"):
        args_cli.width = args_cli.video_width
    if hasattr(args_cli, "height"):
        args_cli.height = args_cli.video_height
else:
    args_cli.enable_cameras = False
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import pathlib
import torch

from importlib import metadata as importlib_metadata

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

installed_rsl_rl_version = importlib_metadata.version("rsl-rl-lib")
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.runtime import (
    apply_task_runtime_config,
    resolve_motion_source_for_env_cfg,
)
from whole_body_tracking.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    resolve_policy_module,
    resolve_policy_normalizer,
)

def _parse_video_views(raw_views: str) -> list[str]:
    """Parse and validate video view names from a CSV string."""
    requested = [item.strip().lower() for item in raw_views.split(",") if item.strip()]
    if not requested:
        raise ValueError("--video_views must contain at least one view.")
    valid_views = {"east", "west", "north", "south", "isometric"}
    invalid = [view for view in requested if view not in valid_views]
    if invalid:
        raise ValueError(f"Unsupported --video_views entries: {invalid}. Valid values: {sorted(valid_views)}")
    return requested


def _set_wide_video_camera_view(env, view_name: str) -> None:
    """Place a wider-angle camera with a safer stand-off from the scene."""
    scene = env.unwrapped.scene
    if not hasattr(scene, "env_origins"):
        return

    env_origins = scene.env_origins
    if env_origins.numel() == 0:
        return

    min_corner = torch.min(env_origins, dim=0).values
    max_corner = torch.max(env_origins, dim=0).values

    center_x = 0.5 * (min_corner[0] + max_corner[0])
    center_y = 0.5 * (min_corner[1] + max_corner[1])
    span_x = max_corner[0] - min_corner[0]
    span_y = max_corner[1] - min_corner[1]
    widest_span = max(float(span_x), float(span_y), 1.0)

    # Keep a wide shot, but bias slightly farther away so the robot remains
    # visible across different tasks and scene scales.
    horizontal_half_extent = 0.5 * math.sqrt(float(span_x) ** 2 + float(span_y) ** 2)
    side_distance = max(7.5, 1.25 * horizontal_half_extent / math.tan(math.radians(50.0)))
    eye_height = max(2.8, 0.16 * widest_span + 1.0)

    eye_by_view = {
        "east": (float(max_corner[0]) + side_distance, float(center_y), float(eye_height)),
        "west": (float(min_corner[0]) - side_distance, float(center_y), float(eye_height)),
        "north": (float(center_x), float(max_corner[1]) + side_distance, float(eye_height)),
        "south": (float(center_x), float(min_corner[1]) - side_distance, float(eye_height)),
        "isometric": (
            float(max_corner[0]) + 0.75 * side_distance,
            float(max_corner[1]) + 0.75 * side_distance,
            float(eye_height + 0.5),
        ),
    }

    eye = eye_by_view[view_name]
    lookat = (float(center_x), float(center_y), 1.0)
    env.unwrapped.sim.set_camera_view(eye, lookat)


def _set_close_video_camera_view(env, view_name: str, env_id: int = 0) -> None:
    """Place a closer camera around one env (default env 0)."""
    scene = env.unwrapped.scene
    if not hasattr(scene, "env_origins"):
        return

    env_origins = scene.env_origins
    if env_origins.numel() == 0:
        return

    clamped_env_id = max(0, min(int(env_id), env_origins.shape[0] - 1))
    origin = env_origins[clamped_env_id]
    center_x = float(origin[0])
    center_y = float(origin[1])

    # For kicking tasks, action happens mostly in front of the robot (+Y).
    # Bias lookat a bit farther and back the camera up so target markers stay in-frame.
    lookat = (center_x, center_y + 6.0, 1.0)
    half_extent = 7.5
    side_distance = max(7.5, 1.15 * half_extent / math.tan(math.radians(52.0)))
    eye_height = 2.7

    eye_by_view = {
        "east": (lookat[0] + side_distance, lookat[1], eye_height),
        "west": (lookat[0] - side_distance, lookat[1], eye_height),
        "north": (lookat[0], lookat[1] + side_distance, eye_height),
        "south": (lookat[0], lookat[1] - side_distance, eye_height),
        "isometric": (lookat[0] + 0.75 * side_distance, lookat[1] + 0.75 * side_distance, eye_height + 0.5),
    }
    env.unwrapped.sim.set_camera_view(eye_by_view[view_name], lookat)


def _set_video_camera_view(env, view_name: str) -> None:
    """Dispatch between wide and close-up video camera layouts."""
    if args_cli.video_closeup:
        _set_close_video_camera_view(env, view_name=view_name, env_id=0)
    else:
        _set_wide_video_camera_view(env, view_name=view_name)


def _resolve_video_length(env_cfg) -> int:
    """Resolve the recording length in steps from CLI settings."""
    if args_cli.video_num_episodes is None:
        return args_cli.video_length
    if args_cli.video_num_episodes <= 0:
        raise ValueError("--video_num_episodes must be a positive integer.")

    episode_steps = math.ceil(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    return args_cli.video_num_episodes * episode_steps


def _disable_interval_push(env_cfg) -> None:
    """Disable training-time interval pushes for deterministic play."""
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None


def _disable_domain_randomization(env_cfg) -> list[str]:
    """Disable DR-related event terms and observation corruption for play."""
    disabled_terms: list[str] = []

    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is not None:
        for term_name in dir(events_cfg):
            if term_name.startswith("_"):
                continue
            term_cfg = getattr(events_cfg, term_name)
            if term_cfg is None:
                continue
            term_mode = str(getattr(term_cfg, "mode", "") or "").lower()
            term_func_name = str(getattr(getattr(term_cfg, "func", None), "__name__", "")).lower()
            is_dr_term = (
                ("randomize" in term_func_name)
                or ("noise" in term_func_name)
                or ("delay" in term_func_name)
                or ("push" in term_func_name)
                or term_mode == "interval"
            )
            if is_dr_term:
                setattr(events_cfg, term_name, None)
                disabled_terms.append(term_name)

    observations_cfg = getattr(env_cfg, "observations", None)
    if observations_cfg is not None:
        for group_name in ("policy", "critic"):
            obs_group = getattr(observations_cfg, group_name, None)
            if obs_group is not None and hasattr(obs_group, "enable_corruption"):
                if bool(getattr(obs_group, "enable_corruption")):
                    setattr(obs_group, "enable_corruption", False)
                    disabled_terms.append(f"observations.{group_name}.enable_corruption")

    return disabled_terms


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.scene.env_spacing = 15
    if args_cli.video:
        env_cfg.viewer.origin_type = "world"

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    wandb_run = None
    download_dir = "./logs/rsl_rl/temp"

    if args_cli.wandb_path:
        resume_path, wandb_run, file = cli_args.resolve_wandb_checkpoint(
            args_cli.wandb_path,
            download_dir,
            checkpoint_ref=args_cli.checkpoint,
        )
        print(f"[INFO]: Loading model checkpoint from WandB: {'/'.join(wandb_run.path)}/{file}")

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file
        else:
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is None:
                raise FileNotFoundError(
                    "No motions artifact found in the WandB run. Provide --motion_file for a local reference motion."
                )
            env_cfg.commands.motion.motion_file = str(
                resolve_motion_source_for_env_cfg(env_cfg, pathlib.Path(art.download()))
            )

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.motion_file is None:
            raise ValueError("Playing a local checkpoint requires --motion_file because no WandB motion artifact is available.")
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    task_params_path = cli_args.resolve_task_params_file(
        args_cli, checkpoint_path=resume_path, wandb_run=wandb_run, download_dir=download_dir
    )
    apply_task_runtime_config(env_cfg, task_params_path)
    if args_cli.disable_dr:
        disabled_terms = _disable_domain_randomization(env_cfg)
        print(
            "[INFO] Disabled domain randomization for play. "
            f"Overridden terms: {disabled_terms if disabled_terms else 'none'}"
    )
    _disable_interval_push(env_cfg)

    video_views = _parse_video_views(args_cli.video_views) if args_cli.video else []
    segment_video_length = _resolve_video_length(env_cfg) if args_cli.video else 0
    total_video_length = segment_video_length * len(video_views) if args_cli.video else 0

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        env.metadata["render_fps"] = args_cli.video_fps
        video_folder = os.path.abspath(os.path.join("logs", "videos"))
        os.makedirs(video_folder, exist_ok=True)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step < total_video_length and step % segment_video_length == 0,
            "video_length": segment_video_length,
            "disable_logger": True,
        }
        if args_cli.video_num_episodes is None:
            print(
                "[INFO] Recording video "
                f"({segment_video_length} steps/view x {len(video_views)} views, {args_cli.video_fps} fps, "
                f"{args_cli.video_width}x{args_cli.video_height}) -> {video_folder}"
            )
        else:
            print(
                "[INFO] Recording video "
                f"({args_cli.video_num_episodes} episodes/view, {segment_video_length} steps/view x {len(video_views)} views, "
                f"{args_cli.video_fps} fps, {args_cli.video_width}x{args_cli.video_height}) -> {video_folder}"
            )
        print(f"[INFO] Video views order: {video_views}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)

    # load previously trained model
    train_cfg = agent_cfg.to_dict()
    ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, map_location=ppo_runner.device)


    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy = resolve_policy_module(ppo_runner.alg)
    export_normalizer = resolve_policy_normalizer(ppo_runner.alg)

    export_motion_policy_as_onnx(
        env.unwrapped,
        export_policy,
        normalizer=export_normalizer,
        path=export_model_dir,
        filename="policy-obs.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    # Reset once before play so command terms resample robot/ball state.
    obs, _ = env.reset()
    active_view_idx = 0
    if args_cli.video:
        _set_video_camera_view(env, video_views[active_view_idx])
        print(f"[INFO] Video camera mode: {'closeup(env0)' if args_cli.video_closeup else 'wide(multi-env)'}")
        print(f"[INFO] Active video view: {video_views[active_view_idx]}")
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        if args_cli.video:
            next_view_idx = min(timestep // segment_video_length, len(video_views) - 1)
            if next_view_idx != active_view_idx:
                active_view_idx = next_view_idx
                _set_video_camera_view(env, video_views[active_view_idx])
                print(f"[INFO] Switched video view: {video_views[active_view_idx]} (step {timestep})")
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording all requested views.
            if timestep == total_video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
