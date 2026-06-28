# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

import argparse
import math
import os
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

cli_args.ensure_repo_extension_on_path()

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--registry_name",
    type=str,
    default=None,
    help="WandB motion artifact containing the single reference motion used for tracking.",
)
parser.add_argument("--motion_file", type=str, default=None, help="Path to a local reference motion file.")
parser.add_argument("--noise_std", type=float, default=None, help="Initial noise std for the policy.")
parser.add_argument(
    "--yaml",
    type=str,
    default=None,
    help="Task params YAML path (e.g. right_kick/task_params.yaml or absolute path).",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if cli_args.should_export_yaml_to_tracking_env(args_cli):
    os.environ["WBT_TASK_PARAMS_YAML"] = args_cli.yaml

args_cli.enable_cameras = bool(args_cli.video)

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

installed_rsl_rl_version = importlib_metadata.version("rsl-rl-lib")
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.runtime import (
    apply_task_runtime_config,
    download_motion_source_for_env_cfg,
    resolve_motion_source_for_env_cfg,
)
from whole_body_tracking.tasks.tracking.task_params import dump_effective_tracking_task_params
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def override_policy_noise_std(runner: OnPolicyRunner, noise_std: float) -> None:
    """Apply --noise_std after checkpoint restore."""
    if noise_std <= 0.0:
        raise ValueError(f"--noise_std must be positive, got {noise_std}.")

    policy = runner.alg.get_policy()
    distribution = getattr(policy, "distribution", None)
    targets = []
    if distribution is not None:
        targets.extend(
            [
                (distribution, "std_param", noise_std),
                (distribution, "log_std_param", math.log(noise_std)),
            ]
        )
    targets.extend(
        [
            (policy, "std", noise_std),
            (policy, "log_std", math.log(noise_std)),
        ]
    )

    for owner, attr_name, value in targets:
        param = getattr(owner, attr_name, None)
        if param is None:
            continue
        with torch.no_grad():
            param.fill_(value)
        optimizer_state = getattr(runner.alg.optimizer, "state", None)
        if optimizer_state is not None:
            optimizer_state.pop(param, None)
        print(f"[INFO]: Overrode policy action noise std after checkpoint load: {noise_std}")
        return

    raise AttributeError("Cannot apply --noise_std because the policy does not expose a std parameter.")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if args_cli.wandb_path:
        agent_cfg.resume = True

    resume_path = None
    wandb_run = None
    download_dir = "./logs/rsl_rl/temp"
    if agent_cfg.resume:
        if args_cli.wandb_path:
            resume_path, wandb_run, wandb_checkpoint_file = cli_args.resolve_wandb_checkpoint(
                args_cli.wandb_path, download_dir, checkpoint_ref=agent_cfg.load_checkpoint
            )
            print(f"[INFO]: Downloaded WandB checkpoint '{wandb_checkpoint_file}' from: {args_cli.wandb_path}")
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    task_params_path = apply_task_runtime_config(
        env_cfg,
        cli_args.resolve_task_params_file(
            args_cli, checkpoint_path=resume_path, wandb_run=wandb_run, download_dir=download_dir
        ),
    )
    motion_source = None
    registry_refs: list[str] = []
    if args_cli.motion_file is not None:
        motion_source = resolve_motion_source_for_env_cfg(env_cfg, args_cli.motion_file)
        print(f"[INFO]: Using motion file from CLI: {motion_source}")
    elif args_cli.registry_name is not None:
        motion_source, registry_refs = download_motion_source_for_env_cfg(env_cfg, registry_name=args_cli.registry_name)
    if motion_source is None and wandb_run is not None:
        motion_artifact = next((artifact for artifact in wandb_run.used_artifacts() if artifact.type == "motions"), None)
        if motion_artifact is not None:
            motion_source = resolve_motion_source_for_env_cfg(env_cfg, motion_artifact.download())
            artifact_ref = getattr(motion_artifact, "source_qualified_name", None) or getattr(
                motion_artifact, "qualified_name", None
            )
            registry_refs = [artifact_ref] if artifact_ref is not None else []
            print(f"[INFO]: Using motion artifact from resumed WandB run: {artifact_ref or motion_artifact.name}")
    if motion_source is None:
        raise ValueError(
            "No motion source could be resolved. Provide --motion_file, --registry_name, or resume from a WandB "
            "run that used a motions artifact."
        )
    env_cfg.commands.motion.motion_file = str(motion_source)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    if args_cli.noise_std is not None:
        agent_cfg.policy.init_noise_std = args_cli.noise_std
        print(f"[INFO]: Using CLI policy action noise std: {args_cli.noise_std}")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    train_cfg = agent_cfg.to_dict()
    runner = OnPolicyRunner(
        env,
        train_cfg,
        log_dir=log_dir,
        device=agent_cfg.device,
        registry_name=registry_refs or None,
    )
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path, map_location=runner.device)
        if args_cli.noise_std is not None:
            override_policy_noise_std(runner, args_cli.noise_std)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    params_dir = os.path.join(log_dir, "params")
    task_params_dst = os.path.join(params_dir, "task_params.yaml")
    dump_effective_tracking_task_params(task_params_path, task_params_dst)
    print(f"[INFO] Task params saved to: {task_params_dst}")
    # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
