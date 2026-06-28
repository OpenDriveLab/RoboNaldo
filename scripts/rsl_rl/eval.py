"""Evaluate a trained RoboNaldo tracking policy with RSL-RL."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

cli_args.ensure_repo_extension_on_path()

os.environ.setdefault("MPLBACKEND", "Agg")

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(line_buffering=True, write_through=True)


parser = argparse.ArgumentParser(description="Evaluate a trained RoboNaldo tracking policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record one evaluation video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in simulation steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to a local reference motion file.")
parser.add_argument(
    "--yaml",
    type=str,
    default=None,
    help="Task params YAML path, for example right_kick/task_params_3.yaml or an absolute path.",
)
parser.add_argument("--eval_steps", type=int, default=2000, help="Number of simulation steps to evaluate.")
parser.add_argument(
    "--eval_print_every",
    type=int,
    default=100,
    help="Print aggregated evaluation stats every N simulation steps. Set <=0 to disable progress logs.",
)
parser.add_argument(
    "--eval_output_dir",
    type=str,
    default=None,
    help="Directory for evaluation JSON output. Defaults to logs/rsl_rl/eval/<checkpoint>.",
)
parser.add_argument("--eval_label", type=str, default=None, help="Optional label used in the evaluation output path.")
parser.add_argument("--seed", type=int, default=None, help="Random seed passed through to the RSL-RL config.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.eval_steps <= 0:
    raise ValueError("--eval_steps must be positive.")
if cli_args.should_export_yaml_to_tracking_env(args_cli):
    os.environ["WBT_TASK_PARAMS_YAML"] = args_cli.yaml

args_cli.enable_cameras = bool(args_cli.video)

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402,F401
from whole_body_tracking.tasks.runtime import (  # noqa: E402
    apply_task_runtime_config,
    resolve_motion_source_for_env_cfg,
)

installed_rsl_rl_version = importlib_metadata.version("rsl-rl-lib")


def _safe_label(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "eval"


def _resolve_eval_output_dir(resume_path: str, wandb_run: Any | None) -> str:
    if args_cli.eval_output_dir:
        return args_cli.eval_output_dir
    if args_cli.eval_label:
        label = args_cli.eval_label
    elif wandb_run is not None:
        label = "_".join(str(part) for part in wandb_run.path[-2:])
    else:
        checkpoint = Path(resume_path)
        label = f"{checkpoint.parent.name}_{checkpoint.stem}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("logs", "rsl_rl", "eval", f"{_safe_label(label)}_{timestamp}")


def _motion_command(env) -> Any:
    return env.unwrapped.command_manager.get_term("motion")


def _zeros_bool(num_envs: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(num_envs, dtype=torch.bool, device=device)


def _as_done_tensor(dones: Any, num_envs: int, device: torch.device) -> torch.Tensor:
    if isinstance(dones, torch.Tensor):
        return dones.to(device=device, dtype=torch.bool).reshape(-1)
    if isinstance(dones, dict):
        if "policy" in dones:
            return _as_done_tensor(dones["policy"], num_envs, device)
        merged = _zeros_bool(num_envs, device)
        for value in dones.values():
            merged |= _as_done_tensor(value, num_envs, device)
        return merged
    return torch.as_tensor(dones, dtype=torch.bool, device=device).reshape(-1)


def _finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return float(value)


def _sync_eval_adapt_motion_command(env, env_ids: torch.Tensor | None = None) -> None:
    """Refresh rule-based adapted command buffers before the first policy step."""
    cmd = _motion_command(env)
    if not hasattr(cmd, "_compute_rule_based_adapt_motion"):
        raise AttributeError("Motion command does not expose the rule-based adapted-anchor planner.")
    if not all(hasattr(cmd, name) for name in ("adapt_motion_pos", "adapt_motion_ori")):
        raise AttributeError("Motion command is missing adapted-anchor command buffers.")

    device = env.unwrapped.device
    if env_ids is None:
        selected_env_ids = torch.arange(env.unwrapped.num_envs, device=device, dtype=torch.long)
    else:
        selected_env_ids = env_ids.to(device=device, dtype=torch.long)
    if selected_env_ids.numel() == 0:
        return

    adapt_pos, adapt_ori = cmd._compute_rule_based_adapt_motion()
    cmd.adapt_motion_pos[selected_env_ids] = adapt_pos[selected_env_ids]
    cmd.adapt_motion_ori[selected_env_ids] = adapt_ori[selected_env_ids]
    if not all(hasattr(cmd, name) for name in ("prev_adapt_motion_pos", "prev_adapt_motion_ori")):
        raise AttributeError("Motion command is missing previous adapted-anchor command buffers.")
    cmd.prev_adapt_motion_pos[selected_env_ids] = adapt_pos[selected_env_ids]
    cmd.prev_adapt_motion_ori[selected_env_ids] = adapt_ori[selected_env_ids]


class ShotEvalAccumulator:
    """Accumulate one shot-quality sample per episode."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.contact_seen = _zeros_bool(num_envs, device)
        self.sample_recorded = _zeros_bool(num_envs, device)
        self.plane_cross_recorded = _zeros_bool(num_envs, device)
        self.episode_min_distance = torch.full((num_envs,), float("inf"), device=device)
        self.episode_peak_velocity = torch.zeros(num_envs, device=device)
        self.records: list[dict[str, float | bool | None]] = []

    def reset_episode_state(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            mask = _zeros_bool(self.num_envs, self.device)
            mask[env_ids.to(device=self.device, dtype=torch.long)] = True
        self.contact_seen[mask] = False
        self.sample_recorded[mask] = False
        self.plane_cross_recorded[mask] = False
        self.episode_min_distance[mask] = float("inf")
        self.episode_peak_velocity[mask] = 0.0

    def update(
        self,
        cmd: Any,
        *,
        prev_ball_pos: torch.Tensor,
        prev_ball_velocity: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        ball_pos = cmd.ball_pos
        ball_velocity = cmd.ball_velocity
        target_pos = cmd.target_pos

        if not hasattr(cmd, "contacted_flag"):
            raise AttributeError("Motion command is missing contacted_flag required for evaluation metrics.")
        contacted = cmd.contacted_flag.to(dtype=torch.bool)
        self.contact_seen |= contacted

        distance_to_target = torch.norm(ball_pos - target_pos, dim=-1)
        self.episode_min_distance = torch.minimum(self.episode_min_distance, distance_to_target)
        self.episode_peak_velocity = torch.maximum(self.episode_peak_velocity, torch.norm(ball_velocity, dim=-1))

        valid_plane_cross, xz_error, speed_at_plane = self._plane_crossing(
            cmd,
            prev_ball_pos=prev_ball_pos,
            prev_ball_velocity=prev_ball_velocity,
        )
        valid_plane_cross &= ~self.sample_recorded

        if valid_plane_cross.any():
            self._append_samples(valid_plane_cross, plane_cross=True, xz_error=xz_error, speed_at_plane=speed_at_plane)
            self.sample_recorded[valid_plane_cross] = True
            self.plane_cross_recorded[valid_plane_cross] = True

        timeout_samples = dones & ~self.sample_recorded
        if timeout_samples.any():
            self._append_samples(timeout_samples, plane_cross=False)
            self.sample_recorded[timeout_samples] = True

        if dones.any():
            self.reset_episode_state(torch.nonzero(dones, as_tuple=False).squeeze(-1))

    def finalize_open_episodes(self) -> None:
        pending = ~self.sample_recorded
        if pending.any():
            self._append_samples(pending, plane_cross=False)
            self.sample_recorded[pending] = True

    def _plane_crossing(
        self,
        cmd: Any,
        *,
        prev_ball_pos: torch.Tensor,
        prev_ball_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ball_pos = cmd.ball_pos
        ball_velocity = cmd.ball_velocity
        target_pos = cmd.target_pos

        prev_to_target_y = target_pos[:, 1] - prev_ball_pos[:, 1]
        current_to_target_y = target_pos[:, 1] - ball_pos[:, 1]
        valid_plane_cross = (
            (prev_to_target_y >= 0.0)
            & (current_to_target_y <= 0.0)
            & (ball_velocity[:, 1] > 0.0)
            & (~self.plane_cross_recorded)
        )

        denom = ball_pos[:, 1] - prev_ball_pos[:, 1]
        safe_denom = torch.where(denom.abs() > 1e-6, denom, torch.ones_like(denom))
        alpha = ((target_pos[:, 1] - prev_ball_pos[:, 1]) / safe_denom).clamp(0.0, 1.0).unsqueeze(-1)
        plane_pos = prev_ball_pos + alpha * (ball_pos - prev_ball_pos)
        plane_vel = prev_ball_velocity + alpha * (ball_velocity - prev_ball_velocity)
        plane_pos[:, 1] = target_pos[:, 1]
        xz_error = torch.norm(plane_pos[:, [0, 2]] - target_pos[:, [0, 2]], dim=-1)
        speed_at_plane = torch.norm(plane_vel, dim=-1)
        return valid_plane_cross, xz_error, speed_at_plane

    def _append_samples(
        self,
        mask: torch.Tensor,
        *,
        plane_cross: bool,
        xz_error: torch.Tensor | None = None,
        speed_at_plane: torch.Tensor | None = None,
    ) -> None:
        env_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        for env_id_tensor in env_ids.detach().cpu():
            env_id = int(env_id_tensor.item())
            self.records.append(
                {
                    "contact": bool(self.contact_seen[env_id].item()),
                    "plane_cross": bool(plane_cross),
                    "success_0.5m": bool(self.episode_min_distance[env_id].item() <= 0.5),
                    "success_1.0m": bool(self.episode_min_distance[env_id].item() <= 1.0),
                    "success_1.5m": bool(self.episode_min_distance[env_id].item() <= 1.5),
                    "min_ball_target_distance": _finite_float(self.episode_min_distance[env_id].item()),
                    "xz_error_at_target_plane": _finite_float(xz_error[env_id].item() if xz_error is not None else None),
                    "speed_at_target_plane": _finite_float(
                        speed_at_plane[env_id].item() if speed_at_plane is not None else None
                    ),
                    "peak_ball_velocity": _finite_float(self.episode_peak_velocity[env_id].item()),
                }
            )


def _mean(values: list[float]) -> float | None:
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _rate(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return sum(1 for record in records if bool(record.get(key))) / len(records)


def _summary(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    xz_errors = [record["xz_error_at_target_plane"] for record in records if record["xz_error_at_target_plane"] is not None]
    speeds_at_plane = [record["speed_at_target_plane"] for record in records if record["speed_at_target_plane"] is not None]
    peak_velocities = [record["peak_ball_velocity"] for record in records if record["peak_ball_velocity"] is not None]
    min_distances = [record["min_ball_target_distance"] for record in records if record["min_ball_target_distance"] is not None]
    return {
        "samples": len(records),
        "contact_rate": _rate(records, "contact"),
        "plane_cross_rate": _rate(records, "plane_cross"),
        "success_0.5m": _rate(records, "success_0.5m"),
        "success_1.0m": _rate(records, "success_1.0m"),
        "success_1.5m": _rate(records, "success_1.5m"),
        "mean_min_ball_target_distance": _mean(min_distances),
        "mean_xz_error_at_target_plane": _mean(xz_errors),
        "mean_speed_at_target_plane": _mean(speeds_at_plane),
        "mean_peak_ball_velocity": _mean(peak_velocities),
        "max_peak_ball_velocity": max(peak_velocities) if peak_velocities else None,
    }


def _fmt(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}{suffix}"


def _print_summary(summary: dict[str, float | int | None]) -> None:
    print(
        "[EVAL] "
        f"samples={summary['samples']} "
        f"contact_rate={_fmt(summary['contact_rate'])} "
        f"plane_cross_rate={_fmt(summary['plane_cross_rate'])} "
        f"success@0.5m={_fmt(summary['success_0.5m'])} "
        f"success@1.0m={_fmt(summary['success_1.0m'])} "
        f"success@1.5m={_fmt(summary['success_1.5m'])} "
        f"mean_min_dist={_fmt(summary['mean_min_ball_target_distance'], 'm')} "
        f"mean_xz_error={_fmt(summary['mean_xz_error_at_target_plane'], 'm')} "
        f"mean_peak_v={_fmt(summary['mean_peak_ball_velocity'], 'm/s')} "
        f"max_peak_v={_fmt(summary['max_peak_ball_velocity'], 'm/s')}",
        flush=True,
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    wandb_run = None
    download_dir = os.path.join("logs", "rsl_rl", "temp", f"eval_{os.getpid()}")

    if args_cli.wandb_path:
        resume_path, wandb_run, checkpoint_file = cli_args.resolve_wandb_checkpoint(
            args_cli.wandb_path,
            download_dir,
            checkpoint_ref=args_cli.checkpoint,
        )
        print(f"[INFO] Loading checkpoint from WandB: {'/'.join(wandb_run.path)}/{checkpoint_file}", flush=True)
        if args_cli.motion_file is None:
            artifact = next((item for item in wandb_run.used_artifacts() if item.type == "motions"), None)
            if artifact is not None:
                env_cfg.commands.motion.motion_file = str(
                    resolve_motion_source_for_env_cfg(env_cfg, Path(artifact.download()))
                )
                print(f"[INFO] Using motion artifact from WandB run: {artifact.name}", flush=True)
            else:
                raise FileNotFoundError(
                    "No motions artifact found in the WandB run. Provide --motion_file for a local reference motion."
                )
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}", flush=True)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading checkpoint from: {resume_path}", flush=True)
        if args_cli.motion_file is None:
            raise ValueError("Evaluating a local checkpoint requires --motion_file because no WandB motion artifact is available.")

    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        print(f"[INFO] Using motion file from CLI: {args_cli.motion_file}", flush=True)

    task_params_path = cli_args.resolve_task_params_file(
        args_cli,
        checkpoint_path=resume_path,
        wandb_run=wandb_run,
        download_dir=download_dir,
    )
    apply_task_runtime_config(env_cfg, task_params_path)
    eval_output_dir = _resolve_eval_output_dir(resume_path, wandb_run)
    os.makedirs(eval_output_dir, exist_ok=True)
    print(f"[INFO] Evaluation output directory: {eval_output_dir}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(eval_output_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording evaluation video.", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    train_cfg = agent_cfg.to_dict()
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, map_location=runner.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.reset()
    _sync_eval_adapt_motion_command(env)
    obs = env.get_observations()
    cmd = _motion_command(env)
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    accumulator = ShotEvalAccumulator(num_envs, device)
    prev_ball_pos = cmd.ball_pos.clone()
    prev_ball_velocity = cmd.ball_velocity.clone()

    for step in range(args_cli.eval_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        cmd = _motion_command(env)
        done_tensor = _as_done_tensor(dones, num_envs, device)
        accumulator.update(
            cmd,
            prev_ball_pos=prev_ball_pos,
            prev_ball_velocity=prev_ball_velocity,
            dones=done_tensor,
        )
        prev_ball_pos = cmd.ball_pos.clone()
        prev_ball_velocity = cmd.ball_velocity.clone()

        if args_cli.eval_print_every > 0 and (step + 1) % args_cli.eval_print_every == 0:
            _print_summary(_summary(accumulator.records))

    accumulator.finalize_open_episodes()
    summary = _summary(accumulator.records)
    _print_summary(summary)

    output_path = os.path.join(eval_output_dir, "metrics.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "summary": summary,
                "records": accumulator.records,
                "checkpoint": str(resume_path),
                "task": args_cli.task,
                "num_envs": num_envs,
                "eval_steps": args_cli.eval_steps,
            },
            output_file,
            indent=2,
        )
    print(f"[INFO] Wrote evaluation metrics: {output_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
