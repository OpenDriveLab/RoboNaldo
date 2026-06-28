from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def str_to_bool(value: str | bool) -> bool:
    """Parse boolean CLI values while also supporting argparse const flags."""
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def ensure_repo_extension_on_path() -> Path:
    """Prefer this checkout's extension over any older editable install."""
    repo_root = Path(__file__).resolve().parents[2]
    extension_path = repo_root / "source" / "whole_body_tracking"
    if not extension_path.is_dir():
        raise FileNotFoundError(f"Could not find RoboNaldo extension path: {extension_path}")

    extension_str = str(extension_path)
    if extension_str in sys.path:
        sys.path.remove(extension_str)
    sys.path.insert(0, extension_str)

    pythonpath_entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    pythonpath_entries = [entry for entry in pythonpath_entries if entry != extension_str]
    os.environ["PYTHONPATH"] = os.pathsep.join([extension_str, *pythonpath_entries])
    return extension_path


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments."""
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument(
        "--resume",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="Whether to resume from a checkpoint. Accepts a bare flag or true/false.",
    )
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument(
        "--wandb_path",
        type=str,
        default=None,
        help=(
            "WandB run path used to load a checkpoint, for example 'entity/project/run_id' or "
            "'entity/project/run_id/model_1000.pt'."
        ),
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Load and update the registered RSL-RL config."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    return update_rsl_rl_cfg(rslrl_cfg, args_cli)


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Apply CLI overrides to an RSL-RL config."""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    noise_std = getattr(args_cli, "noise_std", None)
    if noise_std is not None:
        agent_cfg.policy.init_noise_std = noise_std
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg


def resolve_checkpoint_reference(
    checkpoint_ref: str | None,
    *,
    search_roots: list[str | Path] | tuple[str | Path, ...] = (),
) -> str | None:
    """Resolve a checkpoint path from an absolute path, cwd-relative path, or shorthand under known roots."""
    if checkpoint_ref is None:
        return None

    raw = Path(checkpoint_ref).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = [raw]
    if not raw.is_absolute():
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((repo_root / raw).resolve())
        for root in search_roots:
            candidates.append((Path(root).expanduser() / raw).resolve())

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.is_file():
            return str(candidate)

    searched = ", ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(f"Could not resolve checkpoint path '{checkpoint_ref}'. Searched: {searched}")


def _is_model_checkpoint_name(file_name: str) -> bool:
    path = Path(file_name)
    return path.suffix == ".pt" and path.stem.startswith("model")


def _checkpoint_iteration(file_name: str) -> int:
    stem = Path(file_name).stem
    try:
        return int(stem.split("_", maxsplit=1)[1])
    except (IndexError, ValueError) as err:
        raise ValueError(f"Could not infer checkpoint iteration from WandB file '{file_name}'.") from err


def _normalize_wandb_checkpoint_name(checkpoint_ref: str | None) -> str | None:
    """Normalize CLI checkpoint shorthands into a WandB model filename."""
    if checkpoint_ref is None:
        return None

    checkpoint_name = Path(checkpoint_ref).name
    if checkpoint_name in {"", "model_.*.pt", "model_*.pt"}:
        return None
    if checkpoint_name.isdigit():
        checkpoint_name = f"model_{checkpoint_name}.pt"
    elif checkpoint_name.startswith("model_") and Path(checkpoint_name).suffix == "":
        checkpoint_name = f"{checkpoint_name}.pt"

    if not _is_model_checkpoint_name(checkpoint_name):
        raise ValueError(
            "WandB checkpoint overrides must be a model checkpoint filename like 'model_1000.pt' "
            "or an iteration shorthand like '1000'."
        )
    return checkpoint_name


def _normalize_wandb_run_path(wandb_path: str) -> str:
    """Normalize W&B UI paths like entity/project/runs/run_id into API run paths."""
    parsed = urlparse(wandb_path)
    raw_path = parsed.path if parsed.scheme and parsed.netloc else wandb_path
    path_parts = [part for part in raw_path.strip("/").split("/") if part]
    if len(path_parts) >= 4 and path_parts[-2] == "runs":
        path_parts.pop(-2)
    return "/".join(path_parts)


def resolve_wandb_checkpoint(
    wandb_path: str,
    download_dir: str | Path,
    checkpoint_ref: str | None = None,
) -> tuple[str, Any, str]:
    """Download a checkpoint from a WandB run and return the local path, run object, and remote file name."""
    import wandb

    normalized_wandb_path = _normalize_wandb_run_path(wandb_path)
    path_parts = [part for part in normalized_wandb_path.strip("/").split("/") if part]
    if len(path_parts) < 3:
        raise ValueError(
            "--wandb_path must be a WandB run path like 'entity/project/run_id' or "
            "'entity/project/run_id/model_1000.pt'."
        )

    checkpoint_file = _normalize_wandb_checkpoint_name(checkpoint_ref)
    if _is_model_checkpoint_name(path_parts[-1]):
        if checkpoint_file is None:
            checkpoint_file = path_parts[-1]
        run_path = "/".join(path_parts[:-1])
    else:
        run_path = "/".join(path_parts)

    api = wandb.Api()  # type: ignore[attr-defined]
    wandb_run = api.run(run_path)
    if checkpoint_file is None:
        checkpoint_files = [file.name for file in wandb_run.files() if _is_model_checkpoint_name(file.name)]
        if not checkpoint_files:
            raise FileNotFoundError(f"No model_*.pt checkpoint files found in WandB run '{run_path}'.")
        checkpoint_file = max(checkpoint_files, key=_checkpoint_iteration)

    wandb_run.file(checkpoint_file).download(str(download_dir), replace=True)
    checkpoint_path = Path(download_dir) / checkpoint_file
    return str(checkpoint_path), wandb_run, checkpoint_file


def resolve_task_params_file(
    args_cli: argparse.Namespace,
    *,
    checkpoint_path: str | None = None,
    wandb_run=None,
    download_dir: str | None = None,
) -> str:
    """Resolve the effective task-params file from CLI, local logs, or a WandB run."""
    if getattr(args_cli, "yaml", None):
        return args_cli.yaml
    if checkpoint_path is not None:
        local_params = Path(checkpoint_path).resolve().parent / "params" / "task_params.yaml"
        if local_params.is_file():
            return str(local_params)

    if wandb_run is not None and download_dir is not None:
        run_file_names = {file.name for file in wandb_run.files()}
        if "params/task_params.yaml" in run_file_names:
            wandb_run.file("params/task_params.yaml").download(download_dir, replace=True)
            downloaded_params = Path(download_dir) / "params" / "task_params.yaml"
            if downloaded_params.is_file():
                return str(downloaded_params)

    raise FileNotFoundError(
        "No task params YAML was resolved. Provide --yaml, or use a checkpoint/run that archives "
        "params/task_params.yaml."
    )


def should_export_yaml_to_tracking_env(args_cli: argparse.Namespace) -> bool:
    """Whether --yaml should be exported through the tracking env override."""
    yaml_path = getattr(args_cli, "yaml", None)
    return bool(yaml_path)
