from __future__ import annotations

from pathlib import Path

from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    apply_tracking_task_params,
    configure_scene_terrain,
)


def apply_task_runtime_config(env_cfg, task_params_path: str | Path | None = None) -> Path:
    resolved = apply_tracking_task_params(env_cfg, task_params_path)
    configure_scene_terrain(env_cfg)
    return resolved


def _normalize_registry_name(registry_name: str) -> str:
    registry_name = str(registry_name).strip()
    return registry_name if ":" in registry_name else f"{registry_name}:latest"


def download_motion_source_for_env_cfg(
    env_cfg,
    registry_name: str,
) -> tuple[Path, list[str]]:
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is None:
        raise AttributeError("Tracking env config must define commands.motion before downloading a motion artifact.")

    if not str(registry_name).strip():
        raise ValueError("registry_name must point to a WandB motion artifact.")

    import wandb

    normalized_registry = _normalize_registry_name(registry_name)
    api = wandb.Api(timeout=29)
    artifact = api.artifact(normalized_registry)
    motion_source = resolve_motion_source_for_env_cfg(env_cfg, Path(artifact.download()))
    return motion_source, [normalized_registry]


def resolve_motion_source_for_env_cfg(env_cfg, artifact_root: str | Path) -> Path:
    artifact_path = Path(artifact_root).expanduser()

    if artifact_path.is_file():
        return artifact_path.resolve()

    artifact_motion_path = artifact_path / "motion.npz"
    if artifact_motion_path.is_file():
        return artifact_motion_path.resolve()

    npz_files = sorted(artifact_path.glob("*.npz")) if artifact_path.is_dir() else []
    if len(npz_files) == 1:
        return npz_files[0].resolve()

    raise FileNotFoundError(
        f"Task '{env_cfg.__class__.__module__}' expects a single motion '.npz', but artifact '{artifact_root}' "
        "does not provide an unambiguous motion file."
    )
