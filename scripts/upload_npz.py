"""Upload a converted motion artifact to a WandB registry."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


parser = argparse.ArgumentParser(description="Upload a motion artifact to WandB.")
parser.add_argument("--artifact_path", type=Path, required=True, help="Path to a motion .npz file.")
parser.add_argument("--name", type=str, required=True, help="Artifact collection name.")
parser.add_argument("--type", type=str, default="motions", help="WandB artifact type.")
parser.add_argument("--project", type=str, default="motion-registry", help="Temporary WandB project for upload.")
parser.add_argument("--entity", type=str, default=None, help="Optional WandB entity.")
parser.add_argument("--alias", type=str, default="latest", help="Artifact alias to attach.")
args = parser.parse_args()

if not args.artifact_path.is_file():
    raise FileNotFoundError(f"Motion file not found: {args.artifact_path}")

run = wandb.init(project=args.project, entity=args.entity, job_type="upload_motion", name=args.name)
try:
    artifact = wandb.Artifact(name=args.name, type=args.type)
    artifact.add_file(str(args.artifact_path), name="motion.npz")
    logged_artifact = run.log_artifact(artifact, aliases=[args.alias])
    run.link_artifact(
        artifact=logged_artifact,
        target_path=f"wandb-registry-{args.type}/{args.name}",
        aliases=[args.alias],
    )
    print(f"[INFO] Uploaded {args.artifact_path} to wandb-registry-{args.type}/{args.name}:{args.alias}")
finally:
    run.finish()
