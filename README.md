<div align="center">

# RoboNaldo

**Accurate, stable, and powerful humanoid soccer shooting via motion-guided curriculum reinforcement learning.**

<p>
  <a href="https://arxiv.org/abs/2606.11092"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a>
  <a href="https://opendrivelab.com/RoboNaldo/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://github.com/OpenDriveLab/RoboNaldo_Deploy/tree/f60f24459aaabc3aea9187a2b13f8923049b629c"><img src="https://img.shields.io/badge/Deploy-Code-brightgreen" alt="Deploy Code"></a>
</p>

<p>
  🌎English | <a href="README_CN.md">🇨🇳中文</a>
</p>

</div>

<p align="center">
  <img src="assets/teaser-crop.png" alt="RoboNaldo teaser" width="100%">
</p>

RoboNaldo trains Unitree G1 soccer-shooting policies in Isaac Lab.

This repository contains the simulation training code. For real-world hardware
deployment, export the trained policy to ONNX and use it with
[Deploy Repo](https://github.com/OpenDriveLab/RoboNaldo_Deploy/tree/f60f24459aaabc3aea9187a2b13f8923049b629c).

## Repository Overview

- `source/whole_body_tracking/`: Isaac Lab extension, G1 robot config, tracking task, rewards, observations, commands, and PPO config.
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/yaml/`: public right-foot task presets.
- `scripts/rsl_rl/`: direct train, play, and evaluation entrypoints.
- `docs/`: detailed setup, task-parameter, and reward references.

## News

- `2026-06` Training and deployment code release.

## Getting Started

### 1. Install Isaac Lab

Install Isaac Sim and Isaac Lab first. This codebase follows the Isaac Lab
extension layout and is intended for the Isaac Lab Python environment.

Recommended baseline:

| Dependency | Version |
| --- | --- |
| Isaac Sim | 4.5.0 |
| Isaac Lab | 2.1.0 |
| Python | 3.10 |

### 2. Install BeyondMimic

Install both the upstream
[BeyondMimic repository](https://github.com/HybridRobotics/whole_body_tracking)
and the RoboNaldo extension in the same Isaac Lab Python environment:

```bash
git clone https://github.com/HybridRobotics/whole_body_tracking.git
cd whole_body_tracking
python -m pip install -e source/whole_body_tracking
cd ..
python -m pip install -e source/whole_body_tracking
```

### 3. Download Robot Assets

The Unitree G1 description is not committed to this repository. Download it
before creating the environment from the same asset source used by BeyondMimic:

```bash
mkdir -p source/whole_body_tracking/whole_body_tracking/assets
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/
rm unitree_description.tar.gz
test -f source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf
```

The code resolves this path through `whole_body_tracking/assets.py`, where
`ASSET_DIR` points to `source/whole_body_tracking/whole_body_tracking/assets`.
Do not add an `assets/__init__.py`; unlike the upstream BeyondMimic setup, this
repository already provides the Python module that owns `ASSET_DIR`.

The downloaded `source/.../assets/` directory is ignored by `.gitignore` and
should not be committed. The soccer ball is created with Isaac Lab native
`SphereCfg`, so no separate ball mesh is required.

### 4. Prepare Motions and Checkpoints

Training requires one retargeted kick motion in RoboNaldo NPZ format. This
repository includes the open-source right-foot kick reference CSV retargeted by GVHMR+GMR:

```text
motions/right_kick_reference.csv
```

Of course you can replace it with yours.

Convert the included reference CSV into NPZ:

```bash
python scripts/csv_to_npz.py \
  --input_file motions/right_kick_reference.csv \
  --input_fps 50 \
  --output_name right_kick \
  --headless
```

Optional: upload the converted NPZ to a W&B registry:

```bash
python scripts/upload_npz.py \
  --artifact_path motions/right_kick.npz \
  --entity <entity> \
  --name right_kick
```

## Training

RoboNaldo uses a staged curriculum. Continue each stage from the previous stage checkpoint.

| Stage | Purpose | Right-foot preset |
| --- | --- | --- |
| Stage 1a | Plane motion-tracking prior, no task reward | `right_kick/tracking_params.yaml` |
| Stage 1b, optional | Mixed-terrain tracking robustness fine-tune | `right_kick/tracking_mixed_params.yaml` |
| Stage 2a | Small-range static-ball adaptation | `right_kick/task_params_1.yaml` |
| Stage 2b | Wider stationary-ball shooting | `right_kick/task_params_2.yaml` |
| Stage 3 | Dynamic incoming-ball shooting with jump trigger/adaptive sampling | `right_kick/task_params_3.yaml` |

For training, use `scripts/rsl_rl/train.py` directly:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --motion_file motions/right_kick.npz \
  --yaml right_kick/tracking_params.yaml \
  --headless \
  --logger wandb \
  --log_project_name kick \
  --run_name right_kick_tracking
```

For different stages, change the `--yaml` argument value to switch among different stage presets.

Resume training:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --motion_file motions/right_kick.npz \
  --yaml <yaml_file> \
  --resume True \
  --load_run <plane_tracking_run_folder> \
  --checkpoint model_<iter>.pt \
  --headless
```


>It is recommended to use a small policy noise std for Stage 2 and Stage 3 resume runs, so the task policy does not destroy the learned kick prior with excessive exploration.

> This release ships right-foot presets and the right-foot reference motion. A left-foot curriculum should use mirrored motion data and change `main_foot_name` to `left_ankle_roll_link`.

Use `Tracking-Body-Frame-Flat-G1-v0` registry for the paper-style body-frame observation
setup and `Tracking-Flat-G1-v0` for external-mocap-style global observation setup.

## Play and Evaluation

Use `scripts/rsl_rl/play.py` for playback. Known Stage-2 hot-test run:

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --wandb_path <your_checkpoint_path> \
  --yaml right_kick/task_params_2.yaml \
  --motion_file motions/right_kick.npz \
  --num_envs 1 \
  --headless
```

Use `scripts/rsl_rl/eval.py` for evaluation:

```bash
python scripts/rsl_rl/eval.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --wandb_path <your_checkpoint_path> \
  --yaml <your_yaml_file> \
  --motion_file motions/right_kick.npz \
  --num_envs 6000 \
  --headless
```

`eval.py` writes per-episode shot metrics and aggregate accuracy/speed summaries
under `logs/rsl_rl/eval/`.

## Deployment (ONNX Export)

Real-robot deployment expects an ONNX policy from this repo. The exporter writes
`policy-obs.onnx` with embedded metadata (joint names, PD gains, default poses,
observation/action layout, and motion anchor settings) for
[RoboNaldo_Deploy](https://github.com/OpenDriveLab/RoboNaldo_Deploy/tree/f60f24459aaabc3aea9187a2b13f8923049b629c).

| When | Output |
| --- | --- |
| W&B training (`--logger wandb`) | `<run_folder>/<run_name>.onnx` next to each saved `model_*.pt` |
| `play.py` playback | `<checkpoint_folder>/exported/policy-obs.onnx` |

Run `play.py` once on the checkpoint you plan to deploy (same `--task`, `--yaml`,
and `--motion_file` as training) to generate the ONNX artifact.

## Documentation

- [Quickstart](docs/quickstart.md)
- [中文 README](README_CN.md)
- [Task Parameters](docs/task_params.md)
- [Rewards](docs/rewards.md)

## Citation

If RoboNaldo helps your research, please consider citing:

```bibtex
@article{robonaldo2026,
  title={RoboNaldo: Accurate, stable, and powerful humanoid soccer shooting via motion-guided curriculum reinforcement learning},
  author={OpenDriveLab},
  journal={arXiv preprint arXiv:2606.11092},
  year={2026},
  url={https://arxiv.org/abs/2606.11092}
}
```

## Acknowledgement

This repository builds on [Isaac Lab (IsaacLab)](https://github.com/isaac-sim/IsaacLab),
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking), and
[RSL-RL](https://github.com/leggedrobotics/rsl_rl).
