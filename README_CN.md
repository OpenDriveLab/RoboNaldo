<div align="center">

# RoboNaldo

**Accurate, stable, and powerful humanoid soccer shooting via motion-guided curriculum reinforcement learning.**

<p>
  <a href="https://arxiv.org/abs/2606.11092"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a>
  <a href="https://opendrivelab.com/RoboNaldo/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://github.com/OpenDriveLab/RoboNaldo_Deploy/tree/f60f24459aaabc3aea9187a2b13f8923049b629c"><img src="https://img.shields.io/badge/Deploy-Code-brightgreen" alt="Deploy Code"></a>
</p>

<p>
  <a href="README.md">🌎English</a> | 🇨🇳中文
</p>

</div>

<p align="center">
  <img src="assets/teaser-crop.png" alt="RoboNaldo teaser" width="100%">
</p>

RoboNaldo 在 Isaac Lab 中训练 Unitree G1 足球射门策略。训练从一个重定向的人类踢球参考动作出发，逐步过渡到静态球、动态来球下的精准大力射门。

本仓库包含仿真训练代码。真实机器人部署需要先将训练好的策略导出为 ONNX，再配合
[Deploy Repo](https://github.com/OpenDriveLab/RoboNaldo_Deploy/tree/f60f24459aaabc3aea9187a2b13f8923049b629c) 使用。

## 仓库结构

- `source/whole_body_tracking/`：Isaac Lab extension、G1 机器人配置、tracking task、reward、observation、command、PPO 配置。
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/yaml/`：公开的右脚训练课程 preset。
- `motions/right_kick_reference.csv`：开源右脚踢球参考动作 CSV。
- `scripts/rsl_rl/`：训练、播放、评估入口。
- `docs/`：更详细的 quickstart、任务参数和 reward 文档。

## 新闻

- `2026-06` 训练和部署代码发布。

## 快速开始

### 1. 安装 Isaac Lab

请先安装 Isaac Sim 和 Isaac Lab。本仓库采用 Isaac Lab extension 布局，需要在能导入 `isaaclab`、`isaaclab_tasks`、`isaaclab_rl` 的 Python 环境中运行。

推荐版本：

| 依赖 | 版本 |
| --- | --- |
| Isaac Sim | 4.5.0 |
| Isaac Lab | 2.1.0 |
| Python | 3.10 |

### 2. 安装 BeyondMimic

请在同一个 Isaac Lab Python 环境中安装
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)：

```bash
git clone https://github.com/HybridRobotics/whole_body_tracking.git
cd whole_body_tracking
python -m pip install -e .
cd ..
python -m pip install -e source/whole_body_tracking
```

### 3. 下载机器人资产

Unitree G1 描述文件不随仓库提交。创建环境前需要从 BeyondMimic 使用的同一来源下载：

```bash
mkdir -p source/whole_body_tracking/whole_body_tracking/assets
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/
rm unitree_description.tar.gz
test -f source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf
```

本仓库通过 `whole_body_tracking/assets.py` 中的 `ASSET_DIR` 定位资产目录，不需要像旧版 BeyondMimic README 那样额外创建 `assets/__init__.py`。

下载后的 `source/.../assets/` 已被 `.gitignore` 忽略，不应提交。足球使用 Isaac Lab 原生 `SphereCfg` 创建，不需要额外球体 mesh。

### 4. 准备参考动作和 checkpoint

仓库已开源右脚踢球参考动作 CSV：

```text
motions/right_kick_reference.csv
```


将 CSV 转换为训练使用的 NPZ：

```bash
python scripts/csv_to_npz.py \
  --input_file motions/right_kick_reference.csv \
  --input_fps 50 \
  --output_name right_kick \
  --headless
```

可选：将转换后的 NPZ 上传到 W&B registry：

```bash
python scripts/upload_npz.py \
  --artifact_path motions/right_kick.npz \
  --entity <entity> \
  --name right_kick
```

## 训练

直接使用 `scripts/rsl_rl/train.py`：

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

如果你把动作保存在 W&B artifact registry 中，也可以用
`--registry_name <entity>/wandb-registry-motions/right_kick:latest` 代替
`--motion_file`。

RoboNaldo 使用分阶段课程训练。每个阶段从上一阶段 checkpoint 继续训练。建议先在平地上训练 tracking prior；当动作跟踪稳定后，再可选使用 mixed-terrain preset 从平地 checkpoint 微调，以增强鲁棒性。直接从零开始训练 mixed terrain 难度更高，不建议作为第一阶段。

| 论文阶段 | 目的 | 右脚 preset |
| --- | --- | --- |
| Stage 1a | 平地动作跟踪先验，无任务奖励 | `right_kick/tracking_params.yaml` |
| Stage 1b，可选 | mixed terrain tracking 鲁棒性微调 | `right_kick/tracking_mixed_params.yaml` |
| Stage 2a | 小范围静态球任务适应 | `right_kick/task_params_1.yaml` |
| Stage 2b | 更大范围静态球射门 | `right_kick/task_params_2.yaml` |
| Stage 3 | 动态来球射门，启用 jump trigger / adaptive sampling | `right_kick/task_params_3.yaml` |

Stage 2 和 Stage 3 从 checkpoint 继续训练时建议使用较小的策略噪声，避免过强探索破坏已经学到的踢球先验。

当前 release 提供右脚 preset 和右脚参考动作。左脚课程需要使用镜像后的动作数据，并将 `main_foot_name` 改为 `left_ankle_roll_link`。

从本地平地 tracking checkpoint 继续做 mixed-terrain 微调：

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --motion_file motions/right_kick.npz \
  --yaml right_kick/tracking_mixed_params.yaml \
  --resume True \
  --load_run <plane_tracking_run_folder> \
  --checkpoint model_<iter>.pt \
  --headless
```

`--wandb_path` 可以传 W&B 网页 URL 或
`entity/project/run_id` 路径，默认自动加载最新 `model_*.pt`，也可用
`--checkpoint` 指定某个 checkpoint。

论文默认设置建议使用 `Tracking-Body-Frame-Flat-G1-v0`。`Tracking-Flat-G1-v0` 仍保留给世界坐标观测实验。

## 播放和评估

使用 `scripts/rsl_rl/play.py` 播放策略。可直接使用的 Stage-2 hot-test run：

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --wandb_path <your_checkpoint_path> \
  --yaml right_kick/task_params_2.yaml \
  --motion_file motions/right_kick.npz \
  --num_envs 1 \
  --headless
```

使用 `scripts/rsl_rl/eval.py` 评估：

```bash
python scripts/rsl_rl/eval.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --wandb_path <your_checkpoint_path> \
  --yaml right_kick/task_params_2.yaml \
  --motion_file motions/right_kick.npz \
  --num_envs 6000 \
  --headless
```

从 hot-test run 恢复训练也使用 `scripts/rsl_rl/train.py`：

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --wandb_path <your_checkpoint_path> \
  --motion_file motions/right_kick.npz \
  --yaml right_kick/task_params_2.yaml \
  --headless
```

`eval.py` 会把逐 episode 射门指标和汇总精度/速度结果写到 `logs/rsl_rl/eval/`。

## 部署（ONNX 导出）

真实机器人部署需要本仓库导出的 ONNX 策略。导出文件为 `policy-obs.onnx`，并内嵌
关节名称、PD 增益、默认姿态、观测/动作布局以及 motion anchor 等元数据，供
[RoboNaldo_Deploy](https://github.com/OpenDriveLab/RoboNaldo_Deploy/tree/f60f24459aaabc3aea9187a2b13f8923049b629c) 读取。

| 时机 | 输出路径 |
| --- | --- |
| W&B 训练（`--logger wandb`） | 每次保存 `model_*.pt` 时，在同目录生成 `<run_name>.onnx` |
| `play.py` 播放 | `<checkpoint 目录>/exported/policy-obs.onnx` |

对准备部署的 checkpoint 运行一次 `play.py`（`--task`、`--yaml`、`--motion_file`
与训练保持一致）即可生成 ONNX。射门策略请使用 Stage-2 或 Stage-3 的 task preset。
公开文档中请保持 checkpoint 路径和 run ID 为通用占位符；部署侧应使用导出的
ONNX artifact，而不是依赖私有 checkpoint 引用。

## 更多文档

- [Quickstart](docs/quickstart.md)
- [Task Parameters](docs/task_params.md)
- [Rewards](docs/rewards.md)

## 引用

如果 RoboNaldo 对你的研究有帮助，请考虑引用：

```bibtex
@article{robonaldo2026,
  title={RoboNaldo: Accurate, stable, and powerful humanoid soccer shooting via motion-guided curriculum reinforcement learning},
  author={OpenDriveLab},
  journal={arXiv preprint arXiv:2606.11092},
  year={2026},
  url={https://arxiv.org/abs/2606.11092}
}
```

## 致谢

本仓库基于 [Isaac Lab (IsaacLab)](https://github.com/isaac-sim/IsaacLab)、[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) 和 [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 构建。
