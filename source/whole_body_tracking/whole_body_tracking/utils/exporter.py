# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import copy
from typing import Any
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy-obs.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


def _infer_input_size(actor: torch.nn.Module) -> int | None:
    if hasattr(actor, "input_size"):
        return int(getattr(actor, "input_size"))
    for module in actor.modules():
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)
    return None


def _module_device(module: torch.nn.Module) -> torch.device:
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cpu")


def resolve_policy_module(policy_like: object) -> torch.nn.Module:
    """Return the policy module used for deterministic inference/export."""
    if hasattr(policy_like, "get_policy") and callable(policy_like.get_policy):
        return resolve_policy_module(policy_like.get_policy())
    for attr_name in ("policy", "actor"):
        module = getattr(policy_like, attr_name, None)
        if isinstance(module, torch.nn.Module):
            return module
    if isinstance(policy_like, torch.nn.Module):
        return policy_like
    raise TypeError(f"Could not resolve policy module from object of type {type(policy_like).__name__}.")


def resolve_policy_normalizer(policy_owner: object) -> object | None:
    """Return a separate normalizer only when the policy module does not already own one."""
    policy = resolve_policy_module(policy_owner)
    if hasattr(policy, "obs_normalizer"):
        return None

    for holder in (policy_owner, policy):
        for attr_name in ("normalizer", "actor_obs_normalizer", "obs_normalizer"):
            normalizer = getattr(holder, attr_name, None)
            if normalizer is not None and not isinstance(normalizer, torch.nn.Identity):
                return normalizer
    return None


def _get_motion_tensor(source: object, name: str) -> torch.Tensor:
    if hasattr(source, name):
        value = getattr(source, name)
    else:
        private_name = f"_{name}"
        if not hasattr(source, private_name):
            raise AttributeError(f"{type(source).__name__} object has no motion tensor '{name}'.")
        value = getattr(source, private_name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected motion tensor '{name}' to be a torch.Tensor, got {type(value).__name__}.")
    return value.to("cpu")


def _motion_reference_source(cmd: Any) -> object:
    motion = getattr(cmd, "motion", None)
    if motion is not None and hasattr(motion, "joint_pos"):
        return motion
    if motion is None:
        raise AttributeError(f"{type(cmd).__name__} object has no 'motion' reference source.")
    return motion


class _OnnxMotionPolicyExporter(torch.nn.Module):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.normalizer = torch.nn.Identity()

        if hasattr(actor_critic, "as_onnx"):
            self.actor = actor_critic.as_onnx(verbose)
            self._actor_has_internal_normalizer = True
        elif hasattr(actor_critic, "input_size"):
            self.actor = copy.deepcopy(actor_critic)
            self._actor_has_internal_normalizer = True
        elif hasattr(actor_critic, "actor"):
            self.actor = copy.deepcopy(actor_critic.actor)
            self._actor_has_internal_normalizer = False
        else:
            self.actor = copy.deepcopy(actor_critic)
            self._actor_has_internal_normalizer = hasattr(self.actor, "obs_normalizer")

        if normalizer is not None and not self._actor_has_internal_normalizer:
            self.normalizer = copy.deepcopy(normalizer)

        self.obs_dim = _infer_input_size(self.actor)
        if self.obs_dim is None:
            raise TypeError(
                f"Unable to infer ONNX observation size for exporter from actor type {type(self.actor).__name__}."
            )

        cmd: Any = env.command_manager.get_term("motion")
        motion_source = _motion_reference_source(cmd)

        self.joint_pos = _get_motion_tensor(motion_source, "joint_pos")
        self.joint_vel = _get_motion_tensor(motion_source, "joint_vel")
        self.body_pos_w = _get_motion_tensor(motion_source, "body_pos_w")
        self.body_quat_w = _get_motion_tensor(motion_source, "body_quat_w")
        self.body_lin_vel_w = _get_motion_tensor(motion_source, "body_lin_vel_w")
        self.body_ang_vel_w = _get_motion_tensor(motion_source, "body_ang_vel_w")
        if hasattr(motion_source, "lengths"):
            self.time_step_total = motion_source.lengths.to("cpu")
        else:
            self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x):
        if self._actor_has_internal_normalizer:
            return (self.actor(x),)
        return (
            self.actor(self.normalizer(x)),
        )

    def export(self, path, filename):
        actor_device = _module_device(self.actor)
        self.to("cpu")
        self.eval()
        obs = torch.zeros(1, self.obs_dim)
        try:
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=18,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
        finally:
            self.actor.to(actor_device)


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers -> format, strings -> as-is
    )


def _resolve_default_joint_pos(robot_data: Any) -> list[float]:
    """Resolve default joint positions from IsaacLab robot data."""
    if hasattr(robot_data, "default_joint_pos_nominal"):
        nominal = getattr(robot_data, "default_joint_pos_nominal")
        if isinstance(nominal, torch.Tensor):
            if nominal.ndim == 1:
                return nominal.detach().cpu().tolist()
            if nominal.ndim >= 2:
                return nominal[0].detach().cpu().tolist()

    if hasattr(robot_data, "default_joint_pos"):
        default_pos = getattr(robot_data, "default_joint_pos")
        if isinstance(default_pos, torch.Tensor):
            if default_pos.ndim == 1:
                return default_pos.detach().cpu().tolist()
            if default_pos.ndim >= 2:
                return default_pos[0].detach().cpu().tolist()

    raise AttributeError("Robot data does not expose default_joint_pos_nominal or default_joint_pos.")


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy-obs.onnx") -> None:
    onnx_path = os.path.join(path, filename)
    robot_data = env.scene["robot"].data
    metadata = {
        "run_path": run_path,
        "joint_names": robot_data.joint_names,
        "joint_stiffness": robot_data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": robot_data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": _resolve_default_joint_pos(robot_data),
        "command_names": env.command_manager.active_terms,
        "observation_names": env.observation_manager.active_terms["policy"],
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
