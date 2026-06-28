from collections.abc import Sequence
import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

import wandb

from whole_body_tracking.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    resolve_policy_module,
    resolve_policy_normalizer,
)


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name: str | Sequence[str] | None = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model, export the policy, and bind the motion artifact for WandB runs."""
        super().save(path, infos)
        if self.logger.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            policy = resolve_policy_module(self.alg)
            normalizer = resolve_policy_normalizer(self.alg)
            export_motion_policy_as_onnx(
                self.env.unwrapped, policy, normalizer=normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            if self.registry_name is not None:
                registry_names = [self.registry_name] if isinstance(self.registry_name, str) else list(self.registry_name)
                for registry_name in registry_names:
                    wandb.run.use_artifact(registry_name)
                self.registry_name = None
