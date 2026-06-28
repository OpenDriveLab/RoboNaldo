from __future__ import annotations

import inspect

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.task_params import (
    is_number,
    select_stage_value,
    select_weight_stage_value,
)


def term_params(term, context: str) -> dict:
    params = getattr(term, "params", None)
    if params is None:
        term.params = {}
        return term.params
    if not isinstance(params, dict):
        raise TypeError(f"{context}.params must be a mapping, got {type(params).__name__}.")
    return params


def term_accepts_param(term, param_name: str) -> bool:
    params = getattr(term, "params", None)
    if isinstance(params, dict) and param_name in params:
        return True
    func = getattr(term, "func", None)
    if func is None:
        return False
    signature = inspect.signature(func)
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return param_name in signature.parameters


def set_term_param(term, param_name: str, param_value, context: str) -> None:
    if not term_accepts_param(term, param_name):
        raise KeyError(f"{context} targets unknown parameter '{param_name}'.")
    term_params(term, context)[param_name] = param_value


def apply_reward_overrides(env_cfg, params: dict, training_stage: str, category_weights: dict[str, float]) -> None:
    """Apply grouped reward overrides from YAML."""
    reward_overrides = params["reward_overrides"] if "reward_overrides" in params else {}
    if not isinstance(reward_overrides, dict):
        raise TypeError("reward_overrides must be a mapping when provided.")

    for category_name, terms in reward_overrides.items():
        if not isinstance(terms, dict):
            raise TypeError(f"reward_overrides.{category_name} must be a mapping of reward terms.")
        if category_name not in category_weights:
            raise KeyError(f"Unknown reward override category '{category_name}'.")
        category_weight = category_weights[category_name]
        for term_name, term_overrides in terms.items():
            reward_term = getattr(env_cfg.rewards, term_name, None)
            if reward_term is None:
                raise AttributeError(f"reward_overrides.{category_name}.{term_name} is not a configured reward term.")

            if is_number(term_overrides):
                reward_term.weight = float(term_overrides)
                continue
            if not isinstance(term_overrides, dict):
                raise TypeError(f"reward_overrides.{category_name}.{term_name} must be a number or mapping.")

            allowed_override_keys = {"weight", "weight_scale", "params"}
            unknown_override_keys = set(term_overrides) - allowed_override_keys
            if unknown_override_keys:
                raise KeyError(
                    f"reward_overrides.{category_name}.{term_name} has unsupported keys: "
                    f"{sorted(unknown_override_keys)}."
                )
            if "weight" in term_overrides and "weight_scale" in term_overrides:
                raise ValueError(
                    f"reward_overrides.{category_name}.{term_name} cannot set both 'weight' and 'weight_scale'."
                )
            if not any(key in term_overrides for key in allowed_override_keys):
                raise ValueError(f"reward_overrides.{category_name}.{term_name} must set weight, weight_scale, or params.")

            if "weight" in term_overrides:
                reward_term.weight = float(
                    select_weight_stage_value(
                        term_overrides["weight"],
                        training_stage,
                        f"reward_overrides.{category_name}.{term_name}.weight",
                    )
                )
            elif "weight_scale" in term_overrides:
                reward_term.weight = (
                    float(
                        select_weight_stage_value(
                            term_overrides["weight_scale"],
                            training_stage,
                            f"reward_overrides.{category_name}.{term_name}.weight_scale",
                        )
                    )
                    * category_weight
                )

            term_params_overrides = term_overrides["params"] if "params" in term_overrides else {}
            if isinstance(term_params_overrides, dict):
                for param_name, param_value in term_params_overrides.items():
                    set_term_param(
                        reward_term,
                        param_name,
                        select_stage_value(
                            param_value,
                            training_stage,
                            f"reward_overrides.{category_name}.{term_name}.params.{param_name}",
                        ),
                        f"reward_overrides.{category_name}.{term_name}.params",
                    )
            elif term_params_overrides is not None:
                raise TypeError(f"reward_overrides.{category_name}.{term_name}.params must be a mapping.")


def make_self_collision_termination() -> DoneTerm:
    return DoneTerm(
        func=mdp.self_collision,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "body_pairs": [
                ("left_ankle_roll_link", "right_ankle_roll_link", 0.06),
            ],
        },
    )


def apply_termination_overrides(env_cfg, params: dict) -> None:
    """Apply termination overrides from YAML."""
    if not hasattr(env_cfg, "terminations"):
        return

    termination_overrides = params["termination_overrides"] if "termination_overrides" in params else {}
    if not isinstance(termination_overrides, dict):
        raise TypeError("termination_overrides must be a mapping when provided.")

    for term_name, term_overrides in termination_overrides.items():
        if term_name in {"default", "tracking", "task"}:
            raise KeyError(
                f"termination_overrides.{term_name} is no longer supported. "
                "Put termination terms directly under termination_overrides."
            )
        termination_term = getattr(env_cfg.terminations, term_name, None)
        if term_name == "self_collision":
            if not isinstance(term_overrides, bool):
                raise TypeError("termination_overrides.self_collision must be true or false.")
            env_cfg.terminations.self_collision = make_self_collision_termination() if bool(term_overrides) else None
            continue
        if termination_term is None:
            raise AttributeError(f"termination_overrides.{term_name} is not a configured termination term.")
        if isinstance(term_overrides, bool) and not term_overrides:
            setattr(env_cfg.terminations, term_name, None)
            continue

        if is_number(term_overrides):
            set_term_param(
                termination_term,
                "threshold",
                float(term_overrides),
                f"termination_overrides.{term_name}",
            )
            if term_name == "ee_body_pos" and hasattr(env_cfg, "rewards"):
                reward_term = getattr(env_cfg.rewards, "ee_body_pos_termination_penalty", None)
                if reward_term is None:
                    raise AttributeError(
                        "termination_overrides.ee_body_pos also targets missing reward term "
                        "'ee_body_pos_termination_penalty'."
                    )
                set_term_param(
                    reward_term,
                    "threshold",
                    float(term_overrides),
                    f"termination_overrides.{term_name}",
                )
            continue
        if not isinstance(term_overrides, dict):
            raise TypeError(f"termination_overrides.{term_name} must be a number, false, or mapping.")
        for param_name, param_value in term_overrides.items():
            set_term_param(
                termination_term,
                param_name,
                param_value,
                f"termination_overrides.{term_name}",
            )
            if term_name == "ee_body_pos" and hasattr(env_cfg, "rewards"):
                reward_term = getattr(env_cfg.rewards, "ee_body_pos_termination_penalty", None)
                if reward_term is None:
                    raise AttributeError(
                        "termination_overrides.ee_body_pos also targets missing reward term "
                        "'ee_body_pos_termination_penalty'."
                    )
                set_term_param(
                    reward_term,
                    param_name,
                    param_value,
                    f"termination_overrides.{term_name}",
                )
