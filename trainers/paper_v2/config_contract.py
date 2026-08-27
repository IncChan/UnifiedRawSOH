"""Strict configuration contracts for the independent Paper-v2 entry point."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping


PAPER_VERSION = "Paper-v2"
BOL_LABEL_MODE = "bol_peak_relative"
BOL_REFERENCE_RULE = "bol_peak_mean_top5_first100_v1"
RAW_MODEL_VARIANTS = ("base", "dense_adapter", "residual_moe")
TRAINER_VARIANTS = ("erm", "first_order_mldg")


def _safe_component(field: str, value: Any) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"output.{field} must be one safe path component; got {value!r}")
    return value


def _require_mapping(config: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = config.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"Paper-v2 config requires object field {field!r}.")
    return value


def _require_exact(mapping: Mapping[str, Any], field: str, expected: Any, prefix: str) -> None:
    if mapping.get(field) != expected:
        raise ValueError(
            f"{prefix}.{field} must be exactly {expected!r}; got {mapping.get(field)!r}."
        )


def _require_nonempty_domain_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty list of domain IDs.")
    result = [str(item).strip() for item in value]
    if any(not item for item in result):
        raise ValueError(f"{field} contains an empty domain ID.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicate domains: {result}")
    return result


def is_lodo_config(config: Mapping[str, Any]) -> bool:
    experiment = config.get("experiment", {})
    return str(experiment.get("loader", "")) == "leave_one_domain_out"


def _validate_lodo(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment = _require_mapping(config, "experiment")
    reusability = _require_mapping(config, "reusability")
    source = _require_nonempty_domain_list(experiment.get("source_domain_ids"), "experiment.source_domain_ids")
    target_value = experiment.get("target_domain_id")
    target_values = experiment.get("target_domain_ids")
    if (target_value is None) == (target_values is None):
        raise ValueError(
            "LODO requires exactly one of experiment.target_domain_id or target_domain_ids."
        )
    if target_value is not None:
        target = [str(target_value).strip()]
    else:
        target = _require_nonempty_domain_list(target_values, "experiment.target_domain_ids")
    if len(target) != 1 or not target[0]:
        raise ValueError(f"Paper-v2 zero-cell LODO requires exactly one target domain; got {target}")
    overlap = sorted(set(source) & set(target))
    if overlap:
        raise ValueError(f"LODO source/target domains overlap: {overlap}")
    configured = _require_nonempty_domain_list(experiment.get("domain_ids"), "experiment.domain_ids")
    expected = source + target
    if set(configured) != set(expected) or len(configured) != len(expected):
        raise ValueError(
            "experiment.domain_ids must contain exactly source plus target domains; "
            f"configured={configured}, expected={expected}"
        )
    _require_exact(reusability, "protocol", "leave_one_domain_out", "reusability")
    _require_exact(reusability, "evaluation", "zero_shot", "reusability")
    for field in ("target_train_and_validation_usage", "source_test_usage"):
        _require_exact(reusability, field, "forbidden", "reusability")
    for field, expected_value in (
        ("source_train_split", "train"),
        ("source_validation_split", "val"),
        ("target_test_split", "test"),
    ):
        _require_exact(reusability, field, expected_value, "reusability")
    return {
        "source_domain_ids": source,
        "target_domain_id": target[0],
        "domain_ids": configured,
        "target_train_and_validation_usage": "forbidden",
        "source_test_usage": "forbidden",
    }


def _validate_episode_trainer(config: Mapping[str, Any], trainer: Mapping[str, Any]) -> dict[str, Any]:
    variant = str(trainer.get("variant", "")).strip()
    if variant not in TRAINER_VARIANTS:
        raise ValueError(
            "Paper-v2 trainer.variant must be explicitly set to 'erm' or 'first_order_mldg'; "
            f"got {variant!r}."
        )
    result: dict[str, Any] = {"variant": variant}
    if variant == "first_order_mldg":
        for field in ("inner_steps", "inner_learning_rate", "beta"):
            if field not in trainer:
                raise ValueError(f"trainer.{field} is required for first_order_mldg.")
        if int(trainer["inner_steps"]) != 1:
            raise ValueError("Paper-v2 first_order_mldg currently supports exactly one inner step.")
        if isinstance(trainer["inner_learning_rate"], bool) or float(trainer["inner_learning_rate"]) <= 0:
            raise ValueError("trainer.inner_learning_rate must be positive.")
        if isinstance(trainer["beta"], bool) or float(trainer["beta"]) < 0:
            raise ValueError("trainer.beta must be non-negative.")
        probabilities = {
            "dataset_episode_probability": trainer.get("dataset_episode_probability"),
            "strategy_episode_probability": trainer.get("strategy_episode_probability"),
        }
        if any(value is None for value in probabilities.values()):
            raise ValueError("Both dataset and strategy episode probabilities are required for MLDG.")
        if any(isinstance(value, bool) or float(value) < 0 for value in probabilities.values()):
            raise ValueError("Episode probabilities must be non-negative.")
        if sum(float(value) for value in probabilities.values()) <= 0:
            raise ValueError("At least one episode probability must be positive.")
        result.update(
            {
                "inner_steps": 1,
                "inner_learning_rate": float(trainer["inner_learning_rate"]),
                "beta": float(trainer["beta"]),
                **{key: float(value) for key, value in probabilities.items()},
            }
        )
        if is_lodo_config(config) and "episode_source" in trainer:
            _require_exact(trainer, "episode_source", "source_train_only", "trainer")
    return result


def validate_v2_config(
    config: Mapping[str, Any],
    *,
    require_runnable: bool = False,
) -> dict[str, Any]:
    """Validate a resolved Paper-v2 config and return a compact contract report.

    Validation is intentionally fail-fast.  In particular, an absent model or
    trainer variant is an error; the V2 entry point never silently switches to
    a Base implementation.
    """

    if not isinstance(config, Mapping):
        raise TypeError("Paper-v2 config must be a mapping.")
    if require_runnable and config.get("status") != "runnable":
        raise ValueError(
            f"Paper-v2 training requires status='runnable'; got {config.get('status')!r}."
        )
    output = _require_mapping(config, "output")
    _require_exact(output, "paper_version", PAPER_VERSION, "output")
    identity = {
        field: _safe_component(field, output.get(field))
        for field in ("paper_version", "experiment_id", "model_id", "data_id")
    }
    experiment = _require_mapping(config, "experiment")
    data = _require_mapping(config, "data")
    model = _require_mapping(config, "model")
    train = _require_mapping(config, "train")
    _require_exact(data, "label_mode", BOL_LABEL_MODE, "data")
    _require_exact(data, "bol_reference_rule", BOL_REFERENCE_RULE, "data")
    if "label_field" in data:
        _require_exact(data, "label_field", "soh_bol", "data")
    if "label_mode" in experiment:
        _require_exact(experiment, "label_mode", BOL_LABEL_MODE, "experiment")
    if "label_rule" in experiment:
        _require_exact(experiment, "label_rule", BOL_REFERENCE_RULE, "experiment")
    _require_exact(model, "use_cycle_prediction", False, "model")
    _require_exact(model, "use_predicted_cycle_for_soh", False, "model")
    _require_exact(train, "lambda_cycle", 0.0, "train")
    if "cycle_loss_mode" in train:
        _require_exact(train, "cycle_loss_mode", "disabled", "train")
    for field in ("q_ref_is_model_input", "q_ref_in_normalization"):
        if field in data:
            _require_exact(data, field, False, "data")
    output_root_values = [
        experiment.get("output_root"),
        output.get("output_root"),
    ]
    for value in output_root_values:
        if value is not None and "paper-v1" in str(value).lower():
            raise ValueError(f"Paper-v2 config points at a Paper-v1 output root: {value!r}")

    model_variant = str(model.get("variant", "")).strip()
    model_type = str(model.get("type", "")).strip()
    if not model_variant:
        raise ValueError(
            "model.variant must be explicit; refusing to silently fall back to Base."
        )
    if model_variant in RAW_MODEL_VARIANTS:
        if model_type not in {"PaperRawSOHModel", "PaperV2RawMambaModel"}:
            raise ValueError(
                f"Raw Paper-v2 variant {model_variant!r} requires a raw model type; got {model_type!r}."
            )
        if model_variant == "dense_adapter":
            for field in ("adapter_bottleneck_dim", "adapter_init"):
                if field not in model:
                    raise ValueError(f"model.{field} is required for dense_adapter.")
            if int(model["adapter_bottleneck_dim"]) <= 0:
                raise ValueError("model.adapter_bottleneck_dim must be positive.")
            _require_exact(model, "adapter_init", "zero_output", "model")
    elif model_variant == "feature_mlp":
        if not model_type:
            raise ValueError("FeatureMLP config requires model.type.")
    else:
        raise ValueError(
            f"Unknown Paper-v2 model.variant {model_variant!r}; expected {RAW_MODEL_VARIANTS + ('feature_mlp',)}."
        )
    trainer_report = _validate_episode_trainer(config, _require_mapping(config, "trainer"))
    if model_variant == "residual_moe":
        for field in ("num_experts", "top_k", "expert_bottleneck_dim"):
            if field not in model:
                raise ValueError(f"model.{field} is required for residual_moe.")
        experts = int(model["num_experts"])
        top_k = int(model["top_k"])
        if experts <= 0 or not 1 <= top_k <= experts:
            raise ValueError("model.top_k must satisfy 1 <= top_k <= num_experts.")
        if int(model["expert_bottleneck_dim"]) <= 0:
            raise ValueError("model.expert_bottleneck_dim must be positive.")
        _require_exact(model, "expert_init", "zero_output", "model")
        _require_exact(model, "router_input", "z_base", "model")
    sampler = data.get("sampler", {})
    if sampler and not isinstance(sampler, Mapping):
        raise ValueError("data.sampler must be an object when provided.")
    sampler_kind = str(sampler.get("kind", "sequential"))
    if sampler_kind not in {"sequential", "hierarchical"}:
        raise ValueError("data.sampler.kind must be 'sequential' or 'hierarchical'.")
    lodo_report = _validate_lodo(config) if is_lodo_config(config) else None
    return {
        "valid": True,
        "paper_version": PAPER_VERSION,
        "output": identity,
        "model_variant": model_variant,
        "model_type": model_type,
        "trainer_variant": trainer_report["variant"],
        "sampler_kind": sampler_kind,
        "lodo": lodo_report,
    }


def assert_valid_v2_config(config: Mapping[str, Any], *, require_runnable: bool = False) -> Mapping[str, Any]:
    validate_v2_config(config, require_runnable=require_runnable)
    return config


validate_paper_v2_config = validate_v2_config


def validate_data_readiness(config: Mapping[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Check declared split/data roots before a real child process starts."""

    project_root = Path(project_root)
    data = config["data"]
    paths: list[tuple[str, Path]] = []
    if data.get("data_root"):
        paths.append(("data.data_root", Path(data["data_root"])))
    for domain, value in dict(data.get("data_roots", {})).items():
        paths.append((f"data.data_roots[{domain}]", Path(value)))
    for field, value in (("data.split_file", data.get("split_file")),):
        if value:
            paths.append((field, Path(value)))
    for domain, value in dict(data.get("split_files", {})).items():
        paths.append((f"data.split_files[{domain}]", Path(value)))
    missing = []
    checked = []
    for field, path in paths:
        resolved = path if path.is_absolute() else project_root / path
        expects_file = "split_file" in field
        ready = resolved.is_file() if expects_file else resolved.is_dir()
        checked.append(
            {
                "field": field,
                "path": str(resolved),
                "kind": "file" if expects_file else "directory",
                "exists": bool(ready),
            }
        )
        if not ready:
            missing.append(f"{field}={resolved}")
    if missing:
        raise FileNotFoundError("Paper-v2 readiness check failed: " + "; ".join(missing))
    return {"checked": checked, "ready": True}


def v2_output_identity(config: Mapping[str, Any]) -> dict[str, str]:
    output = config["output"]
    return {
        field: _safe_component(field, output[field])
        for field in ("paper_version", "experiment_id", "model_id", "data_id")
    }


def build_v2_seed_output_dir(
    output_root: str | Path,
    config: Mapping[str, Any],
    run_time: str,
    seed: int,
) -> Path:
    identity = v2_output_identity(config)
    if identity["paper_version"] != PAPER_VERSION:
        raise ValueError("V2 output builder refuses a non-Paper-v2 config.")
    root = Path(output_root)
    if "paper-v1" in str(root).lower():
        raise ValueError(f"V2 output root cannot contain Paper-v1: {root}")
    runtime = str(run_time).strip()
    if not runtime:
        raise ValueError("run_time must not be empty.")
    if not runtime.startswith("runtime_"):
        runtime = "runtime_" + runtime
    runtime = _safe_component("run_time", runtime)
    return root.joinpath(
        identity["paper_version"],
        identity["experiment_id"],
        identity["model_id"],
        identity["data_id"],
        runtime,
        f"seed_{int(seed)}",
    )


__all__ = [
    "BOL_LABEL_MODE",
    "BOL_REFERENCE_RULE",
    "PAPER_VERSION",
    "RAW_MODEL_VARIANTS",
    "TRAINER_VARIANTS",
    "assert_valid_v2_config",
    "build_v2_seed_output_dir",
    "is_lodo_config",
    "validate_data_readiness",
    "validate_paper_v2_config",
    "validate_v2_config",
    "v2_output_identity",
]
