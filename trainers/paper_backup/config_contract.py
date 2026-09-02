"""Validation rules shared by all Paper-Backup launch paths."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from ...models.paper_backup.model_factory import SUPPORTED_MODEL_TYPES
from ...datasets.paper_backup.sequence_views import ALL_VIEW_IDS
from ...datasets.splits import load_split_spec


PAPER_VERSION = "Paper-Backup"
EXPERIMENT_IDS = {
    "e1_main_estimation",
    "e1_shared_crate_fullvi",
    "e1_shared_crate_128x128",
    "e2_charging_information",
    "e2_final_256budget",
    "e1_final_interaction_5seed",
    "e1_bicontext_5seed",
    "e1_bicontext_adaptive_fusion_5seed",
    "e1_bicontext_cycle_mtl_5seed",
    "e2_final_interaction_5seed",
    "e3_strategy_pooling",
}
E1_MODEL_TYPES = {"HI-MLP", "Transformer", "Ours"}
E2_MODEL_TYPES = {"VanillaMamba", "SingleStreamMamba", "Ours"}
E3_MODEL_TYPES = {"Ours"}
FINAL_E1_MODEL_TYPES = {
    "FinalHI-MLP",
    "FinalRawCNN",
    "FinalRawLSTM",
    "FinalRawTransformer",
    "FinalRawVanillaMamba",
    "FinalRawCCVanillaMamba",
    "FinalRawCVVanillaMamba",
    "FinalRawDualVanillaMamba",
    "FinalInteractionMamba",
    "FinalBiContextMamba",
    "FinalBiContextAdaptiveFusion",
    "FinalBiContextCycleMTL",
}
FINAL_E2_MODEL_TYPES = {
    "FinalRawVanillaMamba",
    "FinalRawDualVanillaMamba",
    "FinalInteractionMamba",
}


def _walk(value: Any, path: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield child_path, str(key), child
            yield from _walk(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _is_disabled(value: Any) -> bool:
    """Return whether a contract value is an explicit disabled sentinel."""

    if isinstance(value, (list, tuple, dict, set)):
        return False
    return value in {False, 0, 0.0, "", "disabled", "none", "not_used"}


def _safe_root(value: Any) -> str:
    root = str(value or "").strip()
    if not root:
        raise ValueError("Paper-Backup requires output.root")
    parts = Path(root).parts
    if "Paper-Backup" not in parts and "Paper-Backup" not in root.replace("\\", "/").split("/"):
        raise ValueError("Paper-Backup output.root must be inside an explicit Paper-Backup namespace")
    return root


def _validate_static_split(path: Path) -> dict[str, Any]:
    split = load_split_spec(path)
    if "roles" in split:
        roles = split["roles"]
        values = [str(item) for role in ("train", "val", "test") for item in roles.get(role, [])]
        if len(values) != len(set(values)):
            raise ValueError(f"Split file has overlapping fixed battery roles: {path}")
    test_by_condition = split.get("test_batteries_by_condition")
    if test_by_condition:
        all_values = [str(item) for values in test_by_condition.values() for item in values]
        if len(all_values) != len(set(all_values)):
            raise ValueError(f"Split file assigns a test battery to multiple strategies: {path}")
    test_values = split.get("test_batteries", [])
    if not test_values and test_by_condition:
        test_values = [item for values in test_by_condition.values() for item in values]
    return {
        "path": str(path),
        "test_battery_count": len(set(str(item) for item in test_values)),
        "has_condition_assignments": bool(test_by_condition),
        "development_protocol": split.get("development_split", split.get("protocol", {})),
    }


def validate_config(
    config: Mapping[str, Any],
    repo_root: str | Path | None = None,
    *,
    check_files: bool = False,
) -> dict[str, Any]:
    """Validate a resolved config and return a compact contract audit."""

    output = config.get("output", {})
    if str(output.get("paper_version", "")) != PAPER_VERSION:
        raise ValueError("Paper-Backup config must set output.paper_version='Paper-Backup'")
    experiment_id = str(output.get("experiment_id", ""))
    if experiment_id not in EXPERIMENT_IDS:
        raise ValueError(f"Unsupported Paper-Backup output.experiment_id: {experiment_id!r}")
    model = config.get("model", {})
    model_type = str(model.get("type", ""))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unknown Paper-Backup model.type: {model_type!r}")
    allowed = {
        "e1_main_estimation": E1_MODEL_TYPES,
        "e1_shared_crate_fullvi": {"Transformer", "Ours"},
        "e1_shared_crate_128x128": E1_MODEL_TYPES,
        "e2_charging_information": E2_MODEL_TYPES,
        "e2_final_256budget": {"VanillaMamba", "Ours"},
        "e1_final_interaction_5seed": FINAL_E1_MODEL_TYPES,
        "e1_bicontext_5seed": {"FinalBiContextMamba"},
        "e1_bicontext_adaptive_fusion_5seed": {"FinalBiContextAdaptiveFusion"},
        "e1_bicontext_cycle_mtl_5seed": {"FinalBiContextCycleMTL"},
        "e2_final_interaction_5seed": FINAL_E2_MODEL_TYPES,
        "e3_strategy_pooling": E3_MODEL_TYPES,
    }[experiment_id]
    if model_type not in allowed:
        raise ValueError(f"Model {model_type!r} is not in {experiment_id} matrix: {sorted(allowed)}")
    _safe_root(output.get("root", ""))
    view_id = str(config.get("data", {}).get("input_view", config.get("experiment", {}).get("input_view", "")))
    if experiment_id in {
        "e1_main_estimation",
        "e1_shared_crate_fullvi",
        "e1_shared_crate_128x128",
        "e1_final_interaction_5seed",
        "e1_bicontext_5seed",
        "e1_bicontext_adaptive_fusion_5seed",
        "e1_bicontext_cycle_mtl_5seed",
    } and model_type not in {"HI-MLP", "FinalHI-MLP"} and view_id not in TERMINAL_VIEWS:
        raise ValueError(f"E1 raw model requires a terminal input view, got {view_id!r}")
    if experiment_id in {"e2_charging_information", "e2_final_256budget", "e2_final_interaction_5seed"} and view_id not in {"full_cccv", "full_joint", "terminal_joint", "terminal_cc", "terminal_cv", "terminal_phase"}:
        raise ValueError(f"E2 config has invalid input view: {view_id!r}")
    if experiment_id == "e3_strategy_pooling" and (view_id != "terminal_phase" or model_type != "Ours"):
        raise ValueError("E3 requires Ours with input_view='terminal_phase'")
    if model_type not in {"HI-MLP", "FinalHI-MLP"} and view_id not in ALL_VIEW_IDS:
        raise ValueError(f"Raw Paper-Backup model has invalid input view: {view_id!r}")

    train = config.get("train", {})
    cycle_mtl = (
        experiment_id == "e1_bicontext_cycle_mtl_5seed"
        and model_type == "FinalBiContextCycleMTL"
    )
    if float(train.get("lambda_cycle", 0.0)) != 0.0:
        raise ValueError(
            "The historical lifetime-coordinate loss remains disabled: "
            "train.lambda_cycle must be 0"
        )
    if cycle_mtl:
        lambda_cycle_aux = float(train.get("lambda_cycle_aux", 0.0))
        if not math.isfinite(lambda_cycle_aux) or lambda_cycle_aux <= 0.0:
            raise ValueError("Cycle MTL requires a positive finite lambda_cycle_aux")
        if str(train.get("cycle_loss_mode", "")) != "auxiliary_only_mse":
            raise ValueError("Cycle MTL requires cycle_loss_mode='auxiliary_only_mse'")
        if int(train.get("cycle_aux_warmup_epochs", -1)) < 0:
            raise ValueError("Cycle MTL warm-up epochs must be non-negative")
        if int(model.get("cycle_head_hidden_dim", -1)) < 0:
            raise ValueError("Cycle MTL requires cycle_head_hidden_dim >= 0")
        if model.get("use_predicted_cycle_for_soh", False) not in {False, 0}:
            raise ValueError("Cycle MTL must not inject predicted cycle into the SOH head")
    else:
        if float(train.get("lambda_cycle_aux", 0.0)) != 0.0:
            raise ValueError("lambda_cycle_aux is reserved for the isolated Cycle MTL suite")
        if str(train.get("cycle_loss_mode", "disabled")) not in {"disabled", "none"}:
            raise ValueError("Paper-Backup SOH-only suites require cycle_loss_mode=disabled")
    if model_type == "Ours":
        if model.get("use_cycle_prediction") is not False:
            raise ValueError("Paper-Backup Ours requires model.use_cycle_prediction=false")
        if model.get("use_predicted_cycle_for_soh") is not False:
            raise ValueError("Paper-Backup Ours requires model.use_predicted_cycle_for_soh=false")
    for path, key, value in _walk(model):
        key_lower = key.lower()
        if any(token in key_lower for token in ("cycle", "lifetime", "eol", "rul")):
            if cycle_mtl and key_lower == "cycle_head_hidden_dim":
                continue
            if key_lower not in {"cycle_loss_mode"} and not _is_disabled(value):
                raise ValueError(f"Forbidden lifetime/cycle model option at {path}: {value!r}")
        if any(token in key_lower for token in ("strategy_id", "domain_id", "battery_id", "cell_id")) and not _is_disabled(value) and value is not None:
            raise ValueError(f"Metadata ID cannot be a model input at {path}")
    input_names = [str(item).lower() for item in model.get("input_features", [])]
    if any("strategy" in item or "domain" in item or "battery" in item or "cell" in item for item in input_names):
        raise ValueError("Strategy/domain/battery metadata cannot be listed in model.input_features")
    if experiment_id == "e3_strategy_pooling" and "strategy_id" in json.dumps(model, sort_keys=True).lower():
        raise ValueError("E3 pooled/specific model config cannot inject strategy_id")

    data = config.get("data", {})
    if cycle_mtl:
        if str(data.get("cycle_aux_target_mode", "")) != "log1p_rank_train_max":
            raise ValueError(
                "Cycle MTL requires cycle_aux_target_mode='log1p_rank_train_max'"
            )
    elif str(data.get("cycle_aux_target_mode", "disabled")) != "disabled":
        raise ValueError("Cycle auxiliary targets are isolated from the SOH-only suites")
    source_mode = str(data.get("source_mode", "legacy_runtime"))
    if source_mode not in {"preprocessed_v1", "preprocessed_v2", "legacy_runtime"}:
        raise ValueError("Paper-Backup data.source_mode must be preprocessed_v1, preprocessed_v2 or legacy_runtime")
    is_preprocessed = source_mode in {"preprocessed_v1", "preprocessed_v2"}
    if is_preprocessed and not str(data.get("preprocessed_data_root", "")).strip():
        raise ValueError(f"{source_mode} requires data.preprocessed_data_root")
    if source_mode == "preprocessed_v2":
        if int(data.get("preprocessed_schema_version", 0)) != 2:
            raise ValueError("preprocessed_v2 requires preprocessed_schema_version=2")
        if str(config.get("normalization", {}).get("current_mode", "nominal_c_rate")) != "nominal_c_rate":
            raise ValueError("preprocessed_v2 requires nominal C-rate current normalization")
    if experiment_id in {
        "e1_shared_crate_fullvi",
        "e1_shared_crate_128x128",
    }:
        if source_mode != "preprocessed_v2":
            raise ValueError("The isolated C-rate E1 suite requires preprocessed_v2")
        phase_mode = str(data.get("phase_signal_mode", ""))
        if model_type == "Ours" and phase_mode not in {
            "shared_dominant",
            "shared_full_vi",
            "shared_gated_full_vi",
        }:
            raise ValueError("C-rate Ours requires an explicit shared phase_signal_mode")
        gated = str(model.get("phase_input_fusion", "standard")) == "gated_residual_full_vi"
        if (phase_mode == "shared_gated_full_vi") != gated:
            raise ValueError(
                "shared_gated_full_vi data and gated_residual_full_vi model fusion "
                "must be enabled together"
            )
        if gated:
            if int(model.get("signal_input_dim", 0)) != 3 or int(model.get("input_dim", 0)) != 5:
                raise ValueError("Gated FullVI requires signal_input_dim=3 and input_dim=5")
            if int(model.get("gate_hidden_dim", 0)) < 1:
                raise ValueError("Gated FullVI requires a positive gate_hidden_dim")
            if str(model.get("gate_context", "")) != "masked_mean":
                raise ValueError("Gated FullVI requires gate_context='masked_mean'")
            if str(model.get("secondary_residual_init", "")) != "zero":
                raise ValueError("Gated FullVI requires secondary_residual_init='zero'")
        bridge_type = str(model.get("cc_to_cv_bridge_type", "zero_init_linear"))
        if bridge_type not in {"zero_init_linear", "adaptive_pointwise_zero_init"}:
            raise ValueError(f"Unsupported C-rate CC-to-CV bridge type: {bridge_type!r}")
        if bridge_type == "adaptive_pointwise_zero_init" and phase_mode != "shared_full_vi":
            raise ValueError("Adaptive pointwise CC-to-CV bridge requires shared_full_vi input")
    if experiment_id == "e1_shared_crate_128x128":
        if int(data.get("raw_len_cc", 0)) != 128 or int(data.get("raw_len_cv", 0)) != 128:
            raise ValueError("The isolated 128x128 E1 suite requires raw_len_cc=raw_len_cv=128")
    if experiment_id in {
        "e1_final_interaction_5seed",
        "e1_bicontext_5seed",
        "e1_bicontext_adaptive_fusion_5seed",
        "e1_bicontext_cycle_mtl_5seed",
    }:
        if source_mode != "preprocessed_v2":
            raise ValueError("Final interaction E1 requires schema-v2 offline preprocessing")
        if int(data.get("raw_len_cc", 0)) != 128 or int(data.get("raw_len_cv", 0)) != 128:
            raise ValueError("Final interaction E1 requires 128+128 terminal samples")
        if int(config.get("train", {}).get("epochs", 0)) != 600:
            raise ValueError("Final interaction E1 requires train.epochs=600")
        if int(config.get("train", {}).get("patience", 0)) != 30:
            raise ValueError("Final interaction E1 requires train.patience=30")
        if str(data.get("cohort", "")) != "all":
            raise ValueError("Final interaction E1 requires the complete terminal cohort")
        if str(data.get("sample_filter_mode", "")) != "none":
            raise ValueError(
                "Final interaction E1 requires data.sample_filter_mode='none' "
                "for a shared unfiltered raw/feature cohort"
            )
        if model_type in {
            "FinalRawDualVanillaMamba",
            "FinalInteractionMamba",
            "FinalBiContextMamba",
            "FinalBiContextAdaptiveFusion",
            "FinalBiContextCycleMTL",
        }:
            if view_id != "terminal_phase" or str(data.get("phase_signal_mode", "")) != "shared_full_vi":
                raise ValueError("Final dual/interaction E1 models require shared_full_vi terminal_phase input")
        if model_type == "FinalRawCCVanillaMamba" and view_id != "terminal_cc":
            raise ValueError("Final raw CC Vanilla Mamba requires terminal_cc")
        if model_type == "FinalRawCVVanillaMamba" and view_id != "terminal_cv":
            raise ValueError("Final raw CV Vanilla Mamba requires terminal_cv")
    if is_preprocessed and experiment_id in {
        "e2_charging_information",
        "e2_final_256budget",
        "e2_final_interaction_5seed",
    }:
        if str(data.get("cohort", "full_matched")) != "full_matched":
            raise ValueError("Preprocessed E2 views must use the full_matched cohort")
    if experiment_id == "e2_final_256budget":
        if source_mode != "preprocessed_v2":
            raise ValueError("Final E2 requires schema-v2 offline preprocessing")
        if int(data.get("raw_len_cc", 0)) != 128 or int(data.get("raw_len_cv", 0)) != 128:
            raise ValueError("Final E2 terminal phases require 128+128 physical points")
        if str(config.get("normalization", {}).get("current_mode", "")) != "nominal_c_rate":
            raise ValueError("Final E2 requires nominal C-rate current normalization")
        active_phase = str(data.get("active_phase", "both"))
        if active_phase not in {"both", "cc", "cv"}:
            raise ValueError("Final E2 active_phase must be both, cc or cv")
        if model_type == "VanillaMamba":
            if view_id not in {"full_joint", "terminal_joint"}:
                raise ValueError("Final E2 Vanilla Mamba requires a joint input view")
            if not bool(model.get("use_boundary_token", False)):
                raise ValueError("Final E2 Vanilla Mamba requires the shared CC-CV boundary token")
            if active_phase != "both":
                raise ValueError("Final E2 Vanilla Mamba must observe both phases")
            if view_id == "full_joint" and int(data.get("full_joint_len", 0)) != 256:
                raise ValueError("Final E2 FULL reference requires 256 physical samples")
        if model_type == "Ours":
            if view_id != "terminal_phase":
                raise ValueError("Final E2 Ours variants require terminal_phase input")
            if str(data.get("phase_signal_mode", "")) != "shared_full_vi":
                raise ValueError("Final E2 Ours variants require shared_full_vi")
            if str(model.get("active_phase", "both")) != active_phase:
                raise ValueError("Final E2 model/data active_phase must match")
            if not bool(model.get("use_cc_to_cv_bridge", False)):
                raise ValueError(
                    "Final E2 Ours input ablations keep the complete PointBridge architecture"
                )
    if experiment_id == "e2_final_interaction_5seed":
        if source_mode != "preprocessed_v2":
            raise ValueError("Final interaction E2 requires schema-v2 offline preprocessing")
        if int(data.get("raw_len_cc", 0)) != 128 or int(data.get("raw_len_cv", 0)) != 128:
            raise ValueError("Final interaction E2 requires 128+128 terminal samples")
        if int(config.get("train", {}).get("epochs", 0)) != 600 or int(config.get("train", {}).get("patience", 0)) != 30:
            raise ValueError("Final interaction E2 requires epochs=600 and patience=30")
        if model_type == "FinalRawVanillaMamba":
            if view_id != "full_joint" or int(data.get("full_joint_len", 0)) != 256:
                raise ValueError("Final interaction E2 FULL Vanilla requires full_joint length 256")
            if not bool(model.get("use_boundary_token", False)):
                raise ValueError("Final interaction E2 FULL Vanilla requires a boundary token")
        else:
            if view_id != "terminal_phase" or str(data.get("phase_signal_mode", "")) != "shared_full_vi":
                raise ValueError("Final interaction E2 terminal models require shared_full_vi terminal_phase")
    full_audit = None
    if view_id in {"full_cccv", "full_joint"}:
        if not str(data.get("full_source_kind", "")).strip():
            raise ValueError("full_cccv config must declare full_source_kind")
        if not str(data.get("full_source_format", "")).strip():
            raise ValueError("full_cccv config must declare full_source_format")
        if str(data.get("full_source_kind", "")) in {"canonical_terminal", "terminal_raw", "terminal_only"}:
            raise ValueError("full_cccv cannot use a terminal-only source")
        if is_preprocessed:
            if str(data.get("full_source_kind")) != "preprocessed_full_cccv":
                raise ValueError("preprocessed full_cccv requires full_source_kind=preprocessed_full_cccv")
            expected_format = (
                "paper_backup_npy_v2_joint"
                if view_id == "full_joint"
                else "paper_backup_npy_v1"
            )
            if str(data.get("full_source_format")) != expected_format:
                raise ValueError(
                    f"preprocessed {view_id} requires full_source_format={expected_format}"
                )
        else:
            if "full_data_root" not in data:
                raise ValueError("legacy full_cccv must declare full_data_root")
            if data.get("full_data_root") and data.get("terminal_data_root") and str(data["full_data_root"]) == str(data["terminal_data_root"]):
                raise ValueError("full_data_root and terminal_data_root cannot be the same terminal product")
    if data.get("matched_full_data_root") and data.get("terminal_data_root") and str(data["matched_full_data_root"]) == str(data["terminal_data_root"]):
        raise ValueError("matched_full_data_root cannot be the terminal product")

    split_audit = None
    if repo_root is not None:
        root = Path(repo_root).resolve()
        split_value = data.get("split_file") or config.get("experiment", {}).get("split_file")
        if split_value:
            split_path = Path(split_value)
            if not split_path.is_absolute():
                split_path = root / split_path
            if check_files:
                if not split_path.is_file():
                    raise ValueError(f"Configured split file does not exist: {split_path}")
                split_audit = _validate_static_split(split_path)
        if check_files and is_preprocessed:
            product_root = Path(str(data["preprocessed_data_root"]))
            if not product_root.is_absolute():
                product_root = root / product_root
            domain_directory = product_root / str(data.get("domain_id", config.get("experiment", {}).get("domain_id", "")))
            manifest = domain_directory / "manifest.json"
            # Config-only validation remains usable before preprocessing; an
            # existing product, however, must have the expected manifest.
            if domain_directory.exists() and not manifest.is_file():
                raise ValueError(f"Paper-Backup preprocessed manifest is missing: {manifest}")
    return {
        "paper_version": PAPER_VERSION,
        "experiment_id": experiment_id,
        "model_type": model_type,
        "input_view": view_id,
        "output_root": str(output["root"]),
        "split": split_audit,
        "soh_only": True,
        "metadata_in_forward": False,
        "full_source_declared": view_id == "full_cccv",
        "source_mode": source_mode,
    }


TERMINAL_VIEWS = {"terminal_joint", "terminal_cc", "terminal_cv", "terminal_phase"}


__all__ = [
    "E1_MODEL_TYPES",
    "E2_MODEL_TYPES",
    "E3_MODEL_TYPES",
    "EXPERIMENT_IDS",
    "PAPER_VERSION",
    "TERMINAL_VIEWS",
    "validate_config",
]
