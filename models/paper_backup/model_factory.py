"""Explicit model factory for Paper-Backup.

Unknown model identifiers are errors.  Keeping the dispatch table here makes
it possible for tests and manifests to prove that two benchmark labels do not
silently instantiate the same model.
"""

from __future__ import annotations

from typing import Any

from ..baselines.pinn4soh_no_leak_onlyf import PINNFOnlyMLP
from .sequence_baselines import (
    LSTMSOHModel,
    PhaseMambaSOHOnly,
    RawCNNSOHModel,
    SingleStreamMambaSOHModel,
    TransformerSOHModel,
    VanillaMambaSOHModel,
)
from .final_models.model_factory import FINAL_MODEL_TYPES, build_final_model


SUPPORTED_MODEL_TYPES = (
    "HI-MLP",
    "RawCNN",
    "LSTM",
    "Transformer",
    "VanillaMamba",
    "SingleStreamMamba",
    "Ours",
    *FINAL_MODEL_TYPES.keys(),
)


def _without_type(config: dict[str, Any]) -> dict[str, Any]:
    values = dict(config)
    values.pop("type", None)
    values.pop("model_id", None)
    return values


def _without_cycle_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Remove experiment-contract switches from sequence constructors.

    The contract records the absence of the historical cycle auxiliary task,
    but those switches are not constructor arguments for the plain sequence
    baselines.  Keeping this filtering in the factory prevents a JSON contract
    field from accidentally changing a baseline's architecture.
    """

    values = dict(config)
    for key in ("use_cycle_prediction", "use_predicted_cycle_for_soh", "cycle_loss_mode"):
        values.pop(key, None)
    return values


def build_model(model_config: dict[str, Any], *, backend_override: str | None = None):
    """Build one explicitly named Paper-Backup model."""

    model_type = str(model_config.get("type", "")).strip()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unknown Paper-Backup model.type={model_type!r}; "
            f"supported={list(SUPPORTED_MODEL_TYPES)}"
        )
    values = _without_type(model_config)
    if model_type in {"Transformer", "VanillaMamba", "SingleStreamMamba", "RawCNN", "LSTM"}:
        values = _without_cycle_contract(values)
    if backend_override is not None and model_type in {
        "Ours", "VanillaMamba", "SingleStreamMamba",
        "FinalRawVanillaMamba", "FinalRawCCVanillaMamba",
        "FinalRawCVVanillaMamba", "FinalRawDualVanillaMamba",
        "FinalInteractionMamba", "FinalBiContextMamba",
        "FinalBiContextAdaptiveFusion",
        "FinalBiContextCycleMTL",
        "FinalLateLatentTokenBiContext",
    }:
        values["backend"] = backend_override

    if model_type in FINAL_MODEL_TYPES:
        values = _without_cycle_contract(values)
        return build_final_model(model_type, values)

    if model_type == "HI-MLP":
        return PINNFOnlyMLP(
            input_dim=int(values.get("input_dim", 24)),
            encoder_hidden_dim=int(values.get("encoder_hidden_dim", 60)),
            encoder_output_dim=int(values.get("encoder_output_dim", 32)),
            encoder_layers_num=int(values.get("encoder_layers_num", 3)),
            predictor_hidden_dim=int(values.get("predictor_hidden_dim", 32)),
            dropout=float(values.get("dropout", 0.2)),
        )
    if model_type == "Transformer":
        return TransformerSOHModel(**values)
    if model_type == "Ours":
        return PhaseMambaSOHOnly(**values)
    if model_type == "VanillaMamba":
        return VanillaMambaSOHModel(**values)
    if model_type == "SingleStreamMamba":
        return SingleStreamMambaSOHModel(**values)
    if model_type == "RawCNN":
        return RawCNNSOHModel(**values)
    if model_type == "LSTM":
        return LSTMSOHModel(**values)
    raise AssertionError(f"Unhandled Paper-Backup model type: {model_type}")


def model_input_kind(model_type: str) -> str:
    model_type = str(model_type)
    if model_type == "HI-MLP":
        return "features"
    if model_type == "Ours":
        return "phase"
    if model_type in {"Transformer", "VanillaMamba", "SingleStreamMamba", "RawCNN", "LSTM"}:
        return "sequence"
    if model_type == "FinalHI-MLP":
        return "features"
    if model_type in {
        "FinalRawCNN", "FinalRawLSTM", "FinalRawTransformer",
        "FinalRawVanillaMamba", "FinalRawCCVanillaMamba",
        "FinalRawCVVanillaMamba",
    }:
        return "sequence"
    if model_type in {
        "FinalRawDualVanillaMamba", "FinalInteractionMamba",
        "FinalBiContextMamba",
        "FinalBiContextAdaptiveFusion",
        "FinalBiContextCycleMTL",
        "FinalLateLatentTokenBiContext",
    }:
        return "phase"
    raise ValueError(f"Unknown Paper-Backup model type: {model_type!r}")


__all__ = ["SUPPORTED_MODEL_TYPES", "build_model", "model_input_kind"]
