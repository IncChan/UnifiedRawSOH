"""Factory local to the final five-seed E1/E2 model collection."""

from __future__ import annotations

from typing import Any

from . import (
    BiContextAdaptiveFusionSOHModel,
    BiContextCycleMTLSOHModel,
    BiContextMambaSOHModel,
    FinalPINN4SOHLikeMLP,
    FinalRawCNNSOHModel,
    FinalRawLSTMSOHModel,
    FinalRawTransformerSOHModel,
    FinalRawVanillaMambaSOHModel,
    InteractionFusionMambaSOHModel,
    RawCCVanillaMambaSOHModel,
    RawCVVanillaMambaSOHModel,
    RawDualVanillaMambaSOHModel,
)


FINAL_MODEL_TYPES = {
    "FinalHI-MLP": FinalPINN4SOHLikeMLP,
    "FinalRawCNN": FinalRawCNNSOHModel,
    "FinalRawLSTM": FinalRawLSTMSOHModel,
    "FinalRawTransformer": FinalRawTransformerSOHModel,
    "FinalRawVanillaMamba": FinalRawVanillaMambaSOHModel,
    "FinalRawCCVanillaMamba": RawCCVanillaMambaSOHModel,
    "FinalRawCVVanillaMamba": RawCVVanillaMambaSOHModel,
    "FinalRawDualVanillaMamba": RawDualVanillaMambaSOHModel,
    "FinalInteractionMamba": InteractionFusionMambaSOHModel,
    "FinalBiContextMamba": BiContextMambaSOHModel,
    "FinalBiContextAdaptiveFusion": BiContextAdaptiveFusionSOHModel,
    "FinalBiContextCycleMTL": BiContextCycleMTLSOHModel,
}


def build_final_model(model_type: str, values: dict[str, Any]):
    constructor = FINAL_MODEL_TYPES[str(model_type)]
    if model_type == "FinalHI-MLP":
        return constructor(
            input_dim=int(values.get("input_dim", 24)),
            encoder_hidden_dim=int(values.get("encoder_hidden_dim", 60)),
            encoder_output_dim=int(values.get("encoder_output_dim", 32)),
            encoder_layers_num=int(values.get("encoder_layers_num", 3)),
            predictor_hidden_dim=int(values.get("predictor_hidden_dim", 32)),
            dropout=float(values.get("dropout", 0.2)),
        )
    return constructor(**values)


__all__ = ["FINAL_MODEL_TYPES", "build_final_model"]
