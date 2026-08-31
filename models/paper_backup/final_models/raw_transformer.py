"""Final-suite raw joint-sequence Transformer."""

from ..sequence_baselines import TransformerSOHModel


class FinalRawTransformerSOHModel(TransformerSOHModel):
    model_id = "Raw-Transformer-Final5"


__all__ = ["FinalRawTransformerSOHModel"]
