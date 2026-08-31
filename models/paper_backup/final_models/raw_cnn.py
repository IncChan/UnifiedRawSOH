"""Final-suite raw joint-sequence CNN."""

from ..sequence_baselines import RawCNNSOHModel


class FinalRawCNNSOHModel(RawCNNSOHModel):
    model_id = "Raw-CNN-Final5"


__all__ = ["FinalRawCNNSOHModel"]
