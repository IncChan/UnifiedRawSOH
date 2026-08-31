"""Final-suite raw joint-sequence LSTM."""

from ..sequence_baselines import LSTMSOHModel


class FinalRawLSTMSOHModel(LSTMSOHModel):
    model_id = "Raw-LSTM-Final5"


__all__ = ["FinalRawLSTMSOHModel"]
