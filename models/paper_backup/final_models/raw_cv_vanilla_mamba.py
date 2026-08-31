"""Final-suite raw terminal-CV-only Vanilla Mamba."""

from ..sequence_baselines import SingleStreamMambaSOHModel


class RawCVVanillaMambaSOHModel(SingleStreamMambaSOHModel):
    model_id = "Raw-CV-Vanilla-Mamba-Final5"
    phase = "cv"


__all__ = ["RawCVVanillaMambaSOHModel"]
