"""Final-suite raw terminal-CC-only Vanilla Mamba."""

from ..sequence_baselines import SingleStreamMambaSOHModel


class RawCCVanillaMambaSOHModel(SingleStreamMambaSOHModel):
    model_id = "Raw-CC-Vanilla-Mamba-Final5"
    phase = "cc"


__all__ = ["RawCCVanillaMambaSOHModel"]
