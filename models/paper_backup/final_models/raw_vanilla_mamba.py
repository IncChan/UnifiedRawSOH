"""Final-suite raw joint-sequence Vanilla Mamba."""

from ..sequence_baselines import VanillaMambaSOHModel


class FinalRawVanillaMambaSOHModel(VanillaMambaSOHModel):
    model_id = "Raw-Vanilla-Mamba-Final5"


__all__ = ["FinalRawVanillaMambaSOHModel"]
