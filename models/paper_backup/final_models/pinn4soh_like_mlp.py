"""Final-suite PINN4SOH-like no-leak statistical-feature MLP."""

from ...baselines.pinn4soh_no_leak_onlyf import PINNFOnlyMLP


class FinalPINN4SOHLikeMLP(PINNFOnlyMLP):
    model_id = "PINN4SOH-like-MLP-Final5"


__all__ = ["FinalPINN4SOHLikeMLP"]
