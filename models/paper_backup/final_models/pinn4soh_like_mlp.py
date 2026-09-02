"""Unfiltered Feature MLP using the PINN4SOH F-only network structure.

Only the encoder/predictor architecture is reused.  The final E1 suite reads
the same complete offline terminal cohort as the raw models and deliberately
does not reproduce the archived 3-sigma or adjacent-x1 sample filters.
"""

from ...baselines.pinn4soh_no_leak_onlyf import PINNFOnlyMLP


class FinalPINN4SOHLikeMLP(PINNFOnlyMLP):
    model_id = "Feature-MLP-PINN4SOH-Structure-Final5"


__all__ = ["FinalPINN4SOHLikeMLP"]
