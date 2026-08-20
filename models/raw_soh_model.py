"""Formal Paper-v1 raw SOH model entry point.

The implementation is kept in ``c5b_model.py`` to preserve the small, tested
extraction. This module is the stable paper-facing import path.
"""

from .c5b_model import PaperRawSOHModel


def build_raw_soh_model(model_config, backend_override=None):
    from .c5b_model import build_c5b_model

    return build_c5b_model(model_config, backend_override=backend_override)


__all__ = ["PaperRawSOHModel", "build_raw_soh_model"]
