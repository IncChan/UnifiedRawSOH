"""Fixed physical normalization for the unified raw CC/CV interface.

The C5B path deliberately does not fit statistics from train/validation/test
records.  Voltage/current ranges, temperature references, and time scales are
protocol/physical constants recorded in the experiment configuration.
"""

from __future__ import annotations

import copy

import numpy as np


class PhysicalWindowNormalizer:
    """Normalize one CC or CV phase using fixed physical windows."""

    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        if self.config.get("mode") != "physical_window":
            raise ValueError("Only normalization.mode='physical_window' is supported.")
        for key in ("cc_current_mode", "cv_voltage_mode"):
            mode = self.config.get(key, "physical")
            if mode not in {"physical", "zero"}:
                raise ValueError(f"{key} must be 'physical' or 'zero', got {mode!r}.")

    def normalize_cc_voltage(self, voltage):
        voltage = np.asarray(voltage, dtype=np.float32)
        denominator = self.config["cc_voltage_high"] - self.config["cc_voltage_low"]
        if denominator <= 0:
            raise ValueError("cc_voltage_high must be greater than cc_voltage_low.")
        return (2.0 * (voltage - self.config["cc_voltage_low"]) / denominator - 1.0).astype(
            np.float32
        )

    def normalize_cv_current(self, current):
        current = np.asarray(current, dtype=np.float32)
        current_value = np.abs(current) if self.config.get("current_use_abs", True) else current
        denominator = self.config["cv_current_high"] - self.config["cv_current_low"]
        if denominator <= 0:
            raise ValueError("cv_current_high must be greater than cv_current_low.")
        return (
            2.0 * (current_value - self.config["cv_current_low"]) / denominator - 1.0
        ).astype(np.float32)

    def normalize_cc_current_abs(self, current, scale=None):
        current = np.asarray(current, dtype=np.float32)
        current_value = np.abs(current) if self.config.get("current_use_abs", True) else current
        scale = (
            self.config.get("cc_current_scale", self.config.get("cc_current_ref", 4.0))
            if scale is None
            else scale
        )
        if float(scale) <= 0:
            raise ValueError("CC-current normalization scale must be positive.")
        return (current_value / float(scale)).astype(np.float32)

    def state_dict(self):
        return copy.deepcopy(self.config)

