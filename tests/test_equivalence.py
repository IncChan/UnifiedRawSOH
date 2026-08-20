"""Small equivalence checks against the historical C5B source semantics."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

XJTU_RAW_ROOT = REPOSITORY_ROOT / "datasets/XJTU_raw"
LEGACY_V2_ROOT = WORKSPACE_ROOT / "SC_TempMamba_v2"

from UnifiedRawSOH.datasets.xjtu import (  # noqa: E402
    UnifiedCCCVSampleDataset,
    parse_file_identity,
    read_xjtu_file,
)
from UnifiedRawSOH.datasets.splits import load_battery_roles  # noqa: E402
from UnifiedRawSOH.models.c5b_model import StandardSingleCycleMamba  # noqa: E402


class C5BEquivalenceTest(unittest.TestCase):
    def test_fixed_xjtu_roles_cover_all_reference_batteries(self):
        roles = load_battery_roles(REPOSITORY_ROOT / "splits/xjtu/c5b_battery_split.json")
        observed = {
            parse_file_identity(path)[1]
            for path in XJTU_RAW_ROOT.glob("*.csv")
            if not path.name.endswith("_report.csv")
        }
        declared = {battery_id for values in roles.values() for battery_id in values}
        self.assertEqual(observed, declared)
        self.assertEqual(len(declared), 55)

    def test_reader_contract_on_raw_t_v1_file(self):
        path = XJTU_RAW_ROOT / "2C_battery-1.csv"
        records = read_xjtu_file(path, nominal_capacity=2.0, label_scale_mode="auto_capacity_to_soh")
        self.assertGreater(len(records), 1)
        first = records[0]
        self.assertEqual(first["battery_id"], "2C_battery-1")
        self.assertEqual(first["dataset_id"], "xjtu")
        self.assertEqual(first["cycle_id"], 2)
        self.assertGreaterEqual(int((first["segment"] == "CC").sum()), 4)
        self.assertGreaterEqual(int((first["segment"] == "CV").sum()), 4)
        self.assertTrue(np.isfinite(first["voltage"]).all())
        self.assertTrue(np.isfinite(first["temperature"]).all())

    def test_single_sample_preprocessing_matches_v2(self):
        if not LEGACY_V2_ROOT.is_dir():
            self.skipTest("historical SC_TempMamba_v2 reference is not installed")
        from SC_TempMamba_v2.data.standard_mamba_dataset import (
            StandardMambaCycleDataset as LegacyDataset,
            build_full_life_cycle_metadata as legacy_cycle_metadata,
        )
        path = XJTU_RAW_ROOT / "2C_battery-1.csv"
        records = read_xjtu_file(path, nominal_capacity=2.0, label_scale_mode="auto_capacity_to_soh")
        normalization = {
            "mode": "physical_window",
            "cc_voltage_low": 4.0,
            "cc_voltage_high": 4.195,
            "cv_current_low": 0.1,
            "cv_current_high": 0.5,
            "cc_current_ref": 4.0,
            "cv_voltage_ref": 4.19975,
            "cv_voltage_scale": 0.001,
            "cc_current_mode": "physical",
            "cv_voltage_mode": "physical",
            "current_use_abs": True,
            "temp_room": 25.0,
            "temp_abs_scale": 20.0,
            "temp_delta_scale": 10.0,
        }
        metadata = legacy_cycle_metadata(records)
        legacy = LegacyDataset(
            [records[0]],
            raw_len_cc=128,
            raw_len_cv=256,
            min_cc_points=4,
            min_cv_points=4,
            normalizer_config=normalization,
            use_real_time=True,
            time_origin="charge_window_start",
            use_tau=True,
            use_temperature=True,
            temperature_reference_c=25.0,
            temperature_scale_c=20.0,
            temperature_delta_origin="charge_window_start",
            temperature_delta_scale_c=10.0,
            use_t0_temperature_meta=True,
            t0_temperature_reference_c=25.0,
            t0_temperature_scale_c=20.0,
            use_cycle_prediction_targets=True,
            cycle_life_metadata_by_battery=metadata,
        )
        modern = UnifiedCCCVSampleDataset(
            [read_xjtu_file(path, nominal_capacity=2.0, label_scale_mode="auto_capacity_to_soh")[0]],
            {
                "raw_len_cc": 128,
                "raw_len_cv": 256,
                "min_cc_points": 4,
                "min_cv_points": 4,
                "use_real_time": True,
                "use_temperature": True,
                "use_t0_temperature_meta": True,
                "temperature_reference_c": 25.0,
                "temperature_scale_c": 20.0,
                "temperature_delta_scale_c": 10.0,
                "t0_temperature_reference_c": 25.0,
                "t0_temperature_scale_c": 20.0,
            },
            normalization,
            split_name="test",
            cycle_metadata=metadata,
        )
        old_item = legacy[0]
        new_item = modern[0]
        for key in (
            "cc_signal", "cv_signal", "cc_time", "cv_time",
            "cc_temperature", "cv_temperature", "t0_temperature_norm",
            "soh", "cycle_life_norm_target",
        ):
            np.testing.assert_allclose(old_item[key].numpy(), new_item[key].numpy(), rtol=0, atol=1e-6, err_msg=key)

    def test_c5b_official_state_contract_matches_v2(self):
        legacy_config = LEGACY_V2_ROOT / "configs/0/mamba/C5B_strong_multitask.json"
        if not legacy_config.is_file():
            self.skipTest("historical SC_TempMamba_v2 model config is not installed")
        try:
            from SC_TempMamba_v2.models.standard_mamba import build_standard_mamba_from_config
        except Exception as exc:  # pragma: no cover - dependency-specific
            self.skipTest(f"historical model unavailable: {exc}")
        with legacy_config.open() as handle:
            config = json.load(handle)
        try:
            old_model = build_standard_mamba_from_config(config["model"])
            new_model = StandardSingleCycleMamba(
                input_dim=4,
                d_model=32,
                num_layers=3,
                d_state=8,
                d_conv=4,
                expand=2,
                dt_rank="auto",
                dropout=0.1,
                pooling="last_mean",
                fusion_type="concat",
                fusion_phase_dim=64,
                head_hidden_dim=128,
                use_time_as_input=True,
                temperature_injection="input_concat",
                temperature_features="delta",
                use_t0_temperature_meta=True,
                t0_temperature_meta_dim=1,
                use_cc_to_cv_bridge=True,
                cc_to_cv_bridge_type="zero_init_linear",
                cc_to_cv_bridge_input_dim=64,
                cc_to_cv_bridge_output_dim=32,
                use_cycle_prediction=True,
                cycle_target="cycle_life_norm",
                cycle_head_hidden_dim=64,
                cycle_output_activation="sigmoid",
                use_predicted_cycle_for_soh=True,
                detach_predicted_cycle_for_soh=False,
                backend="mamba_ssm.Mamba",
            )
        except Exception as exc:  # pragma: no cover - CUDA/backend-specific
            self.skipTest(f"official Mamba construction unavailable: {exc}")
        self.assertEqual(set(old_model.state_dict()), set(new_model.state_dict()))
        for key, value in old_model.state_dict().items():
            self.assertEqual(tuple(value.shape), tuple(new_model.state_dict()[key].shape), key)


if __name__ == "__main__":
    unittest.main()
