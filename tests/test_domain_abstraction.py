"""Domain registry, SmartHealth audit, and E3 protocol contract tests."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.base import RawTerminalSignalUnavailable  # noqa: E402
from UnifiedRawSOH.datasets.domains import build_default_domain_registry, canonical_domain_id  # noqa: E402
from UnifiedRawSOH.datasets.smarthealth import (  # noqa: E402
    SMARTHEALTH_PROCESSED_REQUIRED_COLUMNS,
    SmartHealthRawAdapter,
    audit_smarthealth_source,
    read_smarthealth_raw_file,
)
from UnifiedRawSOH.datasets.xjtu import UnifiedCCCVSampleDataset  # noqa: E402
from UnifiedRawSOH.trainers.reusability import parse_reusability_protocol  # noqa: E402
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


class DomainAbstractionTest(unittest.TestCase):
    def test_registry_maps_paper_alias_and_legacy_source_names(self):
        registry = build_default_domain_registry()
        self.assertEqual(registry.canonical_id("A"), "xjtu")
        self.assertEqual(registry.canonical_id("C3"), "smarthealth_eve280")
        self.assertEqual(canonical_domain_id("MIT_features"), "mit")
        self.assertEqual(
            registry.get("smarthealth_lishen40").manufacturer,
            "LISHEN",
        )
        self.assertIn("normalization", registry.get("xjtu").metadata())
        self.assertEqual(registry.get("smarthealth_lishen40").availability, "available")

    def test_smarthealth_audit_does_not_pretend_missing_temperature_exists(self):
        header = [
            "循环号", "工步号", "工步类型", "电流(A)", "电压(V)",
            "充电容量(Ah)", "放电容量(Ah)", "temp1_1",
        ]
        without_temp = header[:-1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, columns in (("with-temp.csv", header), ("without-temp.csv", without_temp)):
                with (root / name).open("w", encoding="gb18030", newline="") as handle:
                    csv.writer(handle).writerow(columns)
            audit = audit_smarthealth_source(root, "smarthealth_eve280")
            self.assertEqual(audit["files"], 2)
            self.assertEqual(audit["invalid_header_files"], 1)
            self.assertFalse(audit["raw_signal_columns_confirmed"])
            with self.assertRaises(RawTerminalSignalUnavailable):
                SmartHealthRawAdapter(root, "smarthealth_eve280").load_records()

    def test_vi_ablation_dataset_does_not_require_temperature_rows(self):
        record = {
            "dataset_id": "xjtu",
            "domain_id": "xjtu",
            "condition": "synthetic_condition",
            "battery_id": "synthetic_cell",
            "cycle_id": 1,
            "raw_cycle_order_index": 0,
            "segment": np.asarray(["CC"] * 4 + ["CV"] * 4, dtype=object),
            "time": np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
            "voltage": np.asarray([4.0, 4.05, 4.1, 4.15, 4.2, 4.2, 4.2, 4.2], dtype=np.float32),
            "current": np.asarray([4.0, 4.0, 4.0, 4.0, 0.5, 0.4, 0.3, 0.2], dtype=np.float32),
            "soh": 0.98,
            "soh_raw": 0.98,
        }
        data_config = {
            "raw_len_cc": 4,
            "raw_len_cv": 4,
            "min_cc_points": 4,
            "min_cv_points": 4,
            "use_real_time": True,
            "use_temperature": False,
            "use_t0_temperature_meta": False,
        }
        normalization = build_default_domain_registry().get("xjtu").normalization
        dataset = UnifiedCCCVSampleDataset(
            [record],
            data_config,
            normalization,
            split_name="test",
        )
        item = dataset[0]
        self.assertTrue(np.allclose(item["cc_temperature"].numpy(), 0.0))
        self.assertTrue(np.allclose(item["t0_temperature_norm"].numpy(), 0.0))

    def test_canonical_smarthealth_adapter_validates_phase_then_window_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smarthealth_lishen40__cell__1c_100_dod.csv"
            columns = sorted(SMARTHEALTH_PROCESSED_REQUIRED_COLUMNS)
            base = {
                "dataset": "smarthealth",
                "dataset_id": "smarthealth",
                "domain_id": "smarthealth_lishen40",
                "condition": "1C-100%DOD",
                "cell": "smarthealth_lishen40__cell__1c_100_dod",
                "battery_id": "smarthealth_lishen40__cell__1c_100_dod",
                "source_serial": "cell",
                "logical_sequence_id": "smarthealth_lishen40__cell__1c_100_dod",
                "cycle": "1",
                "SOH": "0.99",
                "label_source": "calibration_direct",
                "split_role": "development",
                "split_status": "complete",
                "split_issue": "",
                "split_strategy_version": "smarthealth_condition_cell_split_2development_1test_v3",
                "temperature_C": "25.0",
                "source_file": "source.csv",
                "chunk_id": "1",
                "source_cycle": "1",
                "strategy_version": "smarthealth_cccv_calibration_v2",
                "phase_policy_version": "smarthealth_cccv_calibration_v2",
                "cc_voltage_low_V": "3.45",
                "cc_voltage_high_V": "3.58",
                "cv_c_rate_low": "0.05",
                "cv_c_rate_high": "0.25",
            }
            rows = [
                {**base, "segment": "CC", "cycle_point_index": "0", "segment_point_index": "0", "relative_time": "0", "voltage_V": "3.45", "current_A": "40", "c_rate": "1.0"},
                {**base, "segment": "CC", "cycle_point_index": "1", "segment_point_index": "1", "relative_time": "1", "voltage_V": "3.58", "current_A": "40", "c_rate": "1.0"},
                {**base, "segment": "CV", "cycle_point_index": "2", "segment_point_index": "0", "relative_time": "2", "voltage_V": "3.60", "current_A": "10.08", "c_rate": "0.252"},
                {**base, "segment": "CV", "cycle_point_index": "3", "segment_point_index": "1", "relative_time": "3", "voltage_V": "3.60", "current_A": "2.0", "c_rate": "0.05"},
            ]
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            records = read_smarthealth_raw_file(path, domain_id="smarthealth_lishen40")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["battery_id"], base["battery_id"])
        self.assertEqual(records[0]["segment"].tolist(), ["CC", "CC", "CV", "CV"])

    def test_reusability_protocol_requires_disjoint_domains_and_fair_budget(self):
        adaptation = load_config(
            PROJECT_ROOT
            / "UnifiedRawSOH/configs/e3_cross_domain_reusability/adaptation/xjtu_to_mit_cycle_fraction.json"
        )
        parsed = parse_reusability_protocol(adaptation)
        self.assertEqual(parsed["source_domain_ids"], ["xjtu"])
        self.assertEqual(parsed["target_domain_ids"], ["mit"])
        self.assertEqual(parsed["target_budget"], {"unit": "cycle_fraction", "value": 0.05})
        self.assertTrue(parsed["scratch_same_target_budget"])

        lodo = load_config(
            PROJECT_ROOT
            / "UnifiedRawSOH/configs/e3_cross_domain_reusability/leave_one_domain_out/lodo_smarthealth_eve280.json"
        )
        self.assertEqual(
            parse_reusability_protocol(lodo)["target_domain_ids"],
            ["smarthealth_eve280"],
        )

        invalid = {
            "experiment": {"source_domain_ids": ["xjtu"], "target_domain_id": "xjtu"},
            "reusability": {"protocol": "leave_one_domain_out"},
        }
        with self.assertRaises(ValueError):
            parse_reusability_protocol(invalid)


if __name__ == "__main__":
    unittest.main()
