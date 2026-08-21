"""Domain registry, SmartHealth audit, and E3 protocol contract tests."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.base import RawTerminalSignalUnavailable  # noqa: E402
from UnifiedRawSOH.datasets.domains import build_default_domain_registry, canonical_domain_id  # noqa: E402
from UnifiedRawSOH.datasets.smarthealth import (  # noqa: E402
    SMARTHEALTH_CANONICAL_POLICY_VERSION,
    SMARTHEALTH_PROCESSED_REQUIRED_COLUMNS,
    SmartHealthRawAdapter,
    audit_smarthealth_source,
    read_smarthealth_raw_file,
)
from UnifiedRawSOH.preprocess.smarthealth_common import (  # noqa: E402
    LISHEN40_CONFIG,
    CellSummary,
    CycleCandidate,
    PhaseResult,
    SourceIdentity,
    assign_chronological_cycle_ids,
    resolve_duplicate_candidates,
    source_absolute_time,
    visit_cycles,
)
from UnifiedRawSOH.datasets.xjtu import UnifiedCCCVSampleDataset  # noqa: E402
from UnifiedRawSOH.trainers.reusability import parse_reusability_protocol  # noqa: E402
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


class DomainAbstractionTest(unittest.TestCase):
    def test_smarthealth_source_timestamp_accepts_unpadded_fields(self):
        parsed = source_absolute_time("2022/8/4 8:27", Path("source.csv"), 0)
        self.assertEqual(parsed, datetime(2022, 8, 4, 8, 27))

    def test_smarthealth_visit_skips_truncated_source_row_with_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            header = [
                "循环号", "工步号", "工步类型", "时间", "绝对时间", "电流(A)",
                "电压(V)", "充电容量(Ah)", "放电容量(Ah)",
            ]
            rows = [
                ["1", "1", "恒流恒压充电", "00:00:00", "2022/8/4 8:27", "40", "3.45", "0", "0"],
                ["1", "1", "恒流恒压充电", "00:00:01", "2022/8/4 8:28", "40", "3.46", "0.01", "0"],
                ["1", "1", "恒流放"],
            ]
            with path.open("w", encoding="gb18030", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows)
            identity = SourceIdentity(
                path=path,
                relative_path="LISHEN/source.csv",
                config=LISHEN40_CONFIG,
                source_series="cell-1C-100%DOD",
                source_serial="cell",
                condition="1C-100%DOD",
                dod_percent=100,
                chunk_id=1,
                logical_sequence_id="smarthealth_lishen40__cell__1c_100_dod",
                file_size_bytes=path.stat().st_size,
                file_mtime_ns=path.stat().st_mtime_ns,
            )
            observed = []
            info = visit_cycles(
                identity,
                lambda cycle, points, has_temperature: observed.append(
                    (cycle, len(points), has_temperature)
                ),
            )
        self.assertEqual(observed, [(1, 2, False)])
        self.assertEqual(info["malformed_rows_skipped"], 1)
        self.assertIn("missing_required_point_values", info["malformed_row_reason_counts"])

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
            "循环号", "工步号", "工步类型", "绝对时间", "电流(A)", "电压(V)",
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
            columns = sorted({*SMARTHEALTH_PROCESSED_REQUIRED_COLUMNS, "label_capacity_Ah"})
            base = {
                "dataset": "smarthealth",
                "dataset_id": "smarthealth",
                "domain_id": "smarthealth_lishen40",
                "condition": "1C-100%DOD",
                "cell": "smarthealth_lishen40__cell__1c_100_dod",
                "battery_id": "smarthealth_lishen40__cell__1c_100_dod",
                "source_serial": "cell",
                "logical_sequence_id": "smarthealth_lishen40__cell__1c_100_dod",
                # Exported model cycles can have chronological-ID gaps when
                # an earlier source event fails phase/label eligibility.
                "cycle": "7",
                "SOH": "0.99",
                "label_capacity_Ah": "40.0",
                "label_source": "calibration_direct",
                "split_role": "development",
                "split_status": "complete",
                "split_issue": "",
                "split_strategy_version": "smarthealth_condition_cell_split_2development_1test_v3",
                "temperature_C": "25.0",
                "source_file": "source.csv",
                "chunk_id": "1",
                "source_cycle": "1",
                "source_absolute_start_time": "2022-01-01 00:00:00",
                "source_absolute_end_time": "2022-01-01 01:00:00",
                "strategy_version": SMARTHEALTH_CANONICAL_POLICY_VERSION,
                "phase_policy_version": SMARTHEALTH_CANONICAL_POLICY_VERSION,
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
            historical_records = read_smarthealth_raw_file(
                path,
                domain_id="smarthealth_lishen40",
                label_scale_mode="none",
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["battery_id"], base["battery_id"])
        self.assertEqual(records[0]["cycle_id"], 7)
        self.assertEqual(records[0]["segment"].tolist(), ["CC", "CC", "CV", "CV"])
        self.assertAlmostEqual(records[0]["soh"], 1.0)
        self.assertAlmostEqual(records[0]["soh_raw"], 40.0)
        self.assertEqual(records[0]["soh_scale_mode"], "label_capacity_to_nominal")
        self.assertAlmostEqual(historical_records[0]["soh"], 0.99)

    def test_smarthealth_chronology_preserves_reused_local_cycle_numbers(self):
        logical_sequence_id = "smarthealth_lishen40__cell__1c_100_dod"

        def candidate(chunk_id, source_cycle, start):
            identity = SourceIdentity(
                path=Path(f"/source/chunk-{chunk_id}.csv"),
                relative_path=f"LISHEN/condition/cell-1C-100%DOD-{chunk_id}.csv",
                config=LISHEN40_CONFIG,
                source_series="cell-1C-100%DOD",
                source_serial="cell",
                condition="1C-100%DOD",
                dod_percent=100,
                chunk_id=chunk_id,
                logical_sequence_id=logical_sequence_id,
                file_size_bytes=1,
                file_mtime_ns=1,
            )
            return CycleCandidate(
                identity=identity,
                source_cycle=source_cycle,
                source_absolute_start_time=start,
                source_absolute_end_time=start + timedelta(hours=1),
                source_rows=10,
                source_temperature_column_present=True,
                phase=PhaseResult(status="ok", reason="ok", temperature_complete=True),
                cycle_discharge_capacity_ah=40.0,
                candidate_eligible=True,
                candidate_eligibility_reason="ok",
            )

        earlier = candidate(20, 1, datetime(2022, 1, 1, 8, 0, 0))
        later = candidate(1, 1, datetime(2023, 1, 1, 8, 0, 0))
        candidates = {
            (
                logical_sequence_id,
                earlier.source_absolute_start_time,
                earlier.source_absolute_end_time,
            ): [earlier],
            (
                logical_sequence_id,
                later.source_absolute_start_time,
                later.source_absolute_end_time,
            ): [later],
        }
        cells = {logical_sequence_id: CellSummary(identity=earlier.identity)}

        selected_events = resolve_duplicate_candidates(candidates, cells)
        selected = assign_chronological_cycle_ids(selected_events, candidates)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[(logical_sequence_id, 1)].source_cycle, 1)
        self.assertEqual(selected[(logical_sequence_id, 2)].source_cycle, 1)
        self.assertEqual(
            selected[(logical_sequence_id, 1)].source_absolute_start_time,
            earlier.source_absolute_start_time,
        )
        self.assertEqual(cells[logical_sequence_id].unique_source_events, 2)

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
