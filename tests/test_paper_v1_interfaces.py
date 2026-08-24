"""Paper-v1 source, registry, and mixed-protocol contract checks."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.base import RawTerminalSignalUnavailable  # noqa: E402
from UnifiedRawSOH.datasets.filters import filter_records_by_invalid_cycles  # noqa: E402
from UnifiedRawSOH.datasets.splits import (  # noqa: E402
    load_split_spec,
    load_test_batteries,
    split_mixed_cycle_records,
    split_records_from_spec,
)
from UnifiedRawSOH.datasets.mit import (  # noqa: E402
    MITFeatureAdapter,
    inspect_mit_raw_inventory,
    list_mit_raw_files,
    parse_mit_file_identity,
    read_mit_raw_file,
    validate_mit_physical_cohort,
)
from UnifiedRawSOH.evaluation.matched_cycle import _canonicalize_saved_mit_paths  # noqa: E402
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


class PaperV1InterfaceTest(unittest.TestCase):
    def test_main_config_has_no_aligned_source(self):
        config = load_config(
            PROJECT_ROOT / "UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_xjtu.json"
        )
        serialized = json.dumps(config)
        self.assertNotIn("XJTU_raw_t_v1_aligned", serialized)
        self.assertNotIn("MIT_t_v1_aligned", serialized)
        self.assertEqual(
            config["experiment"]["split_file"],
            "UnifiedRawSOH/splits/xjtu/paper_v1_mixed_split.json",
        )
        self.assertEqual(config["data"]["sample_filter_mode"], "none")
        self.assertEqual(config["experiment"]["domain_id"], "xjtu")

    def test_mixed_split_keeps_test_batteries_independent(self):
        records = []
        for battery in range(1, 9):
            for cycle in range(5):
                records.append({"battery_id": f"2C_battery-{battery}", "cycle_id": cycle})
        split_spec = load_split_spec(
            PROJECT_ROOT / "UnifiedRawSOH/splits/xjtu/paper_v1_mixed_split.json"
        )
        split = split_mixed_cycle_records(records, split_spec, condition="2C")
        train = {item["battery_id"] for item in split["train"]}
        val = {item["battery_id"] for item in split["val"]}
        test = {item["battery_id"] for item in split["test"]}
        self.assertEqual(test, {"2C_battery-4", "2C_battery-8"})
        self.assertEqual(train & test, set())
        self.assertEqual(val & test, set())
        self.assertTrue(train & val)

    def test_mit_feature_adapter_remains_intermediate_only(self):
        adapter = MITFeatureAdapter(
            PROJECT_ROOT / "UnifiedRawSOH/datasets/MIT_features"
        )
        audit = adapter.inspect()
        self.assertFalse(audit["raw_terminal_signals"])
        with self.assertRaises(RawTerminalSignalUnavailable):
            adapter.to_unified_samples()

    def test_matched_eval_uses_renamed_mit_directories_without_mutating_artifacts(self):
        saved_config = {
            "data": {
                "data_root": "UnifiedRawSOH/datasets/MIT_features_physical124",
                "feature_source": "PINN4SOH/data/MIT_t_v2_physical124",
            }
        }
        migrated = _canonicalize_saved_mit_paths(saved_config)
        self.assertEqual(
            migrated["data"]["data_root"],
            "UnifiedRawSOH/datasets/MIT_features",
        )
        self.assertEqual(
            migrated["data"]["feature_source"],
            "PINN4SOH/data/MIT_features",
        )
        self.assertEqual(
            saved_config["data"]["data_root"],
            "UnifiedRawSOH/datasets/MIT_features_physical124",
        )

    def test_mit_raw_listing_ignores_physical_provenance_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "MIT_2017-05-12_physical-001.csv").touch()
            (root / "MIT_PHYSICAL_PROVENANCE.csv").touch()
            (root / "mit_physical_extraction_report.csv").touch()
            self.assertEqual(
                [path.name for path in list_mit_raw_files(root)],
                ["MIT_2017-05-12_physical-001.csv"],
            )

    def test_mit_header_only_inventory_is_not_a_runnable_raw_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "MIT_2017-05-12_physical-001.csv").write_text(
                "cycle,SOH,capacity_Ah,segment,relative_time_min,voltage_V,current_A,temperature_C\n",
                encoding="utf-8",
            )
            inventory = inspect_mit_raw_inventory(root)
        self.assertEqual(inventory["nonempty_files"], [])
        self.assertEqual(inventory["header_only_files"], ["MIT_2017-05-12_physical-001.csv"])

    def test_mit_raw_adapter_physical_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MIT_2017-05-12_physical-001.csv"
            path.write_text(
                "physical_cell_id,paper_batch,primary_batch_date,cycle,source_batch_date,source_cell,source_cycle,SOH,capacity_Ah,segment,relative_time_min,voltage_V,current_A,temperature_C,c_rate,phase_policy_version,phase_detection_status,phase_detection_reason,cc_voltage_low_V,cc_voltage_high_V,cv_c_rate_low,cv_c_rate_high\n"
                "mit_p001,batch1,2017-05-12,2,2017-05-12,1,2,0.98,1.078,CC,1.0,3.45,1.0,30.0,0.9090909091,mit_proposed_phase_aware_cccv_v3,ok,persistent_current_taper_near_charge_voltage_max,3.45,3.60,0.05,0.25\n"
                "mit_p001,batch1,2017-05-12,2,2017-05-12,1,2,0.98,1.078,CC,1.5,3.60,1.0,30.0,0.9090909091,mit_proposed_phase_aware_cccv_v3,ok,persistent_current_taper_near_charge_voltage_max,3.45,3.60,0.05,0.25\n"
                "mit_p001,batch1,2017-05-12,2,2017-05-12,1,2,0.98,1.078,CV,2.0,3.60,0.2739,30.1,0.249,mit_proposed_phase_aware_cccv_v3,ok,persistent_current_taper_near_charge_voltage_max,3.45,3.60,0.05,0.25\n"
                "mit_p001,batch1,2017-05-12,2,2017-05-12,1,2,0.98,1.078,CV,2.5,3.60,0.055,30.1,0.05,mit_proposed_phase_aware_cccv_v3,ok,persistent_current_taper_near_charge_voltage_max,3.45,3.60,0.05,0.25\n"
                "mit_p001,batch1,2017-05-12,1189,2017-06-30,8,1,0.80,0.88,CC,1.0,3.45,1.0,31.0,0.9090909091,mit_proposed_phase_aware_cccv_v3,ok,persistent_current_taper_near_charge_voltage_max,3.45,3.60,0.05,0.25\n"
                "mit_p001,batch1,2017-05-12,1189,2017-06-30,8,1,0.80,0.88,CC,1.5,3.60,1.0,31.0,0.9090909091,mit_proposed_phase_aware_cccv_v3,ok,persistent_current_taper_near_charge_voltage_max,3.45,3.60,0.05,0.25\n"
                "mit_p001,batch1,2017-05-12,1189,2017-06-30,8,1,0.80,0.88,CV,2.0,3.60,0.2739,31.1,0.249,mit_proposed_phase_aware_cccv_v3,ok,persistent_current_taper_near_charge_voltage_max,3.45,3.60,0.05,0.25\n"
                "mit_p001,batch1,2017-05-12,1189,2017-06-30,8,1,0.80,0.88,CV,2.5,3.60,0.055,31.1,0.05,mit_proposed_phase_aware_cccv_v3,ok,persistent_current_taper_near_charge_voltage_max,3.45,3.60,0.05,0.25\n",
                encoding="utf-8",
            )
            records = read_mit_raw_file(path)
        self.assertEqual([record["cycle_id"] for record in records], [2, 1189])
        self.assertEqual(records[0]["battery_id"], "mit_p001")
        self.assertEqual(records[1]["source_batch_date"], "2017-06-30")
        self.assertEqual(records[1]["source_cell"], 8)
        self.assertEqual(records[1]["source_cycle"], 1)
        self.assertEqual(set(records[0]["segment"].tolist()), {"CC", "CV"})
        self.assertTrue(all(record["dataset_id"] == "mit" for record in records))

    def test_mit_physical_filename_identity(self):
        self.assertEqual(
            parse_mit_file_identity("MIT_2017-06-30_physical-042.csv"),
            ("2017-06-30", "mit_p042", True),
        )

    def test_mit_invalid_cycle_filter_is_explicit(self):
        records = [
            {"battery_id": "mit_p015", "cycle_id": 38},
            {"battery_id": "mit_p015", "cycle_id": 39},
        ]
        kept, audit = filter_records_by_invalid_cycles(
            records,
            [{"battery_id": "mit_p015", "cycle_id": 39}],
        )
        self.assertEqual([item["cycle_id"] for item in kept], [38])
        self.assertEqual(audit["removed_records"], 1)

    def test_mit_physical_modulo_rule_is_consumed_by_mixed_loader(self):
        records = [
            {
                "battery_id": f"mit_p{battery:03d}",
                "condition": date,
                "cycle_id": cycle,
            }
            for date, start, end in (
                ("2017-05-12", 1, 41),
                ("2017-06-30", 42, 84),
                ("2018-04-12", 85, 124),
            )
            for battery in range(start, end + 1)
            for cycle in range(5)
        ]
        split, metadata = split_records_from_spec(
            records,
            load_split_spec(PROJECT_ROOT / "UnifiedRawSOH/splits/mit/mit_paper_physical124_v2_split.json"),
            split_file=PROJECT_ROOT / "UnifiedRawSOH/splits/mit/mit_paper_physical124_v2_split.json",
        )
        self.assertEqual(len(metadata["test_batteries"]), 24)
        self.assertEqual(
            {item["battery_id"] for item in split["train"]}
            & {item["battery_id"] for item in split["test"]},
            set(),
        )
        self.assertTrue(
            {item["battery_id"] for item in split["train"]}
            & {item["battery_id"] for item in split["val"]}
        )

    def test_mit_physical_split_spec_is_fixed_and_mixed(self):
        path = PROJECT_ROOT / "UnifiedRawSOH/splits/mit/mit_paper_physical124_v2_split.json"
        with path.open(encoding="utf-8") as handle:
            split = json.load(handle)
        self.assertEqual(split["test_rule"]["type"], "physical_id_modulo")
        self.assertEqual(split["test_rule"]["modulus"], 5)
        self.assertEqual(len(split["test_batteries"]), 24)
        observed = [f"mit_p{index:03d}" for index in range(1, 125)]
        self.assertEqual(
            set(load_test_batteries(path, observed_battery_ids=observed)),
            set(split["test_batteries"]),
        )
        expanded = observed + ["mit_p125"]
        self.assertIn("mit_p125", load_test_batteries(path, expanded))
        self.assertEqual(split["development_split"]["random_state"], 420)
        self.assertEqual(split["development_split"]["scope"], "single_domain_pool")
        self.assertEqual(split["invalid_cycles"][0]["cycle_id"], 39)

    def test_official_mit_config_can_require_the_complete_physical_cohort(self):
        split = load_split_spec(
            PROJECT_ROOT / "UnifiedRawSOH/splits/mit/mit_paper_physical124_v2_split.json"
        )
        observed = [f"mit_p{index:03d}" for index in range(1, 125)]
        audit = validate_mit_physical_cohort(
            observed,
            split,
            require_full_physical_cohort=True,
        )
        self.assertEqual(len(audit["physical_ids"]), 124)
        self.assertEqual(len(audit["test_ids"]), 24)
        self.assertEqual(len(audit["development_ids"]), 100)
        with self.assertRaisesRegex(ValueError, "does not match declared Paper-124"):
            validate_mit_physical_cohort(
                observed[:-1],
                split,
                require_full_physical_cohort=True,
            )


if __name__ == "__main__":
    unittest.main()
