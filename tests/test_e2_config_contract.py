"""Static contracts for the runnable E2 shared-model configurations."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.utils.config import load_config  # noqa: E402


CONFIG_ROOT = PROJECT_ROOT / "UnifiedRawSOH" / "configs" / "paper_v1"
E1_ROOT = CONFIG_ROOT / "e1_raw_soh_learning" / "benchmark"
E2_ROOT = CONFIG_ROOT / "e2_unified_multidomain" / "unified"


class E2ConfigContractTest(unittest.TestCase):
    """E2 may compose domains, but must not silently retune RawMamba."""

    def setUp(self):
        self.e1_xjtu = load_config(E1_ROOT / "raw_mamba_xjtu.json")

    def _assert_e1_rawmamba_inheritance(self, path, expected_domains):
        config = load_config(path)
        self.assertEqual(config["status"], "runnable")
        self.assertEqual(config["experiment"]["loader"], "unified_multi_dataset")
        self.assertEqual(config["experiment"]["domain_ids"], expected_domains)
        self.assertEqual(config["model"], self.e1_xjtu["model"])
        self.assertEqual(config["eval"], self.e1_xjtu["eval"])
        self.assertEqual(config["debug"], self.e1_xjtu["debug"])

        expected_train = copy.deepcopy(self.e1_xjtu["train"])
        expected_train["monitor"] = "valid_domain_macro_rmse"
        self.assertEqual(config["train"], expected_train)
        return config

    def test_pilot_inherits_e1_and_declares_two_domains(self):
        config = self._assert_e1_rawmamba_inheritance(
            E2_ROOT / "public_xjtu_mit.json", ["xjtu", "mit"]
        )
        self.assertEqual(config["data"]["balance_mode"], "domain_battery")
        self.assertEqual(set(config["data"]["normalizations"]), {"xjtu", "mit"})

    def test_full_inherits_e1_and_matches_each_domain_contract(self):
        domain_configs = {
            "xjtu": E1_ROOT / "raw_mamba_xjtu.json",
            "mit": E1_ROOT / "raw_mamba_mit.json",
            "smarthealth_lishen40": E1_ROOT / "raw_mamba_smarthealth_lishen40.json",
            "smarthealth_catl280": E1_ROOT / "raw_mamba_smarthealth_catl280.json",
            "smarthealth_eve280": E1_ROOT / "raw_mamba_smarthealth_eve280.json",
        }
        config = self._assert_e1_rawmamba_inheritance(
            E2_ROOT / "public_all_domains.json", list(domain_configs)
        )
        data = config["data"]
        self.assertEqual(data["balance_mode"], "domain_battery")
        self.assertEqual(set(data["data_roots"]), set(domain_configs))
        self.assertEqual(set(data["split_files"]), set(domain_configs))
        self.assertEqual(set(data["label_scale_modes"]), set(domain_configs))
        self.assertEqual(set(data["normalizations"]), set(domain_configs))

        for domain_id, source_path in domain_configs.items():
            source = load_config(source_path)
            self.assertEqual(data["nominal_capacities"][domain_id], source["data"]["nominal_capacity"])
            self.assertEqual(data["label_scale_modes"][domain_id], source["data"]["label_scale_mode"])
            self.assertEqual(data["normalizations"][domain_id], source["normalization"])
            self.assertEqual(
                data["batches_by_domain"][domain_id],
                source["experiment"].get("batches", []),
            )


    def _assert_domain_balanced_without_cycle_aux(self, path, expected_domains):
        config = load_config(path)
        self.assertEqual(config["status"], "runnable")
        self.assertEqual(config["experiment"]["domain_ids"], expected_domains)
        self.assertEqual(config["data"]["balance_mode"], "domain_battery_hierarchical")
        self.assertEqual(config["data"]["cycle_target_scope"], "not_used")
        self.assertFalse(config["model"]["use_cycle_prediction"])
        self.assertFalse(config["model"]["use_predicted_cycle_for_soh"])
        self.assertEqual(config["train"]["lambda_cycle"], 0.0)
        self.assertEqual(config["train"]["cycle_loss_mode"], "disabled")
        self.assertEqual(config["train"]["monitor"], "valid_domain_macro_rmse")

    def test_pilot_d_without_cycle_auxiliary(self):
        self._assert_domain_balanced_without_cycle_aux(
            E2_ROOT / "public_xjtu_mit_domain_balanced_no_cycle_aux.json",
            ["xjtu", "mit"],
        )

    def test_full_d_without_cycle_auxiliary(self):
        self._assert_domain_balanced_without_cycle_aux(
            E2_ROOT / "public_all_domains_domain_balanced_no_cycle_aux.json",
            [
                "xjtu",
                "mit",
                "smarthealth_lishen40",
                "smarthealth_catl280",
                "smarthealth_eve280",
            ],
        )


if __name__ == "__main__":
    unittest.main()

