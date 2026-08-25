"""Contracts for E1 RawOurs without cycle auxiliary supervision."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.utils.config import load_config  # noqa: E402


CONFIG_ROOT = (
    PROJECT_ROOT
    / "UnifiedRawSOH"
    / "configs"
    / "paper_v1"
    / "e1_raw_soh_learning"
)
BENCHMARK_ROOT = CONFIG_ROOT / "benchmark"
ABLATION_ROOT = CONFIG_ROOT / "ablation"

DOMAIN_CONFIGS = {
    "xjtu": (
        "raw_mamba_xjtu.json",
        "raw_ours_no_cycle_aux_xjtu.json",
    ),
    "mit": (
        "raw_mamba_mit.json",
        "raw_ours_no_cycle_aux_mit.json",
    ),
    "smarthealth_lishen40": (
        "raw_mamba_smarthealth_lishen40.json",
        "raw_ours_no_cycle_aux_smarthealth_lishen40.json",
    ),
    "smarthealth_catl280": (
        "raw_mamba_smarthealth_catl280.json",
        "raw_ours_no_cycle_aux_smarthealth_catl280.json",
    ),
    "smarthealth_eve280": (
        "raw_mamba_smarthealth_eve280.json",
        "raw_ours_no_cycle_aux_smarthealth_eve280.json",
    ),
}


class E1RawOursNoCycleAuxConfigTest(unittest.TestCase):
    def test_each_domain_changes_only_the_declared_ablation_contract(self):
        for domain_id, (benchmark_name, ablation_name) in DOMAIN_CONFIGS.items():
            with self.subTest(domain_id=domain_id):
                benchmark = load_config(BENCHMARK_ROOT / benchmark_name)
                ablation = load_config(ABLATION_ROOT / ablation_name)

                expected_experiment = copy.deepcopy(benchmark["experiment"])
                expected_experiment.update(
                    {
                        "name": f"E1_RawOurs_{domain_id}_wo_cycle_auxiliary",
                        "task": "e1_raw_soh_learning_cycle_auxiliary_ablation",
                    }
                )
                self.assertEqual(ablation["experiment"], expected_experiment)

                expected_output = copy.deepcopy(benchmark["output"])
                expected_output["model_id"] = "RawOurs-noCycleAux"
                self.assertEqual(ablation["output"], expected_output)

                expected_model = copy.deepcopy(benchmark["model"])
                expected_model["use_cycle_prediction"] = False
                expected_model["use_predicted_cycle_for_soh"] = False
                self.assertEqual(ablation["model"], expected_model)

                expected_train = copy.deepcopy(benchmark["train"])
                expected_train["lambda_cycle"] = 0.0
                expected_train["cycle_loss_mode"] = "disabled"
                self.assertEqual(ablation["train"], expected_train)

                expected_data = copy.deepcopy(benchmark["data"])
                expected_data["cycle_target_scope"] = "not_used"
                expected_data["preprocessed_cache"] = {
                    "enabled": True,
                    "directory": ".cache/unified_cccv",
                    "rebuild": False,
                }
                self.assertEqual(ablation["data"], expected_data)

                self.assertEqual(ablation["normalization"], benchmark["normalization"])
                self.assertEqual(ablation["eval"], benchmark["eval"])
                self.assertEqual(ablation["debug"], benchmark["debug"])
                self.assertEqual(ablation["status"], "runnable")


if __name__ == "__main__":
    unittest.main()
