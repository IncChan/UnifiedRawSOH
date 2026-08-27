from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.trainers.paper_v2.config_contract import (  # noqa: E402
    build_v2_seed_output_dir,
    validate_v2_config,
)
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


class ConfigContractTest(unittest.TestCase):
    def test_existing_base_compatibility_config_validates(self):
        config = load_config(REPO_ROOT / "configs/paper_v2/e1_single_domain/raw_mamba/xjtu.json")
        report = validate_v2_config(config, require_runnable=True)
        self.assertEqual(report["model_variant"], "base")
        self.assertEqual(report["trainer_variant"], "erm")

    def test_paper_version_output_and_no_cycle_fields_are_strict(self):
        config = load_config(REPO_ROOT / "configs/paper_v2/e2_full_domain/moe_erm/config.json")
        for field, value in (("paper_version", "Paper-v1"), ("lambda_cycle", 1.0)):
            broken = copy.deepcopy(config)
            if field == "paper_version":
                broken["output"][field] = value
            else:
                broken["train"][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_v2_config(broken, require_runnable=True)

    def test_missing_variant_does_not_fallback(self):
        config = load_config(REPO_ROOT / "configs/paper_v2/e2_full_domain/base/config.json")
        broken = copy.deepcopy(config)
        broken["model"].pop("variant")
        with self.assertRaisesRegex(ValueError, "model.variant"):
            validate_v2_config(broken, require_runnable=True)

    def test_v2_output_builder_cannot_write_paper_v1(self):
        config = load_config(REPO_ROOT / "configs/paper_v2/e2_full_domain/base/config.json")
        with self.assertRaisesRegex(ValueError, "Paper-v1"):
            build_v2_seed_output_dir("/tmp/Paper-v1", config, "smoke", 42)
        path = build_v2_seed_output_dir("/tmp/paper_v2_contract", config, "smoke", 42)
        self.assertIn("Paper-v2", path.parts)
        self.assertIn("seed_42", path.parts)


if __name__ == "__main__":
    unittest.main()
