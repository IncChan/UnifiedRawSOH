from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.paper_v2.leakage import validate_lodo_provenance  # noqa: E402
from UnifiedRawSOH.trainers.paper_v2.config_contract import validate_v2_config  # noqa: E402
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


class ItemDataset(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class LodoLeakageTest(unittest.TestCase):
    def test_all_new_lodo_folds_pass_static_contract(self):
        paths = sorted(
            path
            for path in REPO_ROOT.glob("configs/paper_v2/e3_lodo_zero_cell/*/lodo_*.json")
            if path.parent.name != "feature_mlp"
        )
        self.assertEqual(len(paths), 20)
        for path in paths:
            with self.subTest(path=path):
                report = validate_v2_config(load_config(path), require_runnable=True)
                self.assertEqual(report["lodo"]["target_train_and_validation_usage"], "forbidden")

    def test_runtime_partitions_are_source_train_val_target_test_only(self):
        config = load_config(REPO_ROOT / "configs/paper_v2/e3_lodo_zero_cell/moe_dg/lodo_xjtu.json")
        rows = lambda domain, split: [
            {
                "domain_id": domain,
                "condition": "s",
                "battery_id": f"{domain}-{split}",
                "cycle_id": 1,
            }
        ]
        loaders = {
            "train": DataLoader(ItemDataset(rows("mit", "train"))),
            "val": DataLoader(ItemDataset(rows("mit", "val"))),
            "test": DataLoader(ItemDataset(rows("xjtu", "test"))),
        }
        report = validate_lodo_provenance(config, loaders=loaders)
        self.assertEqual(report["source_domain_ids"], ["mit", "smarthealth_lishen40", "smarthealth_catl280", "smarthealth_eve280"])
        self.assertEqual(report["target_domain_id"], "xjtu")
        self.assertTrue(report["target_test_only"])

    def test_target_in_source_partition_is_rejected(self):
        config = load_config(REPO_ROOT / "configs/paper_v2/e3_lodo_zero_cell/base_erm/lodo_xjtu.json")
        rows = [{"domain_id": "xjtu", "condition": "s", "battery_id": "x", "cycle_id": 1}]
        loaders = {
            "train": DataLoader(ItemDataset(rows)),
            "val": DataLoader(ItemDataset(rows)),
            "test": DataLoader(ItemDataset(rows)),
        }
        with self.assertRaisesRegex(ValueError, "source partitions"):
            validate_lodo_provenance(config, loaders=loaders)

    def test_contract_rejects_source_target_overlap(self):
        config = load_config(REPO_ROOT / "configs/paper_v2/e3_lodo_zero_cell/base_erm/lodo_xjtu.json")
        broken = copy.deepcopy(config)
        broken["experiment"]["source_domain_ids"] = ["xjtu", "mit"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_v2_config(broken, require_runnable=True)


if __name__ == "__main__":
    unittest.main()
