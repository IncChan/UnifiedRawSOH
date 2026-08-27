from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.paper_v2.hierarchical_sampler import (  # noqa: E402
    HierarchicalReplacementSampler,
)


def synthetic_rows() -> list[dict]:
    rows = []
    for domain, cells in (("d1", ("a",)), ("d2", ("b", "c"))):
        for strategy in ("s1", "s2"):
            for cell in cells:
                for cycle in range(4):
                    rows.append(
                        {
                            "domain_id": domain,
                            "condition": strategy,
                            "battery_id": f"{domain}-{cell}",
                            "cycle_id": cycle,
                        }
                    )
    return rows


class HierarchicalSamplerTest(unittest.TestCase):
    def test_seed_epoch_reproducibility(self):
        rows = synthetic_rows()
        sampler = HierarchicalReplacementSampler(rows, num_samples=100, seed=19)
        first = list(sampler)
        second = list(sampler)
        self.assertEqual(first, second)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))

    def test_four_level_selection_is_approximately_balanced(self):
        rows = synthetic_rows()
        sampler = HierarchicalReplacementSampler(rows, num_samples=4000, seed=23)
        indices = list(sampler)
        domain_counts = Counter(rows[index]["domain_id"] for index in indices)
        strategy_counts = Counter(
            (rows[index]["domain_id"], rows[index]["condition"]) for index in indices
        )
        cell_counts = Counter(
            (rows[index]["domain_id"], rows[index]["battery_id"]) for index in indices
        )
        self.assertLess(abs(domain_counts["d1"] - domain_counts["d2"]), 300)
        self.assertLess(max(strategy_counts.values()) - min(strategy_counts.values()), 300)
        for domain in ("d1", "d2"):
            values = [count for (sampled_domain, _), count in cell_counts.items() if sampled_domain == domain]
            self.assertLess(max(values) - min(values), 300)
        audit = sampler.audit()
        self.assertEqual(audit["inventory"]["domain_count"], 2)
        self.assertEqual(sum(audit["sampled_counts"]["domain"].values()), 4000)

    def test_missing_strategy_is_a_hard_error(self):
        rows = [{"domain_id": "d", "battery_id": "c", "cycle_id": 1}]
        with self.assertRaisesRegex(ValueError, "strategy_group"):
            HierarchicalReplacementSampler(rows)


if __name__ == "__main__":
    unittest.main()
