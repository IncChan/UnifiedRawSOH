from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.paper_v2.episodic_sampler import SourceEpisodeBuilder  # noqa: E402


def episode_rows() -> list[dict]:
    rows = []
    for domain in ("d1", "d2", "d3", "d4"):
        for strategy in ("s1", "s2"):
            for cell in ("c1", "c2"):
                for cycle in (1, 2):
                    rows.append(
                        {
                            "domain_id": domain,
                            "strategy_group": strategy,
                            "battery_id": f"{domain}-{cell}",
                            "cycle_id": cycle,
                        }
                    )
    return rows


class EpisodicDGTest(unittest.TestCase):
    def test_dataset_episode_holds_out_one_whole_domain(self):
        rows = episode_rows()
        builder = SourceEpisodeBuilder(rows, ["d1", "d2", "d3", "d4"], seed=5)
        episode = builder.sample_episode("dataset_level")
        target_domains = {rows[index]["domain_id"] for index in episode.pseudo_target_indices}
        meta_domains = {rows[index]["domain_id"] for index in episode.meta_train_indices}
        self.assertEqual(target_domains, set(episode.pseudo_target_domains))
        self.assertEqual(len(target_domains), 1)
        self.assertNotIn(next(iter(target_domains)), meta_domains)
        self.assertTrue(episode.audit["source_only"])
        self.assertTrue(episode.audit["cell_disjoint"])

    def test_strategy_episode_is_cell_disjoint_and_can_expand(self):
        rows = episode_rows()
        # Add a cell appearing in two strategies to exercise the stricter rule.
        rows.extend(
            [
                {"domain_id": "d1", "strategy_group": strategy, "battery_id": "shared", "cycle_id": cycle}
                for strategy in ("s1", "s2")
                for cycle in (1, 2)
            ]
        )
        builder = SourceEpisodeBuilder(rows, ["d1", "d2", "d3", "d4"], seed=13)
        episode = builder.sample_episode("strategy")
        meta_cells = {
            (rows[index]["domain_id"], rows[index]["battery_id"])
            for index in episode.meta_train_indices
        }
        target_cells = {
            (rows[index]["domain_id"], rows[index]["battery_id"])
            for index in episode.pseudo_target_indices
        }
        self.assertFalse(meta_cells & target_cells)
        self.assertTrue(episode.audit["cell_disjoint"])
        selected = set(episode.audit["selection"]["selected_environment_indices"])
        self.assertTrue(selected <= set(episode.pseudo_target_indices))

    def test_target_or_undeclared_domain_cannot_enter_source_builder(self):
        rows = episode_rows() + [
            {"domain_id": "target", "condition": "s", "battery_id": "x", "cycle_id": 1}
        ]
        with self.assertRaisesRegex(ValueError, "undeclared domains"):
            SourceEpisodeBuilder(rows, ["d1", "d2", "d3", "d4"])

    def test_episode_draw_is_reproducible(self):
        rows = episode_rows()
        left = SourceEpisodeBuilder(rows, ["d1", "d2", "d3", "d4"], seed=17)
        right = SourceEpisodeBuilder(rows, ["d1", "d2", "d3", "d4"], seed=17)
        self.assertEqual(left.sample_episode("dataset").as_dict(), right.sample_episode("dataset").as_dict())


if __name__ == "__main__":
    unittest.main()
