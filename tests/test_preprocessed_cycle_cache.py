"""Contracts for the versioned preprocessed raw-cycle cache."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets import loaders as loader_module  # noqa: E402
from UnifiedRawSOH.datasets.loaders import _make_balanced_sampler  # noqa: E402
from UnifiedRawSOH.datasets.preprocessed_cache import (  # noqa: E402
    build_cache_fingerprint,
    load_or_build_cache,
    resolve_cache_path,
)
from UnifiedRawSOH.datasets.xjtu import UnifiedCCCVSampleDataset  # noqa: E402


def _sample(domain_id="xjtu", battery_id="battery-1", cycle_id=1):
    return {
        "cc_signal": np.zeros((4, 2), dtype=np.float32),
        "cv_signal": np.ones((5, 2), dtype=np.float32),
        "cc_mask": np.ones(4, dtype=np.float32),
        "cv_mask": np.ones(5, dtype=np.float32),
        "cc_time": np.arange(4, dtype=np.float32),
        "cv_time": np.arange(5, dtype=np.float32),
        "cc_temperature": np.zeros((4, 2), dtype=np.float32),
        "cv_temperature": np.zeros((5, 2), dtype=np.float32),
        "t0_temperature_norm": np.zeros(1, dtype=np.float32),
        "soh": np.asarray([0.9], dtype=np.float32),
        "soh_raw": 0.9,
        "cycle_life_norm_target": np.asarray([0.0], dtype=np.float32),
        "battery_id": battery_id,
        "dataset_id": domain_id,
        "domain_id": domain_id,
        "condition": "condition",
        "batch_name": "condition",
        "cycle_id": cycle_id,
        "split": "train",
    }


class PreprocessedCycleCacheTest(unittest.TestCase):
    def test_cache_is_built_once_then_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"
            policy = root / "split.json"
            source.write_text("header\nvalue\n", encoding="utf-8")
            policy.write_text('{"split": 1}\n', encoding="utf-8")
            fingerprint, manifest = build_cache_fingerprint(
                domain_id="xjtu",
                source_files=[source],
                content_files=[policy],
                config_payload={"raw_len_cc": 4},
            )
            cache_path = resolve_cache_path(
                root,
                {"directory": ".cache/unified_cccv"},
                "xjtu",
                fingerprint,
            )
            builds = []

            def builder():
                builds.append(True)
                return {
                    "datasets": {
                        "train": {"samples": [_sample()]},
                        "val": {"samples": [_sample()]},
                        "test": {"samples": [_sample()]},
                    },
                    "split_info": {"sample_counts": {"train": 1, "val": 1, "test": 1}},
                }

            first, first_hit = load_or_build_cache(
                cache_path=cache_path,
                fingerprint=fingerprint,
                domain_id="xjtu",
                manifest=manifest,
                builder=builder,
            )
            second, second_hit = load_or_build_cache(
                cache_path=cache_path,
                fingerprint=fingerprint,
                domain_id="xjtu",
                manifest=manifest,
                builder=builder,
            )

            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertEqual(len(builds), 1)
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertTrue(cache_path.is_file())
            self.assertEqual(cache_path.parent, root / ".cache" / "unified_cccv")

    def test_raw_domain_wrapper_reuses_cache_across_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "2C_battery-1.csv").write_text(
                "placeholder\n", encoding="utf-8"
            )
            split_path = root / "split.json"
            split_path.write_text("{}\n", encoding="utf-8")
            config = {
                "experiment": {
                    "domain_id": "xjtu",
                    "split_file": str(split_path),
                    "batches": [],
                },
                "data": {
                    "data_mode": "single_domain",
                    "preprocessed_cache": {
                        "enabled": True,
                        "directory": ".cache/unified_cccv",
                    },
                },
                "normalization": {"mode": "physical_window"},
                "debug": {"debug_num_samples": 0},
            }

            def fake_uncached(_config, _repo_root, seed, _domain_id, _data_root):
                datasets = {
                    name: UnifiedCCCVSampleDataset.from_preprocessed_cache(
                        {
                            "samples": [
                                {
                                    **_sample(cycle_id=index + 1),
                                    "split": name,
                                }
                            ]
                        },
                        data_config={},
                        split_name=name,
                        seed=seed,
                    )
                    for index, name in enumerate(("train", "val", "test"))
                }
                return datasets, {"sample_counts": {name: 1 for name in datasets}}

            with mock.patch.object(
                loader_module,
                "_build_raw_domain_uncached",
                side_effect=fake_uncached,
            ) as build:
                first_datasets, first_info = loader_module._build_raw_domain(
                    config, root, 42, "xjtu", root
                )
                second_datasets, second_info = loader_module._build_raw_domain(
                    config, root, 52, "xjtu", root
                )

            self.assertEqual(build.call_count, 1)
            self.assertFalse(first_info["preprocessed_cache"]["hit"])
            self.assertTrue(second_info["preprocessed_cache"]["hit"])
            self.assertEqual(len(first_datasets["train"]), 1)
            self.assertEqual(len(second_datasets["train"]), 1)

    def test_raw_inventory_change_invalidates_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"
            policy = root / "split.json"
            source.write_text("a\n", encoding="utf-8")
            policy.write_text("{}\n", encoding="utf-8")
            first, _ = build_cache_fingerprint(
                domain_id="xjtu",
                source_files=[source],
                content_files=[policy],
                config_payload={},
            )
            stat = source.stat()
            os.utime(
                source,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
            )
            second, _ = build_cache_fingerprint(
                domain_id="xjtu",
                source_files=[source],
                content_files=[policy],
                config_payload={},
            )
            self.assertNotEqual(first, second)

    def test_cached_dataset_preserves_model_ready_sample(self):
        original = _sample()
        restored = UnifiedCCCVSampleDataset.from_preprocessed_cache(
            {"samples": [original], "skipped": {"too_few_cc_points": 2}},
            data_config={},
            split_name="train",
            seed=52,
        )
        item = restored[0]
        self.assertTrue(torch.equal(item["cc_signal"], torch.from_numpy(original["cc_signal"])))
        self.assertTrue(torch.equal(item["cv_signal"], torch.from_numpy(original["cv_signal"])))
        self.assertEqual(item["battery_id"], original["battery_id"])
        self.assertEqual(restored.skipped["too_few_cc_points"], 2)

    def test_balanced_sampler_uses_cached_metadata(self):
        samples = [
            _sample("d1", "b1", 1),
            _sample("d1", "b1", 2),
            _sample("d1", "b2", 1),
            _sample("d2", "b3", 1),
            _sample("d2", "b3", 2),
        ]
        dataset = UnifiedCCCVSampleDataset.from_preprocessed_cache(
            {"samples": samples},
            data_config={},
            split_name="train",
        )
        sampler = _make_balanced_sampler(
            dataset, "domain_battery_hierarchical"
        )
        np.testing.assert_allclose(
            sampler.weights.numpy(),
            np.asarray([0.25, 0.25, 0.5, 0.5, 0.5]),
        )


if __name__ == "__main__":
    unittest.main()
