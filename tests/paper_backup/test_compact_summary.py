from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from UnifiedRawSOH.evaluation.paper_backup.aggregation import metrics_from_rows


def _summary_module():
    path = REPO_ROOT / "scripts/paper_backup/summarize_results.py"
    spec = importlib.util.spec_from_file_location("paper_backup_compact_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metrics_include_battery_macro_mape():
    rows = [
        {"y_true": 1.0, "y_pred": 0.9, "battery_id": "a", "strategy_id": "s"},
        {"y_true": 1.0, "y_pred": 1.0, "battery_id": "a", "strategy_id": "s"},
        {"y_true": 1.0, "y_pred": 0.8, "battery_id": "b", "strategy_id": "s"},
    ]
    metrics = metrics_from_rows(rows)
    assert math.isclose(metrics["battery_macro"]["mape"], 0.125)
    assert math.isclose(metrics["strategy_macro"]["mape"], 0.1)


def test_latest_completed_run_is_selected_without_runtime_filter():
    module = _summary_module()
    common = {
        "experiment_id": "e1_main_estimation",
        "model_id": "Ours",
        "data_id": "xjtu",
        "seed": 42,
    }
    runs = [
        {**common, "modified_ns": 10, "run_dir": "runtime_old/seed_42"},
        {**common, "modified_ns": 20, "run_dir": "runtime_new/seed_42"},
    ]
    selected, duplicates = module.select_latest_runs(runs, "e1_main_estimation")
    assert next(iter(selected.values()))["run_dir"] == "runtime_new/seed_42"
    assert duplicates[0]["ignored"] == ["runtime_old/seed_42"]


def test_seed_aggregation_uses_sample_standard_deviation():
    module = _summary_module()
    rows = []
    for seed, value in ((42, 1.0), (52, 2.0), (62, 3.0)):
        rows.append(
            {
                "experiment": "E1",
                "dataset": "xjtu",
                "strategy": "all",
                "model": "Ours",
                "data_id": "xjtu",
                "seed": seed,
                **{metric: value for metric in module.METRIC_COLUMNS},
            }
        )
    summary = module.aggregate_seed_rows(rows)[0]
    assert summary["seed_count"] == 3
    assert summary["mape_percent_mean"] == 2.0
    assert summary["mape_percent_std"] == 1.0


def test_e3_incomplete_matrix_writes_status_only(tmp_path):
    module = _summary_module()
    output_dir = tmp_path / "summaries"
    status = module.summarize_experiment(
        "e3", root=tmp_path / "empty-results", output_dir=output_dir, seeds=(42,)
    )
    assert status["status"] == "waiting_for_strategy_specific_runs"
    assert status["formal_summary_written"] is False
    assert (output_dir / "e3_status.json").is_file()
    assert not (output_dir / "e3_metrics_mean_std.csv").exists()


def test_128x128_summary_matrix_is_registered():
    module = _summary_module()
    expected, config_paths = module.expected_jobs("e1_crate_128x128", (42, 52, 62))
    assert len(config_paths) == 30
    assert len(expected) == 90
    assert {key[0] for key in expected} == {"e1_shared_crate_128x128"}


def test_core3_128x128_summary_matrix_is_registered_for_ten_seeds():
    module = _summary_module()
    expected, config_paths = module.expected_jobs(
        "e1_core3_128x128", (42, 52, 62, 72, 82, 92, 102, 112, 122, 123)
    )
    assert len(config_paths) == 15
    assert len(expected) == 150
    assert {key[0] for key in expected} == {"e1_shared_crate_128x128"}
    assert {key[1] for key in expected} == {
        "PINN4SOH-like-MLP-SharedCRate-128x128",
        "Smaller-Transformer-SharedCRate-128x128",
        "Ours-FullVI-PointBridge-SharedCRate-128x128",
    }
