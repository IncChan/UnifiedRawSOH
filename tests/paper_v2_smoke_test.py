#!/usr/bin/env python3
"""Run a bounded Paper-v2 CPU/torch_reference smoke without formal training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT / "UnifiedRawSOH"
CONFIG = REPO_ROOT / "configs/paper_v2/e1_single_domain/raw_mamba/xjtu.json"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="paper_v2_smoke_") as output_root:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        command = [
            sys.executable,
            "-m",
            "UnifiedRawSOH.main",
            "--config",
            str(CONFIG),
            "--backend_override",
            "torch_reference",
            "--device_override",
            "cpu",
            "--epochs",
            "1",
            "--patience",
            "1",
            "--debug_num_samples",
            "1",
            "--output_root",
            output_root,
            "--run_time",
            "paper_v2_cpu_smoke",
        ]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode
        payload = json.loads(result.stdout)
        run_dir = Path(payload["run_dir"])
        required = (
            "resolved_config.json",
            "split_info.json",
            "test_metrics.json",
            "metrics_by_cell.csv",
            "metrics_by_group.csv",
            "metrics_by_domain.csv",
            "best.pt",
        )
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Smoke output is missing: {missing}")
        if "Paper-v2" not in run_dir.parts or "Paper-v1" in run_dir.parts:
            raise RuntimeError(f"Smoke output escaped the Paper-v2 namespace: {run_dir}")
        split_info = json.loads((run_dir / "split_info.json").read_text(encoding="utf-8"))
        if split_info["label"]["label_mode"] != "bol_peak_relative":
            raise RuntimeError("Smoke did not use the BOL label mode")
        if split_info["label"]["q_ref_is_model_input"] or split_info["label"]["q_ref_in_normalization"]:
            raise RuntimeError("Smoke exposed Q_ref to model input or normalization")
        metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        if metrics["aggregation"] != "domain_macro_over_group_macro_over_physical_cell_metrics":
            raise RuntimeError("Smoke did not emit hierarchical Paper-v2 metrics")
        print(json.dumps({
            "status": "PASS",
            "backend": "torch_reference",
            "device": "cpu",
            "epochs": 1,
            "debug_num_samples": 1,
            "run_dir": str(run_dir),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
