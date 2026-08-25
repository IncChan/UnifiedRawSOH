"""GPU scheduling contracts for one-cell head-only launcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

from UnifiedRawSOH.trainers.one_cell_launcher import (
    assign_jobs_round_robin,
    execute_jobs,
)


class _ImmediateProcess:
    active = {}
    maximum = {}

    def __init__(self, command, cwd, env, stdout, stderr, text):
        del command, cwd, stdout, stderr, text
        self.gpu = env["CUDA_VISIBLE_DEVICES"]
        self.done = False
        self.active[self.gpu] = self.active.get(self.gpu, 0) + 1
        self.maximum[self.gpu] = max(
            self.maximum.get(self.gpu, 0),
            self.active[self.gpu],
        )

    def poll(self):
        if not self.done:
            self.active[self.gpu] -= 1
            self.done = True
        return 0


class OneCellLauncherTest(unittest.TestCase):
    def test_round_robin_assignment_is_deterministic(self):
        jobs = [{"job_id": str(index)} for index in range(7)]
        assign_jobs_round_robin(jobs, ["0", "1"])
        self.assertEqual(
            [job["assigned_gpu"] for job in jobs],
            ["0", "1", "0", "1", "0", "1", "0"],
        )

    def test_each_gpu_never_exceeds_its_independent_slot_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for index in range(8):
                output = root / f"job_{index}"
                output.mkdir()
                spec = output / "job_spec.json"
                spec.write_text("{}", encoding="utf-8")
                jobs.append(
                    {
                        "job_id": str(index),
                        "target_domain": "xjtu",
                        "support_group": "2C",
                        "support_choice": str(index),
                        "support_cell": f"cell_{index}",
                        "checkpoint_path": "/checkpoint.pt",
                        "assigned_gpu": str(index % 2),
                        "output_dir": str(output),
                        "job_spec_path": str(spec),
                    }
                )
            settings = {
                "GPU_IDS": ["0", "1"],
                "JOBS_PER_GPU": 2,
                "DRY_RUN": False,
                "RESUME": False,
                "PYTHON_BIN": "python",
                "DEVICE_OVERRIDE": "cuda:0",
                "BACKEND_OVERRIDE": "",
            }
            _ImmediateProcess.active = {}
            _ImmediateProcess.maximum = {}
            with (
                patch(
                    "UnifiedRawSOH.trainers.one_cell_launcher.subprocess.Popen",
                    _ImmediateProcess,
                ),
                patch(
                    "UnifiedRawSOH.trainers.one_cell_launcher._is_completed",
                    return_value=True,
                ),
                patch(
                    "UnifiedRawSOH.trainers.one_cell_launcher.summarize_runtime",
                    return_value={"status": "completed"},
                ),
            ):
                result = execute_jobs(
                    settings,
                    root,
                    {"jobs": jobs},
                )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(_ImmediateProcess.maximum, {"0": 2, "1": 2})


if __name__ == "__main__":
    unittest.main()
