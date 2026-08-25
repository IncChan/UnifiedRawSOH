"""Plan and schedule Paper-v1 one-cell head-only jobs from shell settings."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from UnifiedRawSOH.datasets.one_cell import (
    discover_support_inventory,
    select_support_cell,
)
from UnifiedRawSOH.evaluation.paper_v1.summarize_one_cell import summarize_runtime
from UnifiedRawSOH.trainers.one_cell_head_only import load_strict_lodo_model
from UnifiedRawSOH.utils.config import load_config, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = (
    PROJECT_ROOT
    / "UnifiedRawSOH/configs/paper_v1/e3_cross_domain_reusability"
    / "one_cell_head_only"
)
TARGETS = (
    "xjtu",
    "mit",
    "smarthealth_lishen40",
    "smarthealth_catl280",
    "smarthealth_eve280",
)
CONFIGS = {
    target: CONFIG_ROOT / f"one_cell_{target}.json"
    for target in TARGETS
}
CHECKPOINT_ROOT_ENV = {
    "xjtu": "CHECKPOINT_ROOT_XJTU",
    "mit": "CHECKPOINT_ROOT_MIT",
    "smarthealth_lishen40": "CHECKPOINT_ROOT_LISHEN40",
    "smarthealth_catl280": "CHECKPOINT_ROOT_CATL280",
    "smarthealth_eve280": "CHECKPOINT_ROOT_EVE280",
}
ALIASES = {
    "lishen40": "smarthealth_lishen40",
    "catl280": "smarthealth_catl280",
    "eve280": "smarthealth_eve280",
}


def _words(value):
    return [item for item in str(value).replace(",", " ").split() if item]


def _targets(value):
    values = _words(value)
    if values == ["all"]:
        return list(TARGETS)
    resolved = [ALIASES.get(value, value) for value in values]
    unknown = [value for value in resolved if value not in TARGETS]
    if unknown:
        raise ValueError(f"Unknown TARGET_DOMAINS: {unknown}")
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("TARGET_DOMAINS must be non-empty and unique")
    return resolved


def _slug(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    if not cleaned:
        raise ValueError(f"Cannot build output component from {value!r}")
    return cleaned


def assign_jobs_round_robin(jobs, gpu_ids):
    if not gpu_ids:
        raise ValueError("gpu_ids cannot be empty")
    for index, job in enumerate(jobs):
        job["assigned_gpu"] = str(gpu_ids[index % len(gpu_ids)])
    return jobs


def support_choices_for_checkpoint(config, checkpoint_seed):
    mode = config["one_cell"]["support_selection_mode"]
    if mode == "stable_seed_rotation":
        choices = [int(value) for value in config["one_cell"]["support_choices"]]
        if int(checkpoint_seed) not in choices:
            raise ValueError(
                f"Checkpoint seed {checkpoint_seed} has no matching support seed"
            )
        return [int(checkpoint_seed)]
    if mode == "ordered_ab":
        return list(config["one_cell"]["support_choices"])
    raise ValueError(f"Unknown support selection mode: {mode}")


def _runtime_root(settings):
    output_root = Path(settings["OUTPUT_ROOT"])
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    runtime = settings["RUN_TIME"] or datetime.now().strftime("runtime_%y%m%d-%H%M%S")
    if not runtime.startswith("runtime_"):
        runtime = f"runtime_{runtime}"
    base = (
        output_root
        / "Paper-v1/e3_cross_domain_reusability"
        / "RawMamba-noCycleAux-OneCellHeadOnly"
        / runtime
    )
    if base.exists() and not settings["RESUME"]:
        suffix = 1
        while base.with_name(f"{base.name}-{suffix}").exists():
            suffix += 1
        base = base.with_name(f"{base.name}-{suffix}")
    return base


def settings_from_environment():
    gpu_ids = _words(os.environ.get("GPU_IDS", "0"))
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU_IDS must contain unique GPU IDs")
    jobs_per_gpu = int(os.environ.get("JOBS_PER_GPU", "1"))
    if jobs_per_gpu < 1:
        raise ValueError("JOBS_PER_GPU must be positive")
    paired_seeds = [int(value) for value in _words(
        os.environ.get("PAIRED_SEEDS", "42 52 62")
    )]
    if not paired_seeds or len(set(paired_seeds)) != len(paired_seeds):
        raise ValueError("PAIRED_SEEDS must contain unique integer seeds")
    resume = os.environ.get("RESUME", "0") == "1"
    run_time = os.environ.get("RUN_TIME", "")
    if resume and not run_time:
        raise ValueError("RESUME=1 requires RUN_TIME to name an existing runtime")
    return {
        "TARGET_DOMAINS": _targets(os.environ.get("TARGET_DOMAINS", "all")),
        "GPU_IDS": gpu_ids,
        "JOBS_PER_GPU": jobs_per_gpu,
        "PAIRED_SEEDS": paired_seeds,
        "OUTPUT_ROOT": os.environ.get("ONE_CELL_OUTPUT_ROOT", "UnifiedRawSOH/outputs"),
        "RUN_TIME": run_time,
        "DRY_RUN": os.environ.get("DRY_RUN", "0") == "1",
        "RESUME": resume,
        "DEVICE_OVERRIDE": os.environ.get("DEVICE_OVERRIDE", "cuda:0"),
        "BACKEND_OVERRIDE": os.environ.get("BACKEND_OVERRIDE", ""),
        "PYTHON_BIN": os.environ.get("PYTHON_BIN", sys.executable),
    }


def plan_jobs(settings):
    runtime_root = _runtime_root(settings)
    runtime_root.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = {}
    resolved_protocol = {
        "settings": settings,
        "targets": {},
    }
    jobs = []
    test_counts = {}

    for target in settings["TARGET_DOMAINS"]:
        checkpoint_root_input = os.environ.get(
            CHECKPOINT_ROOT_ENV[target], ""
        ).strip()
        if not checkpoint_root_input:
            raise ValueError(
                f"{CHECKPOINT_ROOT_ENV[target]} must be set for target {target}"
            )
        checkpoint_root = Path(checkpoint_root_input).expanduser().resolve()
        config = load_config(CONFIGS[target])
        config = copy.deepcopy(config)
        if config["one_cell"]["support_selection_mode"] == "stable_seed_rotation":
            config["one_cell"]["support_seeds"] = settings["PAIRED_SEEDS"]
            config["one_cell"]["support_choices"] = settings["PAIRED_SEEDS"]
        resolved_config_path = runtime_root / "resolved_configs" / f"{target}.json"
        save_json(resolved_config_path, config)

        checkpoint_manifest[target] = {
            "input_root": checkpoint_root_input,
            "resolved_root": str(checkpoint_root),
            "checkpoints": {},
        }
        inventory = discover_support_inventory(config, PROJECT_ROOT, seed=0)
        test_counts[target] = inventory["all_test_sample_count"]
        resolved_protocol["targets"][target] = {
            "config_path": str(resolved_config_path.resolve()),
            "inventory": inventory,
        }
        for checkpoint_seed in settings["PAIRED_SEEDS"]:
            checkpoint_path = (
                checkpoint_root / f"seed_{checkpoint_seed}" / "best.pt"
            )
            model, _, checkpoint_info = load_strict_lodo_model(
                config,
                checkpoint_path,
                target,
                backend_override=settings["BACKEND_OVERRIDE"] or None,
            )
            del model
            checkpoint_manifest[target]["checkpoints"][str(checkpoint_seed)] = {
                "checkpoint_seed": checkpoint_seed,
                **checkpoint_info,
            }
            choices = support_choices_for_checkpoint(
                config, checkpoint_seed
            )
            for support_group in config["one_cell"]["support_groups"]:
                for support_choice in choices:
                    selection = select_support_cell(
                        config,
                        inventory,
                        support_group,
                        support_choice,
                    )
                    choice_slug = _slug(support_choice)
                    group_slug = _slug(support_group)
                    output_dir = (
                        runtime_root
                        / "jobs"
                        / target
                        / f"checkpoint_seed_{checkpoint_seed}"
                        / f"support_{group_slug}"
                        / choice_slug
                    )
                    job_id = (
                        f"{target}::checkpoint_seed_{checkpoint_seed}::"
                        f"{support_group}::{support_choice}"
                    )
                    job = {
                        "job_id": job_id,
                        "target_domain": target,
                        "checkpoint_seed": checkpoint_seed,
                        "support_group": str(support_group),
                        "support_choice": str(support_choice),
                        "support_cell": selection["support_cell"],
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_info["sha256"],
                        "config_path": str(resolved_config_path.resolve()),
                        "output_dir": str(output_dir.resolve()),
                    }
                    jobs.append(job)

    assign_jobs_round_robin(jobs, settings["GPU_IDS"])
    for job in jobs:
        job_path = Path(job["output_dir"]) / "job_spec.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(job_path, job)
        job["job_spec_path"] = str(job_path.resolve())

    save_json(runtime_root / "checkpoint_manifest.json", checkpoint_manifest)
    save_json(runtime_root / "resolved_protocol.json", resolved_protocol)
    manifest = {
        "status": "planned",
        "runtime_root": str(runtime_root),
        "job_count": len(jobs),
        "expected_default_job_count": 117,
        "seed_pairing": (
            "checkpoint seed equals support seed for stable-seed targets; "
            "ordered A/B targets run both choices per checkpoint seed"
        ),
        "test_sample_counts_by_target": test_counts,
        "jobs": jobs,
    }
    save_json(runtime_root / "job_manifest.json", manifest)
    return runtime_root, manifest


def _is_completed(job):
    output = Path(job["output_dir"])
    status_path = output / "status.json"
    metrics_path = output / "metrics_overall.json"
    if not status_path.is_file() or not metrics_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return status.get("status") == "completed"


def _command(settings, job):
    command = [
        settings["PYTHON_BIN"],
        "-m",
        "UnifiedRawSOH.trainers.one_cell_head_only",
        "--job-spec",
        job["job_spec_path"],
        "--device-override",
        settings["DEVICE_OVERRIDE"],
    ]
    if settings["BACKEND_OVERRIDE"]:
        command.extend(["--backend-override", settings["BACKEND_OVERRIDE"]])
    return command


def _print_job(job, prefix="[job]"):
    print(
        f"{prefix} target={job['target_domain']}; "
        f"checkpoint_seed={job['checkpoint_seed']}; "
        f"support_group={job['support_group']}; "
        f"choice={job['support_choice']}; "
        f"support_cell={job['support_cell']}; "
        f"checkpoint={job['checkpoint_path']}; "
        f"gpu={job['assigned_gpu']}; "
        f"output={job['output_dir']}",
        flush=True,
    )


def execute_jobs(settings, runtime_root, manifest):
    jobs = manifest["jobs"]
    if settings["DRY_RUN"]:
        for job in jobs:
            _print_job(job, prefix="[dry-run]")
        save_json(
            runtime_root / "run_status.json",
            {"status": "dry_run", "job_count": len(jobs)},
        )
        return {"status": "dry_run"}

    queues = {gpu: [] for gpu in settings["GPU_IDS"]}
    skipped = []
    for job in jobs:
        if settings["RESUME"] and _is_completed(job):
            skipped.append(job["job_id"])
        else:
            queues[job["assigned_gpu"]].append(job)

    running = {gpu: [] for gpu in settings["GPU_IDS"]}
    failures = []
    completed = list(skipped)
    while any(queues.values()) or any(running.values()):
        for gpu in settings["GPU_IDS"]:
            while (
                queues[gpu]
                and len(running[gpu]) < settings["JOBS_PER_GPU"]
            ):
                job = queues[gpu].pop(0)
                output = Path(job["output_dir"])
                output.mkdir(parents=True, exist_ok=True)
                log_handle = (output / "job.log").open("a", encoding="utf-8")
                environment = dict(os.environ)
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                _print_job(job, prefix="[launch]")
                process = subprocess.Popen(
                    _command(settings, job),
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running[gpu].append((process, log_handle, job))

        made_progress = False
        for gpu in settings["GPU_IDS"]:
            active = []
            for process, log_handle, job in running[gpu]:
                return_code = process.poll()
                if return_code is None:
                    active.append((process, log_handle, job))
                    continue
                log_handle.close()
                made_progress = True
                if return_code == 0 and _is_completed(job):
                    completed.append(job["job_id"])
                    print(f"[completed] {job['job_id']}", flush=True)
                else:
                    failures.append(
                        {
                            "job_id": job["job_id"],
                            "return_code": return_code,
                            "log": str(Path(job["output_dir"]) / "job.log"),
                        }
                    )
                    print(f"[failed] {job['job_id']}", flush=True)
            running[gpu] = active
        if not made_progress and any(running.values()):
            time.sleep(1.0)

    status = "completed" if not failures and len(completed) == len(jobs) else "incomplete"
    payload = {
        "status": status,
        "expected_job_count": len(jobs),
        "completed_job_count": len(completed),
        "resumed_job_ids": skipped,
        "failures": failures,
    }
    save_json(runtime_root / "run_status.json", payload)
    manifest["status"] = status
    manifest["completed_job_count"] = len(completed)
    manifest["failures"] = failures
    save_json(runtime_root / "job_manifest.json", manifest)
    if status != "completed":
        summarize_runtime(runtime_root)
        return payload
    summary = summarize_runtime(runtime_root)
    if summary["status"] != "completed":
        payload["status"] = "incomplete"
        save_json(runtime_root / "run_status.json", payload)
    return payload


def main():
    settings = settings_from_environment()
    runtime_root, manifest = plan_jobs(settings)
    print(f"runtime root: {runtime_root}")
    print(f"targets: {' '.join(settings['TARGET_DOMAINS'])}")
    print(f"GPU IDs: {' '.join(settings['GPU_IDS'])}")
    print(f"jobs per GPU: {settings['JOBS_PER_GPU']}")
    print(f"planned jobs: {manifest['job_count']}")
    result = execute_jobs(settings, runtime_root, manifest)
    print(json.dumps(result, indent=2))
    if result["status"] not in {"completed", "dry_run"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
