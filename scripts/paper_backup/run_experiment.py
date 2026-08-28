#!/usr/bin/env python3
"""Run or validate one isolated Paper-Backup config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_contract import validate_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("UnifiedRawSOH Paper-Backup experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate_only", action="store_true", help="Validate contract without loading data or writing output.")
    parser.add_argument("--check_data", action="store_true", help="Check configured terminal/full roots in addition to JSON/splits.")
    parser.add_argument("--backend_override", choices=("mamba_ssm.Mamba", "torch_reference"), default=None)
    parser.add_argument("--device_override", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--debug_num_samples", type=int, default=None)
    parser.add_argument("--run_time", default=None)
    return parser.parse_args()


def _resolve(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def _data_readiness(config: dict, repo_root: Path) -> dict:
    data = config.get("data", {})
    terminal = _resolve(repo_root, data.get("terminal_data_root", data.get("data_root")))
    full = _resolve(repo_root, data.get("full_data_root"))
    result = {
        "terminal_data_root": str(terminal) if terminal else None,
        "terminal_exists": bool(terminal and terminal.is_dir()),
        "full_data_root": str(full) if full else None,
        "full_exists": bool(full and full.is_dir()),
    }
    split_value = data.get("split_file") or config.get("experiment", {}).get("split_file")
    split = _resolve(repo_root, split_value)
    result["split_file"] = str(split) if split else None
    result["split_exists"] = bool(split and split.is_file())
    if not result["terminal_exists"] and str(config.get("status", "runnable")) == "runnable":
        raise ValueError(f"Configured terminal data root is missing: {terminal}")
    if str(data.get("input_view", "")) == "full_cccv" and not result["full_exists"] and str(config.get("status", "runnable")) == "runnable":
        raise ValueError(f"Configured full data root is missing: {full}")
    if not result["split_exists"] and str(config.get("status", "runnable")) == "runnable":
        raise ValueError(f"Configured split file is missing: {split}")
    return result


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.patience is not None:
        config.setdefault("train", {})["patience"] = int(args.patience)
    if args.debug_num_samples is not None:
        config.setdefault("debug", {})["debug_num_samples"] = int(args.debug_num_samples)
    if args.seed is not None:
        config.setdefault("train", {})["seed"] = int(args.seed)
    contract = validate_config(config, REPO_ROOT, check_files=True)
    result = {
        "status": str(config.get("status", "runnable")),
        "config": str(Path(args.config).resolve()),
        "contract": contract,
    }
    if args.check_data:
        result["data_readiness"] = _data_readiness(config, REPO_ROOT)
    if args.validate_only:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0 if result["status"] in {"runnable", "blocked_by_data"} else 2
    if result["status"] != "runnable":
        result["message"] = "This config is an interface blocked by data; provide a real full point-level source before training."
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 3
    from UnifiedRawSOH.trainers.paper_backup.trainer import train_from_config

    trained = train_from_config(
        config,
        REPO_ROOT,
        seed=args.seed,
        device=args.device_override,
        backend=args.backend_override,
        output_root=args.output_root,
        run_time=args.run_time,
    )
    print(json.dumps(trained, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
