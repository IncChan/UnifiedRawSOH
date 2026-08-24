#!/usr/bin/env python3
"""Run a configured Paper-v1 raw SOH experiment from the repository root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.trainers.c5b_trainer import train_from_config  # noqa: E402
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("UnifiedRawSOH Paper-v1 raw SOH experiments")
    parser.add_argument("--config", required=True, help="Path to a JSON config under UnifiedRawSOH/configs/")
    parser.add_argument("--backend_override", choices=("mamba_ssm.Mamba", "torch_reference"), default=None)
    parser.add_argument("--device_override", default=None, help="Use cpu only for structural smoke tests.")
    parser.add_argument("--seed", type=int, default=None, help="Override train.seed for this process.")
    parser.add_argument("--output_root", default=None, help="Override experiment.output_root for this process.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--debug_num_samples", type=int, default=None)
    parser.add_argument("--run_time", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if config.get("status", "runnable") != "runnable":
        raise SystemExit(
            f"Config {args.config} is marked {config['status']!r}; it is an experiment interface, "
            "not a Paper-v1 runnable path yet."
        )
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.patience is not None:
        config.setdefault("train", {})["patience"] = args.patience
    if args.debug_num_samples is not None:
        config.setdefault("debug", {})["debug_num_samples"] = args.debug_num_samples
    if args.seed is not None:
        config.setdefault("train", {})["seed"] = args.seed
    if args.output_root is not None:
        config.setdefault("experiment", {})["output_root"] = args.output_root
    if args.run_time is not None:
        config.setdefault("experiment", {})["run_time"] = args.run_time
    result = train_from_config(
        config,
        repo_root=PROJECT_ROOT,
        backend_override=args.backend_override,
        device_override=args.device_override,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
