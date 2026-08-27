#!/usr/bin/env python3
"""Independent Paper-v2 Base/Dense/MoE training entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.trainers.paper_v2.config_contract import (  # noqa: E402
    validate_data_readiness,
    validate_v2_config,
)
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("UnifiedRawSOH Paper-v2 RawMamba adapters")
    # These overrides are deliberately process-local: launcher jobs can use
    # different seeds/output runtimes without modifying the source JSON.
    parser.add_argument("--config", required=True, help="Paper-v2 JSON config (may use base_config inheritance).")
    parser.add_argument(
        "--backend_override",
        choices=("mamba_ssm.Mamba", "torch_reference"),
        default=None,
        help="留空使用 config 的正式 backend；torch_reference 仅用于 CPU bounded smoke。",
    )
    parser.add_argument("--device_override", default=None, help="例如 cuda:0；CPU smoke 使用 cpu。")
    parser.add_argument("--seed", type=int, default=None, help="覆盖当前 child process 的随机种子。")
    parser.add_argument("--output_root", default=None, help="覆盖输出根目录；每个 seed 仍写入 Paper-v2 namespace。")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练 epoch 数；bounded smoke 通常设为 1。")
    parser.add_argument("--patience", type=int, default=None, help="覆盖 early-stopping patience。")
    parser.add_argument("--debug_num_samples", type=int, default=None, help="每个 split 截断到少量样本，仅用于 smoke/debug。")
    parser.add_argument("--run_time", default=None, help="覆盖运行批次名；改实验参数后建议使用新名称。")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="只验证配置，不加载训练数据、不训练、不写输出；launcher dry-run 使用此选项。",
    )
    parser.add_argument(
        "--validate_data_readiness",
        action="store_true",
        help="配合 --validate_only 检查声明的数据目录和 split 文件是否存在。",
    )
    parser.add_argument(
        "--skip_data_readiness",
        action="store_true",
        help="仅供 synthetic/unit-test 使用；真实训练应保留默认的数据 readiness 检查。",
    )
    return parser.parse_args()


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
    if args.output_root is not None:
        config.setdefault("experiment", {})["output_root"] = args.output_root
    if args.run_time is not None:
        config.setdefault("experiment", {})["run_time"] = args.run_time
    report = validate_v2_config(config, require_runnable=True)
    if args.validate_only:
        result = {"status": "valid", "config": str(args.config), "contract": report}
        if args.validate_data_readiness:
            result["data_readiness"] = validate_data_readiness(config, PROJECT_ROOT)
        print(json.dumps(result, indent=2))
        return 0
    from UnifiedRawSOH.trainers.paper_v2.seen_domain import train_from_config

    result = train_from_config(
        config,
        project_root=str(PROJECT_ROOT),
        backend_override=args.backend_override,
        device_override=args.device_override,
        check_data=not args.skip_data_readiness,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
