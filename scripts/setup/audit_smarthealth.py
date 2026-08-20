#!/usr/bin/env python3
"""Audit the real SmartHealth source without copying or preprocessing it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from UnifiedRawSOH.datasets.smarthealth import audit_smarthealth_source


DEFAULT_SMARTHEALTH_ROOT = Path(
    os.environ.get("SMARTHEALTH_ROOT", "/data1/chenyanxi/lb_project/datasets/SmartHealth")
)


def main():
    parser = argparse.ArgumentParser("Audit SmartHealth source schema")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_SMARTHEALTH_ROOT)
    parser.add_argument("--domain_id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit_smarthealth_source(args.data_root, domain_id=args.domain_id)
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
