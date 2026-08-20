#!/usr/bin/env python3
"""Canonical RAW preprocessing entry point for SmartHealth LISHEN 40 Ah only."""

from smarthealth_common import LISHEN40_CONFIG, cli_main_raw


if __name__ == "__main__":
    raise SystemExit(cli_main_raw(LISHEN40_CONFIG))
