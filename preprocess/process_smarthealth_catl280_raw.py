#!/usr/bin/env python3
"""Canonical RAW preprocessing entry point for SmartHealth CATL 280 Ah only."""

from smarthealth_common import CATL280_CONFIG, cli_main_raw


if __name__ == "__main__":
    raise SystemExit(cli_main_raw(CATL280_CONFIG))
