#!/usr/bin/env python3
"""Canonical RAW preprocessing entry point for SmartHealth EVE 280 Ah only."""

from smarthealth_common import EVE280_CONFIG, cli_main_raw


if __name__ == "__main__":
    raise SystemExit(cli_main_raw(EVE280_CONFIG))
