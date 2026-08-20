#!/usr/bin/env python3
"""Feature extraction entry point for canonical SmartHealth EVE RAW only."""

from smarthealth_common import EVE280_CONFIG, cli_main_features


if __name__ == "__main__":
    raise SystemExit(cli_main_features(EVE280_CONFIG))
