#!/usr/bin/env python3
"""Feature extraction entry point for canonical SmartHealth CATL RAW only."""

from smarthealth_common import CATL280_CONFIG, cli_main_features


if __name__ == "__main__":
    raise SystemExit(cli_main_features(CATL280_CONFIG))
