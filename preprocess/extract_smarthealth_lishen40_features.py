#!/usr/bin/env python3
"""Feature extraction entry point for canonical SmartHealth LISHEN RAW only."""

from smarthealth_common import LISHEN40_CONFIG, cli_main_features


if __name__ == "__main__":
    raise SystemExit(cli_main_features(LISHEN40_CONFIG))
