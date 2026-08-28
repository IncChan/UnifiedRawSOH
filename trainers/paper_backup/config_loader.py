"""Config loading with explicit, local base-config inheritance."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one JSON config and recursively resolve its ``base_config``."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config root must be an object: {path}")
    base_value = payload.get("base_config")
    if base_value:
        base_path = Path(str(base_value)).expanduser()
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        base = load_config(base_path)
        payload = _deep_merge(base, payload)
    payload.pop("base_config", None)
    payload.setdefault("_config_path", str(path))
    return dict(payload)


__all__ = ["load_config"]
