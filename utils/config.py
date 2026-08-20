"""Configuration loading with explicit inheritance for planned experiments."""

from __future__ import annotations

import copy
import json
from pathlib import Path


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path):
    path = Path(path).resolve()
    payload = load_json(path)
    base_path = payload.pop("base_config", None)
    if base_path:
        base = load_config((path.parent / base_path).resolve())
        payload = deep_merge(base, payload)
    payload["_config_path"] = str(path)
    return payload


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")

