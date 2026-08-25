"""Versioned, fingerprinted cache for eagerly preprocessed raw-cycle datasets."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import torch


CACHE_FORMAT_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache_fingerprint(
    *,
    domain_id: str,
    source_files,
    content_files,
    config_payload: dict,
) -> tuple[str, dict]:
    """Fingerprint raw inventory, exact policy files, and preprocessing config.

    Raw CSVs use path/size/mtime rather than full-content hashes so checking a
    cache remains cheap even for multi-gigabyte canonical exports. Small split
    and implementation files are content-hashed to invalidate the cache when
    policy or preprocessing code changes.
    """

    raw_inventory = []
    for value in sorted((Path(item).resolve() for item in source_files), key=str):
        stat = value.stat()
        raw_inventory.append(
            {
                "path": str(value),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    policy_inventory = [
        {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
        for path in sorted(
            {Path(item).resolve() for item in content_files},
            key=str,
        )
    ]
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "domain_id": str(domain_id),
        "source_files": raw_inventory,
        "content_files": policy_inventory,
        "config": copy.deepcopy(config_payload),
    }
    encoded = json.dumps(
        manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), manifest


def resolve_cache_path(
    data_root,
    cache_config: dict,
    domain_id: str,
    fingerprint: str,
) -> Path:
    configured = Path(
        cache_config.get("directory", ".cache/unified_cccv")
    )
    cache_root = configured if configured.is_absolute() else Path(data_root) / configured
    safe_domain = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(domain_id)
    )
    return cache_root / f"{safe_domain}-{fingerprint[:24]}.pt"


def _torch_load(path: Path):
    # These are trusted, locally generated artifacts under the configured
    # dataset cache directory. Explicit weights_only=False is required because
    # the payload contains NumPy arrays and metadata, not only tensors.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        return torch.load(path, map_location="cpu")


def _valid_payload(payload, fingerprint: str, domain_id: str) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("format_version") == CACHE_FORMAT_VERSION
        and payload.get("fingerprint") == fingerprint
        and payload.get("domain_id") == str(domain_id)
        and isinstance(payload.get("datasets"), dict)
        and set(payload["datasets"]) == {"train", "val", "test"}
        and isinstance(payload.get("split_info"), dict)
    )


def _read_cache(path: Path, fingerprint: str, domain_id: str):
    if not path.is_file():
        return None
    try:
        payload = _torch_load(path)
    except Exception:
        # A truncated/incompatible local cache is disposable. The caller holds
        # the per-cache lock and will rebuild it from canonical raw inputs.
        return None
    return payload if _valid_payload(payload, fingerprint, domain_id) else None


def _atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_or_build_cache(
    *,
    cache_path: Path,
    fingerprint: str,
    domain_id: str,
    manifest: dict,
    builder: Callable[[], dict],
    rebuild: bool = False,
) -> tuple[dict, bool]:
    """Load a valid cache or build it once across concurrent seed processes."""

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if not rebuild:
            cached = _read_cache(cache_path, fingerprint, domain_id)
            if cached is not None:
                return cached, True

        built = builder()
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "domain_id": str(domain_id),
            "manifest": manifest,
            "datasets": built["datasets"],
            "split_info": built["split_info"],
        }
        _atomic_save(cache_path, payload)
        return payload, False
