"""Strategy-specific and family-pooled composition for Paper-Backup E3."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..splits import load_split_spec, split_records_from_spec
from .sequence_views import build_sequence_loaders, load_terminal_records


def canonical_strategy_id(record: Mapping[str, Any]) -> str:
    """Read strategy metadata from the canonical condition field."""

    value = record.get("strategy_id", record.get("condition"))
    if value is None or not str(value).strip():
        raise ValueError(
            "E3 requires explicit canonical strategy metadata; "
            "strategy cannot be inferred from cycle order or filename position."
        )
    return str(value).strip()


def _with_strategy(records: Iterable[dict[str, Any]], strategy_id: str) -> list[dict[str, Any]]:
    output = []
    for record in records:
        copy = dict(record)
        actual = canonical_strategy_id(copy)
        if actual != str(strategy_id):
            raise ValueError(f"Strategy metadata mismatch: expected {strategy_id!r}, got {actual!r}")
        copy["strategy_id"] = actual
        output.append(copy)
    return output


def discover_strategies(records: Iterable[Mapping[str, Any]]) -> list[str]:
    values = sorted({canonical_strategy_id(record) for record in records})
    if not values:
        raise ValueError("No canonical strategies were observed")
    return values


def build_strategy_splits(
    records: Iterable[dict[str, Any]],
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    strategy_ids: Iterable[str] | None = None,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """Split every strategy independently using the same JSON-owned rules."""

    records = list(records)
    requested = [str(value) for value in strategy_ids] if strategy_ids is not None else discover_strategies(records)
    if not requested:
        raise ValueError("E3 strategy_ids cannot be empty")
    split_file = config.get("data", {}).get("split_file") or config.get("experiment", {}).get("split_file")
    if not split_file:
        raise ValueError("E3 requires a split file")
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = (Path(repo_root) / split_path).resolve()
    split_spec = load_split_spec(split_path)
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    audit: dict[str, Any] = {}
    for strategy in requested:
        current = _with_strategy(
            [record for record in records if canonical_strategy_id(record) == strategy],
            strategy,
        )
        if not current:
            raise ValueError(f"Requested E3 strategy has no canonical records: {strategy!r}")
        split, info = split_records_from_spec(current, split_spec, split_file=split_path)
        overlaps = {
            "train_test": sorted({str(x["battery_id"]) for x in split["train"]} & {str(x["battery_id"]) for x in split["test"]}),
            "val_test": sorted({str(x["battery_id"]) for x in split["val"]} & {str(x["battery_id"]) for x in split["test"]}),
        }
        if overlaps["train_test"] or overlaps["val_test"]:
            raise ValueError(f"E3 strategy split leaks test batteries for {strategy}: {overlaps}")
        result[strategy] = split
        audit[strategy] = {
            "split": info,
            "record_counts": {name: len(values) for name, values in split.items()},
            "battery_counts": {name: len({str(x["battery_id"]) for x in values}) for name, values in split.items()},
            "train_val_battery_overlap": sorted({str(x["battery_id"]) for x in split["train"]} & {str(x["battery_id"]) for x in split["val"]}),
            "test_batteries": sorted({str(x["battery_id"]) for x in split["test"]}),
            "strategy_metadata_source": "canonical condition field",
        }
    return result, {
        "strategy_ids": requested,
        "per_strategy": audit,
        "split_file": str(split_path),
    }


def pooled_strategy_splits(
    strategy_splits: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Union strategy splits and prove that the pooled test cohort is exact."""

    pooled = {name: [] for name in ("train", "val", "test")}
    for strategy in sorted(strategy_splits):
        for split_name in pooled:
            pooled[split_name].extend(strategy_splits[strategy][split_name])
    test_union = sorted({str(item["battery_id"]) for split in strategy_splits.values() for item in split["test"]})
    pooled_test = sorted({str(item["battery_id"]) for item in pooled["test"]})
    if test_union != pooled_test:
        raise ValueError(f"Pooled E3 test cohort mismatch: union={test_union}, pooled={pooled_test}")
    all_keys = [(str(item["battery_id"]), int(item["cycle_id"])) for item in pooled["test"]]
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("Pooled E3 test cohort contains duplicate physical cycle identities")
    return pooled, {
        "test_battery_union": test_union,
        "pooled_test_batteries": pooled_test,
        "test_cohort_exact_union": True,
        "record_counts": {name: len(values) for name, values in pooled.items()},
        "strategy_ids_in_model_input": False,
    }


def build_strategy_loaders(
    config: Mapping[str, Any],
    repo_root: str | Path,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one E3 specific or pooled terminal PhaseMamba loader."""

    records, source_info = load_terminal_records(config, Path(repo_root).resolve())
    requested = config.get("experiment", {}).get("strategy_ids")
    strategy_splits, strategy_audit = build_strategy_splits(
        records,
        config,
        repo_root,
        strategy_ids=requested,
    )
    mode = str(config.get("experiment", {}).get("pooling_mode", config.get("experiment", {}).get("mode", "specific")))
    if mode == "specific":
        strategy = str(config.get("experiment", {}).get("strategy_id", ""))
        if strategy not in strategy_splits:
            raise ValueError(f"E3 specific config must select one known strategy; got {strategy!r}")
        selected = strategy_splits[strategy]
        pooled_audit = {
            "mode": "specific",
            "strategy_id": strategy,
            "deployed_model_count": 1,
            "test_batteries": sorted({str(x["battery_id"]) for x in selected["test"]}),
        }
        split_for_loader = selected
    elif mode == "pooled":
        split_for_loader, pooled_audit = pooled_strategy_splits(strategy_splits)
        pooled_audit.update({"mode": "pooled", "deployed_model_count": 1})
    else:
        raise ValueError("E3 pooling_mode must be 'specific' or 'pooled'")
    loaders, sequence_info = build_sequence_loaders(
        config,
        repo_root,
        seed,
        view_id="terminal_phase",
        split_records=split_for_loader,
        strategy_balanced=(mode == "pooled"),
    )
    info = {
        "loader_type": "paper_backup_strategy_pooling",
        "mode": mode,
        "source": source_info,
        "strategy_split": strategy_audit,
        "pooling": pooled_audit,
        "sequence": sequence_info,
        "strategy_metadata_in_forward": False,
    }
    return loaders, info


__all__ = [
    "build_strategy_loaders",
    "build_strategy_splits",
    "canonical_strategy_id",
    "discover_strategies",
    "pooled_strategy_splits",
]
