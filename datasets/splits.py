"""Split-file and mixed-protocol utilities.

The Paper-v1 main protocol uses mixed train/validation cycles with an
independent battery test set.  Split files may provide either an explicit
test-battery list or a rule such as a legacy source ``cell_id % 5 == 0`` or a
canonical physical-cell ``physical_id % 5 == 0``.  Rule-based selection is
resolved against the batteries actually observed at runtime, so adding valid
source batteries does not silently discard them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROLES = ("train", "val", "test")


def _cell_id_from_battery_id(value):
    """Extract a source-cell or canonical physical suffix for modulo rules."""

    match = re.search(r"(?:battery|cell)-0*(\d+)$", str(value))
    if match is not None:
        return int(match.group(1))
    # MIT v2 identities are intentionally ``mit_p###`` rather than a source
    # filename suffix.  Match only a standalone underscore-delimited p token
    # so unrelated IDs cannot silently enter a physical modulo rule.
    match = re.search(r"(?:^|_)p0*(\d+)$", str(value))
    return None if match is None else int(match.group(1))


def load_split_spec(path):
    """Read a dataset split specification without applying dataset logic."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Split file {path} must contain a JSON object")
    if payload.get("split_status") == "manual_confirmation_required":
        conditions = sorted(
            str(item)
            for item in dict(payload.get("manual_confirmation_conditions", {}))
        )
        raise ValueError(
            f"Split file {path} requires manual confirmation before training"
            + (f"; affected conditions={conditions}" if conditions else "")
        )
    return payload


def get_development_protocol(split_spec):
    """Return the JSON-owned development protocol with compatibility defaults."""

    protocol = split_spec.get("development_split", split_spec.get("protocol", {})) or {}
    return {
        "mode": str(protocol.get("mode", "mixed_cycle")),
        "scope": str(protocol.get("scope", "single_domain_pool")),
        "val_ratio": float(protocol.get("val_ratio", 0.2)),
        "random_state": int(protocol.get("random_state", protocol.get("val_random_state", 420))),
        "train_val_battery_overlap_expected": bool(
            protocol.get("train_val_battery_overlap_expected", True)
        ),
    }


def resolve_test_batteries(split_spec, observed_battery_ids=None, condition=None):
    """Resolve test IDs from explicit JSON assignments or a generic rule."""

    explicit_by_condition = split_spec.get("test_batteries_by_condition")
    if explicit_by_condition is not None:
        if condition is None:
            explicit = [
                item
                for values in explicit_by_condition.values()
                for item in values
            ]
        else:
            key = str(condition)
            if key not in explicit_by_condition:
                raise ValueError(
                    f"Split spec has no test-battery assignment for condition {key!r}"
                )
            explicit = list(explicit_by_condition[key])
    else:
        explicit = list(split_spec.get("test_batteries", []))

    rule = split_spec.get("test_rule", {}) or {}
    observed = (
        None
        if observed_battery_ids is None
        else sorted({str(item) for item in observed_battery_ids})
    )
    if rule.get("type") in {"cell_id_modulo", "physical_id_modulo"} and observed is not None:
        modulus = int(rule["modulus"])
        remainder = int(rule.get("remainder", 0))
        if modulus <= 0:
            raise ValueError("Modulo split modulus must be positive")
        selected = [
            battery_id
            for battery_id in observed
            if (
                _cell_id_from_battery_id(battery_id) is not None
                and _cell_id_from_battery_id(battery_id) % modulus == remainder
            )
        ]
    else:
        selected = sorted({str(item) for item in explicit})
        if observed is not None:
            selected = sorted(set(selected) & set(observed))
    if observed is not None and not selected:
        raise ValueError(
            "Split spec selected no observed test batteries"
            + (f" for condition {condition!r}" if condition is not None else "")
        )
    if observed is None and not selected:
        raise ValueError("Split spec has no test-battery assignment")
    return selected


def load_test_batteries(path, observed_battery_ids=None, condition=None):
    """Load or derive the independent test batteries from a split spec.

    When a split file contains ``test_rule.type == 'cell_id_modulo'`` and
    observed IDs are supplied, the rule is applied to all observed IDs.  The
    JSON list remains an auditable snapshot of the current inventory, but it
    is not used to truncate a larger future inventory.
    """

    return resolve_test_batteries(
        load_split_spec(path),
        observed_battery_ids=observed_battery_ids,
        condition=condition,
    )


def load_invalid_cycles(path):
    """Return normalized ``(battery_id, cycle_id)`` invalid-cycle entries."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    invalid = []
    for item in payload.get("invalid_cycles", []):
        if "battery_id" not in item or "cycle_id" not in item:
            raise ValueError(f"Invalid cycle entry in {path}: {item!r}")
        invalid.append({
            "battery_id": str(item["battery_id"]),
            "cycle_id": int(item["cycle_id"]),
            "reason": str(item.get("reason", "")),
        })
    return invalid


def load_battery_roles(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    roles = payload.get("roles", payload)
    missing = [role for role in ROLES if role not in roles]
    if missing:
        raise ValueError(f"Split file {path} is missing roles: {missing}")
    normalized = {role: sorted(str(item) for item in roles[role]) for role in ROLES}
    all_ids = [item for role in ROLES for item in normalized[role]]
    if len(all_ids) != len(set(all_ids)):
        duplicates = sorted({item for item in all_ids if all_ids.count(item) > 1})
        raise ValueError(f"Split file {path} has overlapping battery roles: {duplicates}")
    return normalized


def apply_battery_roles(records, roles, strict=True):
    """Return ``train/val/test`` records according to fixed battery roles."""

    roles = {role: set(str(item) for item in roles[role]) for role in ROLES}
    known = set().union(*roles.values())
    observed = {str(record["battery_id"]) for record in records}
    if strict:
        unknown = sorted(observed - known)
        missing = sorted(known - observed)
        if unknown:
            raise ValueError(f"Observed batteries are absent from the split definition: {unknown}")
        if missing:
            raise ValueError(f"Split definition contains unavailable batteries: {missing}")
    output = {role: [] for role in ROLES}
    for record in records:
        battery_id = str(record["battery_id"])
        for role in ROLES:
            if battery_id in roles[role]:
                item = dict(record)
                item["split"] = role
                output[role].append(item)
                break
        else:
            if strict:
                raise ValueError(f"No role assigned to battery {battery_id!r}.")
    if strict and any(not output[role] for role in ROLES):
        raise ValueError(
            "Fixed split must contain records for train, val, and test; got "
            + repr({role: len(output[role]) for role in ROLES})
        )
    return output


def split_mixed_cycle_records(records, split_spec, condition=None):
    """Apply a JSON-defined mixed-cycle split to one condition/domain."""

    protocol = get_development_protocol(split_spec)
    if protocol["mode"] != "mixed_cycle":
        raise ValueError(
            f"Expected a mixed_cycle split spec, got {protocol['mode']!r}"
        )
    records = list(records)
    observed_batteries = [record["battery_id"] for record in records]
    test_batteries = set(
        resolve_test_batteries(
            split_spec,
            observed_battery_ids=observed_batteries,
            condition=condition,
        )
    )
    pool = [
        record for record in records
        if str(record["battery_id"]) not in test_batteries
    ]
    test = [
        record for record in records
        if str(record["battery_id"]) in test_batteries
    ]
    if len(pool) < 2 or not test:
        raise ValueError("Mixed-cycle split produced an empty development pool or test set")

    import numpy as np

    permutation = np.random.RandomState(protocol["random_state"]).permutation(len(pool))
    n_val = max(1, int(np.ceil(protocol["val_ratio"] * len(pool))))
    n_val = min(n_val, len(pool) - 1)
    val_indices = permutation[:n_val]
    train_indices = permutation[n_val:]
    return {
        "train": [pool[int(index)] for index in train_indices],
        "val": [pool[int(index)] for index in val_indices],
        "test": test,
    }


def split_records_from_spec(records, split_spec, split_file=None):
    """Split records using only the supplied dataset JSON specification."""

    if split_spec.get("split_status") == "manual_confirmation_required":
        raise ValueError(
            "Cannot split records from a SmartHealth specification that requires manual confirmation"
        )
    records = list(records)
    if "roles" in split_spec:
        output = apply_battery_roles(records, split_spec["roles"])
        return output, {
            "validation_split_mode": "battery",
            "validation_split_scope": "fixed_roles",
            "roles": split_spec["roles"],
            "test_batteries": sorted({
                str(item["battery_id"]) for item in output["test"]
            }),
            "test_battery_rule": split_spec.get("name", "dataset_split_spec"),
            "split_file": str(split_file) if split_file is not None else None,
        }

    protocol = get_development_protocol(split_spec)
    scope = protocol["scope"]
    if scope.endswith("_then_pool"):
        grouped = {}
        for record in records:
            grouped.setdefault(str(record["condition"]), []).append(record)
        if not grouped:
            raise ValueError("Cannot apply a per-condition split to empty records")
        output = {"train": [], "val": [], "test": []}
        per_condition = {}
        for condition in sorted(grouped):
            current = split_mixed_cycle_records(
                grouped[condition], split_spec, condition=condition
            )
            for split_name in output:
                output[split_name].extend(current[split_name])
            per_condition[condition] = {
                "record_counts": {name: len(current[name]) for name in current},
                "battery_counts": {
                    name: len({str(item["battery_id"]) for item in current[name]})
                    for name in current
                },
                "train_batteries": sorted({
                    str(item["battery_id"]) for item in current["train"]
                }),
                "val_batteries": sorted({
                    str(item["battery_id"]) for item in current["val"]
                }),
                "test_batteries": sorted({
                    str(item["battery_id"]) for item in current["test"]
                }),
                "train_val_battery_overlap": sorted(
                    {str(item["battery_id"]) for item in current["train"]}
                    & {str(item["battery_id"]) for item in current["val"]}
                ),
            }
        scope_info = scope
    else:
        output = split_mixed_cycle_records(records, split_spec)
        per_condition = None
        scope_info = scope

    metadata = {
        "validation_split_mode": protocol["mode"],
        "validation_split_scope": scope_info,
        "val_ratio": protocol["val_ratio"],
        "val_random_state": protocol["random_state"],
        "train_val_battery_overlap_expected": protocol[
            "train_val_battery_overlap_expected"
        ],
        "test_battery_rule": split_spec.get("name", "dataset_split_spec"),
        "test_batteries": sorted({
            str(item["battery_id"]) for item in output["test"]
        }),
        "split_file": str(split_file) if split_file is not None else None,
    }
    if per_condition is not None:
        metadata["per_condition"] = per_condition
    return output, metadata
