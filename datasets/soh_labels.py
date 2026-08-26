"""Versioned SOH-label construction shared by all Paper-v2 loaders.

The label code intentionally operates on cycle-level records, before a split,
feature cleaning pass, or raw-window transformation.  That makes the
reference independent of the train/validation/test assignment and prevents a
later-life maximum from leaking into a BOL label.

Paper-v1 callers do not import this module and keep their historical ``soh``
construction.  Paper-v2 callers add ``soh_bol`` while preserving the source
``soh``/``SOH`` fields.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable


BOL_RULE_VERSION = "bol_peak_mean_top5_first100_v1"
BOL_LABEL_MODE = "bol_peak_relative"
BOL_EARLY_WINDOW_SIZE = 100
BOL_TOP_K = 5
BOL_MAD_THRESHOLD = 3.5
_MODIFIED_Z_SCALE = 0.6744897501960817


class BOLReferenceError(ValueError):
    """Raised when a cell cannot produce the formally defined Q_ref."""


def is_bol_label_mode(value: Any) -> bool:
    """Return whether a config/value requests the Paper-v2 BOL label."""

    if isinstance(value, Mapping):
        data = value.get("data", {}) or {}
        experiment = value.get("experiment", {}) or {}
        candidates = (
            value.get("label_mode"),
            value.get("soh_label_mode"),
            data.get("label_mode"),
            data.get("soh_label_mode"),
            data.get("feature_target_mode"),
            data.get("feature_label_mode"),
            experiment.get("label_mode"),
            experiment.get("soh_label_mode"),
        )
        return any(str(item).strip() in {BOL_LABEL_MODE, "bol_soh"} for item in candidates if item is not None)
    return str(value or "").strip() in {BOL_LABEL_MODE, "bol_soh"}


def _plain(value: Any) -> Any:
    """Convert common numpy/path scalar values into JSON-native values."""

    if hasattr(value, "item") and callable(value.item):
        try:
            return _plain(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def serialize_reference_provenance(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep JSON-serializable copy of one reference provenance."""

    return _plain(dict(reference))


def _records_list(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if hasattr(records, "to_dict"):
        records = records.to_dict("records")
    output = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError(f"SOH label records must be mappings, got {type(record)!r}")
        output.append(dict(record))
    return output


def _cell_id(record: Mapping[str, Any]) -> str:
    for key in ("battery_id", "physical_cell_id", "logical_sequence_id", "cell_id", "cell"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise BOLReferenceError("SOH label record has no battery/physical-cell/logical-cell ID")


def _domain_id(record: Mapping[str, Any], explicit: str | None = None) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    for key in ("domain_id", "dataset_id", "domain", "dataset"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def _number(value: Any, *, field: str, cell_id: str, cycle_id: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BOLReferenceError(
            f"Cell {cell_id!r} cycle {cycle_id!r} has non-numeric {field}: {value!r}"
        ) from exc
    return result


def _cycle_id(record: Mapping[str, Any], fallback: int) -> Any:
    for key in ("cycle_id", "cycle", "canonical_cycle_id", "global_cycle"):
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            try:
                numeric = float(value)
                if math.isfinite(numeric) and numeric.is_integer():
                    return int(numeric)
            except (TypeError, ValueError):
                pass
            return str(value)
    return int(fallback)


def _chronology_value(record: Mapping[str, Any], fallback: int) -> tuple[int, float | str, int]:
    """Use an explicit canonical/order field before the public cycle ID."""

    for key in ("canonical_cycle_index", "canonical_cycle_order"):
        value = record.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            numeric = float(value)
            if math.isfinite(numeric):
                return 0, numeric, fallback
        except (TypeError, ValueError):
            continue
    for key in ("cycle_id", "cycle", "global_cycle", "raw_cycle_order_index"):
        value = record.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            numeric = float(value)
            if math.isfinite(numeric):
                return 1, numeric, fallback
        except (TypeError, ValueError):
            return 2, str(value), fallback
    # Canonical SmartHealth source intervals are a final deterministic tie
    # breaker for records without a numeric cycle field.
    for key in ("source_absolute_start_time", "source_absolute_end_time"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return 2, str(value), fallback
    return 3, fallback, fallback


def _looks_smarthealth(domain_id: str, record: Mapping[str, Any]) -> bool:
    domain = str(domain_id).lower()
    return domain.startswith("smarthealth") or any(
        key in record for key in ("label_capacity_Ah", "label_capacity_ah", "label_source")
    )


def _capacity_candidates(domain_id: str, record: Mapping[str, Any]) -> tuple[str, Any]:
    """Resolve the formal source field without silently changing domains."""

    domain = str(domain_id).lower()
    if domain.startswith("xjtu"):
        # XJTU's source schema calls this value SOH although it is Ah-like.
        keys = ("SOH", "soh_raw", "capacity_Ah", "capacity_ah", "capacity")
        canonical_field = "SOH"
    elif domain == "mit" or domain.startswith("mit_"):
        keys = ("capacity_Ah", "capacity_ah")
        canonical_field = "capacity_Ah"
    elif _looks_smarthealth(domain, record):
        keys = ("label_capacity_Ah", "label_capacity_ah")
        canonical_field = "label_capacity_Ah"
    else:
        keys = ("capacity_Ah", "capacity_ah", "label_capacity_Ah", "label_capacity_ah", "SOH", "soh_raw", "capacity")
        canonical_field = keys[0]
    for key in keys:
        if key in record and record[key] is not None and str(record[key]).strip() != "":
            return canonical_field, record[key]
    raise BOLReferenceError(
        f"Cell {_cell_id(record)!r} has no required {canonical_field} capacity field "
        f"for domain {domain_id!r}"
    )


def _smarthealth_direct(record: Mapping[str, Any]) -> bool:
    value = str(
        record.get("label_source", record.get("smarthealth_label_status", record.get("capacity_source", "")))
    ).strip().lower()
    return value in {"calibration_direct", "direct", "calibration-direct"}


def _smarthealth_interpolated(record: Mapping[str, Any]) -> bool:
    value = str(
        record.get("label_source", record.get("smarthealth_label_status", record.get("capacity_source", "")))
    ).strip().lower()
    return value in {"calibration_interpolated", "interpolated", "calibration-interpolated"}


def _normalise_rows(records: list[dict[str, Any]], domain_id: str | None) -> tuple[str, str, list[dict[str, Any]]]:
    if not records:
        raise BOLReferenceError("Cannot build Q_ref from an empty trajectory")
    cell_ids = {_cell_id(record) for record in records}
    if len(cell_ids) != 1:
        raise BOLReferenceError(
            "build_bol_reference accepts one physical cell/logical trajectory at a time; "
            f"received cells={sorted(cell_ids)}"
        )
    cell_id = next(iter(cell_ids))
    resolved_domain = _domain_id(records[0], domain_id)
    rows = []
    for index, record in enumerate(records):
        if _cell_id(record) != cell_id:
            raise BOLReferenceError(f"Cell identity changes inside trajectory: {cell_id!r}")
        field, raw_capacity = _capacity_candidates(resolved_domain, record)
        cycle_id = _cycle_id(record, index)
        try:
            capacity = float(raw_capacity)
        except (TypeError, ValueError):
            capacity = float("nan")
        rows.append(
            {
                "record": record,
                "input_index": index,
                "cell_id": cell_id,
                "domain_id": resolved_domain,
                "cycle_id": cycle_id,
                "chronology": _chronology_value(record, index),
                "capacity": capacity,
                "source_capacity_field": field,
                "smarthealth_direct": _smarthealth_direct(record),
                "smarthealth_interpolated": _smarthealth_interpolated(record),
            }
        )
    rows.sort(key=lambda item: item["chronology"])
    return resolved_domain, cell_id, rows


def _reject_outliers(candidates: list[dict[str, Any]], cell_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = [float(item["capacity"]) for item in candidates]
    if not values:
        return [], []
    median = sorted(values)[len(values) // 2] if len(values) % 2 else (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2.0
    deviations = sorted(abs(value - median) for value in values)
    mad = deviations[len(deviations) // 2] if len(deviations) % 2 else (deviations[len(deviations) // 2 - 1] + deviations[len(deviations) // 2]) / 2.0
    kept = []
    rejected = []
    for item in candidates:
        difference = abs(float(item["capacity"]) - median)
        if mad == 0.0:
            is_outlier = not math.isclose(float(item["capacity"]), median, rel_tol=0.0, abs_tol=1e-12)
            modified_z = float("inf") if is_outlier else 0.0
        else:
            # Paper-v2 names the cutoff directly in MAD units: reject when
            # abs(x - median) > 3.5 * MAD. Keep the conventional modified-z
            # value in provenance as an audit aid, but do not use its 0.6745
            # scale to silently widen the specified threshold.
            mad_score = difference / mad
            modified_z = _MODIFIED_Z_SCALE * mad_score
            is_outlier = mad_score > BOL_MAD_THRESHOLD
        if is_outlier:
            rejected.append(
                {
                    "cell_id": cell_id,
                    "cycle_id": _plain(item["cycle_id"]),
                    "capacity": float(item["capacity"]),
                    "median": float(median),
                    "mad": float(mad),
                    "modified_z": _plain(modified_z),
                    "mad_score": _plain(difference / mad) if mad else _plain(modified_z),
                    "reason": "mad_gt_3.5",
                }
            )
        else:
            kept.append(item)
    return kept, rejected


def build_bol_reference(
    records: Iterable[Mapping[str, Any]],
    *,
    domain_id: str | None = None,
    early_window_size: int = BOL_EARLY_WINDOW_SIZE,
    top_k: int = BOL_TOP_K,
    mad_threshold: float = BOL_MAD_THRESHOLD,
) -> dict[str, Any]:
    """Build one deterministic ``Q_ref`` and its serializable provenance.

    ``records`` must represent one physical cell or logical degradation
    trajectory.  SmartHealth candidates are always direct calibration points;
    interpolation can extend the observed chronology but can never define the
    reference peak.
    """

    if int(early_window_size) <= 0 or int(top_k) <= 0:
        raise ValueError("early_window_size and top_k must be positive")
    if int(top_k) != BOL_TOP_K:
        raise ValueError(f"Paper-v2 rule fixes top_k={BOL_TOP_K}; got {top_k}")
    if not math.isclose(float(mad_threshold), BOL_MAD_THRESHOLD, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"Paper-v2 rule fixes mad_threshold={BOL_MAD_THRESHOLD}; got {mad_threshold}")
    domain, cell_id, rows = _normalise_rows(_records_list(records), domain_id)
    valid_rows = [
        item
        for item in rows
        if math.isfinite(float(item["capacity"])) and float(item["capacity"]) > 0.0
    ]
    if not valid_rows:
        raise BOLReferenceError(f"Cell {cell_id!r} has no finite positive capacity observations")

    smarthealth = _looks_smarthealth(domain, rows[0]["record"])
    initial_window = valid_rows[: int(early_window_size)]
    reference_window = list(initial_window)
    if smarthealth:
        direct_in_initial = sum(item["smarthealth_direct"] for item in initial_window)
        if direct_in_initial < BOL_TOP_K:
            reference_window = []
            direct_count = 0
            for item in valid_rows:
                reference_window.append(item)
                if item["smarthealth_direct"]:
                    direct_count += 1
                    if direct_count >= BOL_TOP_K:
                        break
            if direct_count < BOL_TOP_K:
                raise BOLReferenceError(
                    f"Cell {cell_id!r} has only {direct_count} finite positive SmartHealth "
                    "calibration_direct points; five are required for "
                    f"{BOL_RULE_VERSION}"
                )
        candidates = [item for item in reference_window if item["smarthealth_direct"]]
    else:
        candidates = list(reference_window)

    if len(candidates) < BOL_TOP_K:
        raise BOLReferenceError(
            f"Cell {cell_id!r} has only {len(candidates)} valid early capacity points after "
            f"the reference-window policy; {BOL_TOP_K} are required"
        )
    kept, rejected = _reject_outliers(candidates, cell_id)
    if len(kept) < BOL_TOP_K:
        raise BOLReferenceError(
            f"Cell {cell_id!r}: MAD filtering left {len(kept)} valid points; "
            f"{BOL_TOP_K} are required; rejected_outliers={rejected!r}"
        )
    selected = sorted(kept, key=lambda item: (-float(item["capacity"]), item["chronology"]))[:BOL_TOP_K]
    q_ref = sum(float(item["capacity"]) for item in selected) / float(BOL_TOP_K)
    if not math.isfinite(q_ref) or q_ref <= 0.0:
        raise BOLReferenceError(f"Cell {cell_id!r} produced an invalid Q_ref={q_ref!r}")

    direct_total = sum(item["smarthealth_direct"] for item in valid_rows) if smarthealth else 0
    interpolated_total = sum(item["smarthealth_interpolated"] for item in valid_rows) if smarthealth else 0
    source_field = str(candidates[0]["source_capacity_field"])
    provenance = {
        "domain_id": str(domain),
        "battery_id": str(cell_id),
        "cell_id": str(cell_id),
        "rule_version": BOL_RULE_VERSION,
        "source_capacity_field": source_field,
        "source_capacity_storage_keys": sorted({str(item["source_capacity_field"]) for item in candidates}),
        "Q_ref": float(q_ref),
        "q_ref": float(q_ref),
        # candidate_count is the count available to the MAD/top-k operation;
        # the post-filter count is recorded separately for auditability.
        "candidate_count": int(len(candidates)),
        "valid_candidate_count_after_outlier_filter": int(len(kept)),
        "selected_top5_capacity_values": [float(item["capacity"]) for item in selected],
        "selected_top5_capacity_values_Ah": [float(item["capacity"]) for item in selected],
        "selected_cycle_ids": [_plain(item["cycle_id"]) for item in selected],
        "reference_window_start_cycle": _plain(reference_window[0]["cycle_id"]),
        "reference_window_end_cycle": _plain(reference_window[-1]["cycle_id"]),
        "reference_window_observation_count": int(len(reference_window)),
        "reference_window_initial_size": int(min(len(valid_rows), int(early_window_size))),
        "smarthealth_direct_interpolated_status": {
            "is_smarthealth": bool(smarthealth),
            "direct_count_in_trajectory": int(direct_total),
            "interpolated_count_in_trajectory": int(interpolated_total),
            "direct_count_in_reference_window": int(sum(item["smarthealth_direct"] for item in reference_window)) if smarthealth else 0,
            "interpolated_count_in_reference_window": int(sum(item["smarthealth_interpolated"] for item in reference_window)) if smarthealth else 0,
            "candidate_policy": "calibration_direct_only" if smarthealth else "source_capacity_field",
            "interpolated_points_can_define_q_ref": False if smarthealth else None,
        },
        "smarthealth_label_status": "direct_only_candidates" if smarthealth else "not_applicable",
        "rejected_outliers": rejected,
        "outlier_rule": {
            "method": "median_mad_modified_z",
            "threshold_mad": float(BOL_MAD_THRESHOLD),
        },
    }
    return serialize_reference_provenance(provenance)


def build_bol_references(
    records: Iterable[Mapping[str, Any]],
    *,
    domain_id: str | None = None,
    early_window_size: int = BOL_EARLY_WINDOW_SIZE,
    top_k: int = BOL_TOP_K,
    mad_threshold: float = BOL_MAD_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Build one reference per independent cell in a mixed-domain record set."""

    rows = _records_list(records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_cell_id(row)].append(row)
    return {
        cell_id: build_bol_reference(
            cell_rows,
            domain_id=domain_id,
            early_window_size=early_window_size,
            top_k=top_k,
            mad_threshold=mad_threshold,
        )
        for cell_id, cell_rows in sorted(grouped.items())
    }


def _is_single_reference(reference: Any) -> bool:
    return isinstance(reference, Mapping) and ("Q_ref" in reference or "q_ref" in reference)


def apply_bol_relative_soh(
    records: Iterable[Mapping[str, Any]],
    reference: Mapping[str, Any] | Mapping[str, Mapping[str, Any]] | None = None,
    *,
    domain_id: str | None = None,
    inplace: bool = False,
) -> list[dict[str, Any]]:
    """Add ``soh_bol`` to records without overwriting the source label.

    A single reference may be passed for one cell, or a cell-ID keyed
    reference mapping may be passed for a mixed set.  If omitted, references
    are built with :func:`build_bol_references` using the exact same code path
    as the raw and Feature MLP loaders.
    """

    rows = _records_list(records)
    if not rows:
        return []
    if reference is None:
        references: Mapping[str, Any] = build_bol_references(rows, domain_id=domain_id)
    elif _is_single_reference(reference):
        references = {_cell_id(rows[0]): reference}
    else:
        references = reference
    output = []
    for row in rows:
        cell_id = _cell_id(row)
        if cell_id not in references:
            raise BOLReferenceError(f"No Q_ref provenance supplied for cell {cell_id!r}")
        ref = references[cell_id]
        if not _is_single_reference(ref):
            raise BOLReferenceError(f"Invalid Q_ref provenance for cell {cell_id!r}")
        q_ref = _number(ref.get("Q_ref", ref.get("q_ref")), field="Q_ref", cell_id=cell_id, cycle_id=_cycle_id(row, 0))
        if not math.isfinite(q_ref) or q_ref <= 0.0:
            raise BOLReferenceError(f"Cell {cell_id!r} has invalid Q_ref={q_ref!r}")
        resolved_domain = _domain_id(row, domain_id)
        field, raw_capacity = _capacity_candidates(resolved_domain, row)
        cycle_id = _cycle_id(row, 0)
        capacity = _number(raw_capacity, field=field, cell_id=cell_id, cycle_id=cycle_id)
        if not math.isfinite(capacity) or capacity <= 0.0:
            raise BOLReferenceError(
                f"Cell {cell_id!r} cycle {cycle_id!r} has invalid positive capacity for BOL label: {raw_capacity!r}"
            )
        item = row if inplace else dict(row)
        item["soh_bol"] = float(capacity / q_ref)
        item["soh_label_mode"] = BOL_LABEL_MODE
        item["soh_bol_reference"] = serialize_reference_provenance(ref)
        item["soh_bol_source_capacity_field"] = str(field)
        output.append(item)
    return output


__all__ = [
    "BOL_EARLY_WINDOW_SIZE",
    "BOL_LABEL_MODE",
    "BOL_MAD_THRESHOLD",
    "BOL_RULE_VERSION",
    "BOLReferenceError",
    "apply_bol_relative_soh",
    "build_bol_reference",
    "build_bol_references",
    "is_bol_label_mode",
    "serialize_reference_provenance",
]
