"""Pure SmartHealth BOL-reference construction for canonical preprocessing.

This module intentionally has no project, pandas, or torch dependency.  The
family-specific preprocessing entry points can therefore use it before any
model-facing records exist.  Model loaders consume the frozen result and must
not reconstruct it from their (potentially filtered) input rows.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


BOL_RULE_VERSION = "bol_peak_mean_top5_first100_v1"
BOL_REFERENCE_SOURCE = "canonical_source_calibration_provenance"
BOL_REFERENCE_CONTRACT_VERSION = "smarthealth_frozen_bol_reference_v1"
BOL_EARLY_WINDOW_SIZE = 100
BOL_TOP_K = 5
BOL_MAD_THRESHOLD = 3.5
_MODIFIED_Z_SCALE = 0.6744897501960817


class SmartHealthBOLReferenceError(ValueError):
    """Raised when complete source provenance cannot define a frozen Q_ref."""


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _reject_outliers(
    candidates: list[dict[str, Any]], cell_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = [float(item["capacity_Ah"]) for item in candidates]
    center = _median(values)
    mad = _median([abs(value - center) for value in values])
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in candidates:
        capacity = float(item["capacity_Ah"])
        difference = abs(capacity - center)
        if mad == 0.0:
            is_outlier = not math.isclose(
                capacity, center, rel_tol=0.0, abs_tol=1e-12
            )
            mad_score = math.inf if is_outlier else 0.0
        else:
            mad_score = difference / mad
            is_outlier = mad_score > BOL_MAD_THRESHOLD
        if is_outlier:
            rejected.append(
                {
                    "cell_id": str(cell_id),
                    "cycle_id": int(item["cycle_id"]),
                    "capacity_Ah": capacity,
                    "median_Ah": center,
                    "mad_Ah": mad,
                    "mad_score": mad_score if math.isfinite(mad_score) else "Infinity",
                    "modified_z": (
                        _MODIFIED_Z_SCALE * mad_score
                        if math.isfinite(mad_score)
                        else "Infinity"
                    ),
                    "reason": "mad_gt_3.5",
                }
            )
        else:
            kept.append(item)
    return kept, rejected


def build_frozen_smarthealth_bol_reference(
    records: Iterable[Mapping[str, Any]],
    *,
    domain_id: str,
    cell_id: str,
) -> dict[str, Any]:
    """Build Q_ref from complete selected source-cycle provenance.

    Each record represents one chronological source cycle and provides:
    ``cycle_id``, ``capacity_Ah`` (direct calibration capacity or an exported
    interpolated label), and ``calibration_direct``.  Crucially, direct
    calibration records remain present even if their charge trace is not
    eligible for model export.
    """

    rows: list[dict[str, Any]] = []
    seen_cycles: set[int] = set()
    for record in records:
        try:
            cycle_id = int(record["cycle_id"])
            capacity = float(record["capacity_Ah"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SmartHealthBOLReferenceError(
                f"Cell {cell_id!r} has invalid source BOL provenance record: {record!r}"
            ) from exc
        if cycle_id <= 0 or cycle_id in seen_cycles:
            raise SmartHealthBOLReferenceError(
                f"Cell {cell_id!r} has duplicate/non-positive canonical cycle {cycle_id!r}"
            )
        seen_cycles.add(cycle_id)
        if not math.isfinite(capacity) or capacity <= 0.0:
            continue
        rows.append(
            {
                "cycle_id": cycle_id,
                "capacity_Ah": capacity,
                "calibration_direct": bool(record.get("calibration_direct", False)),
                "model_eligible": bool(record.get("model_eligible", False)),
                "label_source": str(record.get("label_source", "")),
            }
        )
    rows.sort(key=lambda item: item["cycle_id"])
    if not rows:
        raise SmartHealthBOLReferenceError(
            f"Cell {cell_id!r} has no finite positive source capacity observations"
        )

    initial_window = rows[:BOL_EARLY_WINDOW_SIZE]
    reference_window = list(initial_window)
    direct_count = sum(item["calibration_direct"] for item in reference_window)
    if direct_count < BOL_TOP_K:
        reference_window = []
        direct_count = 0
        for item in rows:
            reference_window.append(item)
            if item["calibration_direct"]:
                direct_count += 1
                if direct_count >= BOL_TOP_K:
                    break
    if direct_count < BOL_TOP_K:
        raise SmartHealthBOLReferenceError(
            f"Cell {cell_id!r} has only {direct_count} finite positive direct "
            f"calibrations; {BOL_TOP_K} are required for {BOL_RULE_VERSION}"
        )

    candidates = [item for item in reference_window if item["calibration_direct"]]
    kept, rejected = _reject_outliers(candidates, cell_id)
    expanded_after_mad = False
    next_index = len(reference_window)
    while len(kept) < BOL_TOP_K and next_index < len(rows):
        reference_window.append(rows[next_index])
        next_index += 1
        if not reference_window[-1]["calibration_direct"]:
            continue
        expanded_after_mad = True
        candidates = [item for item in reference_window if item["calibration_direct"]]
        kept, rejected = _reject_outliers(candidates, cell_id)
    if len(kept) < BOL_TOP_K:
        raise SmartHealthBOLReferenceError(
            f"Cell {cell_id!r}: MAD filtering left {len(kept)} direct calibrations; "
            f"{BOL_TOP_K} are required; rejected_outliers={rejected!r}"
        )

    selected = sorted(
        kept, key=lambda item: (-float(item["capacity_Ah"]), int(item["cycle_id"]))
    )[:BOL_TOP_K]
    q_ref = sum(float(item["capacity_Ah"]) for item in selected) / BOL_TOP_K
    if not math.isfinite(q_ref) or q_ref <= 0.0:
        raise SmartHealthBOLReferenceError(
            f"Cell {cell_id!r} produced invalid Q_ref={q_ref!r}"
        )

    return {
        "contract_version": BOL_REFERENCE_CONTRACT_VERSION,
        "domain_id": str(domain_id),
        "battery_id": str(cell_id),
        "cell_id": str(cell_id),
        "rule_version": BOL_RULE_VERSION,
        "reference_source": BOL_REFERENCE_SOURCE,
        "source_capacity_field": "source_calibration_discharge_capacity_Ah",
        "Q_ref": float(q_ref),
        "q_ref": float(q_ref),
        "candidate_count": len(candidates),
        "valid_candidate_count_after_outlier_filter": len(kept),
        "selected_top5_capacity_values_Ah": [
            float(item["capacity_Ah"]) for item in selected
        ],
        "selected_cycle_ids": [int(item["cycle_id"]) for item in selected],
        "selected_model_eligible": [
            bool(item["model_eligible"]) for item in selected
        ],
        "reference_window_start_cycle": int(reference_window[0]["cycle_id"]),
        "reference_window_end_cycle": int(reference_window[-1]["cycle_id"]),
        "reference_window_observation_count": len(reference_window),
        "reference_window_initial_size": min(len(rows), BOL_EARLY_WINDOW_SIZE),
        "reference_window_expanded_after_mad": bool(expanded_after_mad),
        "source_observation_count": len(rows),
        "source_direct_calibration_count": sum(
            item["calibration_direct"] for item in rows
        ),
        "source_model_ineligible_direct_calibration_count": sum(
            item["calibration_direct"] and not item["model_eligible"] for item in rows
        ),
        "candidate_policy": "complete_source_provenance_calibration_direct_only",
        "interpolated_points_can_define_q_ref": False,
        "rejected_outliers": rejected,
        "outlier_rule": {
            "method": "median_mad",
            "threshold_mad": BOL_MAD_THRESHOLD,
        },
    }


__all__ = [
    "BOL_REFERENCE_CONTRACT_VERSION",
    "BOL_REFERENCE_SOURCE",
    "BOL_RULE_VERSION",
    "SmartHealthBOLReferenceError",
    "build_frozen_smarthealth_bol_reference",
]
