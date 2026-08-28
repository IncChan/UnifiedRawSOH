"""Frozen SmartHealth v6 combined-charge CC/CV boundary detector.

This module is intentionally not imported by the canonical v7 entry points.
It preserves the old point-count policy for comparisons and reproduction.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import numpy as np

from ...smarthealth_common import (
    CHARGE_STEP,
    DISCHARGE_STEP,
    PhaseResult,
    Point,
    duration_seconds,
    events,
    pick_event,
)


LEGACY_POLICY_VERSION = "smarthealth_cccv_calibration_v6"
LEGACY_DEFAULT_MIN_CC_POINTS = 60
LEGACY_DEFAULT_MIN_CV_POINTS = 60
LEGACY_DEFAULT_CC_REFERENCE_MIN_POINTS = 120
LEGACY_DEFAULT_CV_PERSISTENCE_POINTS = 30
LEGACY_CV_MIN_POINTS_BY_DOMAIN_DOD = {
    "smarthealth_eve280": {20: 30, 60: 30, 100: 30},
}


def legacy_min_cv_points(domain_id: str, dod_percent: int) -> int:
    return LEGACY_CV_MIN_POINTS_BY_DOMAIN_DOD.get(domain_id, {}).get(
        int(dod_percent), LEGACY_DEFAULT_MIN_CV_POINTS
    )


def split_combined_charge_v6(
    points: Sequence[Point],
    args: argparse.Namespace,
    *,
    min_cv_points: int | None = None,
) -> PhaseResult:
    """Run the frozen v6 30/60-point persistent-taper detector."""

    min_cv_points = args.min_cv_points if min_cv_points is None else min_cv_points
    charge_events = events(points, CHARGE_STEP)
    discharge_events = events(points, DISCHARGE_STEP)
    if not charge_events:
        return PhaseResult(
            status="invalid",
            reason="no_combined_charge_event",
            discharge_event_count=len(discharge_events),
        )
    selected = pick_event(
        charge_events, "charge_capacity_ah", prefer_capacity_span=False
    )
    assert selected is not None
    event_index, event = selected
    result = PhaseResult(
        status="invalid",
        reason="",
        charge_event_count=len(charge_events),
        discharge_event_count=len(discharge_events),
        charge_event_index=event_index,
        charge_step_id=event[0].step_id,
        charge_points=len(event),
    )
    if len(event) < args.min_cc_points + min_cv_points:
        result.reason = "combined_charge_too_short"
        return result

    current = np.asarray([abs(point.current_a) for point in event], dtype=float)
    voltage = np.asarray([point.voltage_v for point in event], dtype=float)
    times = np.asarray([duration_seconds(point.time_text) for point in event], dtype=float)
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(voltage)):
        result.reason = "nonfinite_charge_current_or_voltage"
        return result
    if (
        not np.all(np.isfinite(times))
        or times.size < 2
        or np.any(np.diff(times) < 0)
        or float(times[-1] - times[0]) <= 0
    ):
        result.reason = "invalid_charge_time"
        return result
    if not np.all(current > 0):
        result.reason = "nonpositive_charge_current"
        return result

    early_count = min(
        len(event),
        max(
            int(args.cc_reference_min_points),
            int(math.ceil(args.cc_reference_fraction * len(event))),
        ),
    )
    cc_reference = float(np.quantile(current[:early_count], args.cc_reference_quantile))
    taper = current <= cc_reference * (1.0 - args.cv_taper_fraction)
    voltage_max = float(np.max(voltage))
    boundary = None
    for index in range(args.min_cc_points, len(event) - min_cv_points + 1):
        if index + args.cv_persistence_points > len(event):
            break
        if not np.all(taper[index : index + args.cv_persistence_points]):
            continue
        if voltage[index] < voltage_max - args.cv_voltage_tolerance_v:
            continue
        boundary = index
        break

    result.cc_current_reference_a = cc_reference
    result.charge_voltage_max_v = voltage_max
    if boundary is None:
        result.reason = "no_persistent_taper_near_charge_voltage_max"
        return result
    result.cc = event[:boundary]
    result.cv = event[boundary:]
    result.inferred_cc_points = len(result.cc)
    result.inferred_cv_points = len(result.cv)
    result.cv_start_source_row_index = result.cv[0].source_row_index
    result.cv_start_voltage_v = float(voltage[boundary])
    result.cv_start_current_a = float(current[boundary])
    if result.inferred_cc_points < args.min_cc_points or result.inferred_cv_points < min_cv_points:
        result.reason = "phase_point_count_below_minimum"
        return result
    result.status = "ok"
    result.reason = "persistent_current_taper_near_charge_voltage_max"
    return result
