"""Protocol validation for E3/E4 without inventing a transfer trainer.

E3 is deliberately not routed through the E1 trainer: pretraining, target
budget sampling, and target-test evaluation need explicit provenance to avoid
target leakage.  This module gives configs one stable contract now, so the
future trainer can consume it rather than re-encoding domain-specific rules.
"""

from __future__ import annotations

from typing import Any

from UnifiedRawSOH.datasets.domains import canonical_domain_id


_PROTOCOLS = {"leave_one_domain_out", "cross_dataset_holdout", "adaptation"}
_BUDGET_UNITS = {"cycle_fraction", "physical_cell_count"}


def _canonical_domain_list(values: Any, field: str) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"experiment.{field} must be a non-empty list of domain IDs.")
    resolved = [canonical_domain_id(value) for value in values]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"experiment.{field} contains duplicate domains: {resolved}")
    return resolved


def parse_reusability_protocol(config: dict) -> dict:
    """Validate and canonicalize an E3/E4 reusability configuration.

    Returns a compact, JSON-ready protocol description.  It does not load
    data, train a model, or choose a checkpoint.
    """

    experiment = config.get("experiment", {})
    details = config.get("reusability", {})
    protocol = str(details.get("protocol", "")).strip()
    if protocol not in _PROTOCOLS:
        raise ValueError(f"reusability.protocol must be one of {sorted(_PROTOCOLS)}.")

    source_domains = _canonical_domain_list(experiment.get("source_domain_ids"), "source_domain_ids")
    target_value = experiment.get("target_domain_id")
    target_values = experiment.get("target_domain_ids")
    if (target_value is None) == (target_values is None):
        raise ValueError(
            "Specify exactly one of experiment.target_domain_id or experiment.target_domain_ids."
        )
    target_domains = (
        [canonical_domain_id(target_value)]
        if target_value is not None
        else _canonical_domain_list(target_values, "target_domain_ids")
    )
    overlap = sorted(set(source_domains) & set(target_domains))
    if overlap:
        raise ValueError(f"Source and target domains must be disjoint; overlap={overlap}")

    result = {
        "protocol": protocol,
        "evaluation": str(details.get("evaluation", "zero_shot")),
        "source_domain_ids": source_domains,
        "target_domain_ids": target_domains,
    }
    if protocol == "adaptation":
        budget = details.get("target_budget")
        if not isinstance(budget, dict):
            raise ValueError("Adaptation requires reusability.target_budget.")
        unit = str(budget.get("unit", ""))
        value = budget.get("value")
        if unit not in _BUDGET_UNITS:
            raise ValueError(f"target_budget.unit must be one of {sorted(_BUDGET_UNITS)}.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("target_budget.value must be numeric.")
        if unit == "cycle_fraction" and not (0.0 < float(value) <= 1.0):
            raise ValueError("cycle_fraction must be in (0, 1].")
        if unit == "physical_cell_count" and (int(value) != value or int(value) < 1):
            raise ValueError("physical_cell_count must be a positive integer.")
        if details.get("scratch_same_target_budget") is not True:
            raise ValueError("Adaptation requires scratch_same_target_budget=true for a fair comparator.")
        result["target_budget"] = {"unit": unit, "value": int(value) if unit == "physical_cell_count" else float(value)}
        result["scratch_same_target_budget"] = True
    return result
