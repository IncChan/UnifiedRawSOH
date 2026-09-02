"""Battery-domain metadata for the Raw -> Unified -> Reusable paper story.

``dataset_id`` remains the adapter/source identity used by historical data and
saved checkpoints.  ``domain_id`` is the paper-level unit: one battery family
can contain multiple C-rate, DOD, or aging conditions without becoming several
domains.  This module intentionally keeps that metadata outside model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BatteryDomainSpec:
    """Stable metadata needed to compose experiments, not model inference."""

    domain_id: str
    paper_alias: str
    source: str
    manufacturer: str
    battery_model: str
    chemistry: str
    nominal_capacity_ah: float | None
    voltage_range_v: tuple[float, float] | None
    operating_conditions: tuple[str, ...]
    data_root: str | None
    feature_data_root: str | None
    adapter_id: str
    split_file: str | None
    normalization: Mapping[str, Any] | None
    availability: str = "available"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def metadata(self) -> dict[str, Any]:
        """JSON-ready metadata for manifests and split diagnostics."""

        return {
            "domain_id": self.domain_id,
            "paper_alias": self.paper_alias,
            "source": self.source,
            "manufacturer": self.manufacturer,
            "battery_model": self.battery_model,
            "chemistry": self.chemistry,
            "nominal_capacity_ah": self.nominal_capacity_ah,
            "voltage_range_v": None if self.voltage_range_v is None else list(self.voltage_range_v),
            "operating_conditions": list(self.operating_conditions),
            "data_root": self.data_root,
            "feature_data_root": self.feature_data_root,
            "adapter_id": self.adapter_id,
            "split_file": self.split_file,
            "normalization": None if self.normalization is None else dict(self.normalization),
            "availability": self.availability,
            "notes": list(self.notes),
        }


class BatteryDomainRegistry:
    """Registry of semantically stable battery domains and paper aliases."""

    def __init__(self):
        self._specs: dict[str, BatteryDomainSpec] = {}
        self._aliases: dict[str, str] = {}

    def register(self, spec: BatteryDomainSpec) -> "BatteryDomainRegistry":
        domain_id = str(spec.domain_id).strip().lower()
        if not domain_id:
            raise ValueError("domain_id must be non-empty")
        if domain_id in self._specs:
            raise ValueError(f"Domain id is already registered: {domain_id}")
        self._specs[domain_id] = spec
        alias = str(spec.paper_alias).strip().lower()
        if alias:
            if alias in self._aliases and self._aliases[alias] != domain_id:
                raise ValueError(f"Paper alias is already registered: {spec.paper_alias}")
            self._aliases[alias] = domain_id
        return self

    def canonical_id(self, value: str) -> str:
        key = str(value).strip().lower()
        if key in self._specs:
            return key
        if key in self._aliases:
            return self._aliases[key]
        raise KeyError(f"Unknown battery domain {value!r}; available={sorted(self._specs)}")

    def get(self, value: str) -> BatteryDomainSpec:
        return self._specs[self.canonical_id(value)]

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def metadata(self) -> dict[str, dict[str, Any]]:
        return {domain_id: self._specs[domain_id].metadata() for domain_id in self.available()}


XJTU_NORMALIZATION = {
    "mode": "physical_window",
    "cc_voltage_low": 4.0,
    "cc_voltage_high": 4.195,
    "cv_current_low": 0.1,
    "cv_current_high": 0.5,
    "cc_current_ref": 4.0,
    "cv_voltage_ref": 4.19975,
    "cv_voltage_scale": 0.001,
    "cc_current_mode": "physical",
    "cv_voltage_mode": "physical",
    "current_use_abs": True,
    "temp_room": 25.0,
    "temp_abs_scale": 20.0,
    "temp_delta_scale": 10.0,
}

MIT_NORMALIZATION = {
    "mode": "physical_window",
    "cc_voltage_low": 3.45,
    "cc_voltage_high": 3.6,
    "cv_current_low": 0.055,
    "cv_current_high": 0.275,
    "cc_current_ref": 1.0,
    "cv_voltage_ref": 3.6,
    "cv_voltage_scale": 0.001,
    "cc_current_mode": "physical",
    "cv_voltage_mode": "physical",
    "current_use_abs": True,
    "temp_room": 25.0,
    "temp_abs_scale": 20.0,
    "temp_delta_scale": 10.0,
}

# SmartHealth v3 first infers CC/CV inside the source's combined charge step.
# It then exports 3.45--3.58 V from inferred CC and nominal-C-rate
# 0.25C--0.05C from inferred CV.  These are fixed physical protocol windows,
# not train/validation-fitted statistics.
SMARTHEALTH_LISHEN_NORMALIZATION = {
    "mode": "physical_window",
    "cc_voltage_low": 3.45,
    "cc_voltage_high": 3.58,
    "cv_current_low": 2.0,
    "cv_current_high": 10.0,
    "cc_current_ref": 80.0,
    "cv_voltage_ref": 3.6,
    "cv_voltage_scale": 0.01,
    "cc_current_mode": "physical",
    "cv_voltage_mode": "physical",
    "current_use_abs": True,
    "temp_room": 25.0,
    "temp_abs_scale": 20.0,
    "temp_delta_scale": 10.0,
}

SMARTHEALTH_280AH_NORMALIZATION = {
    "mode": "physical_window",
    "cc_voltage_low": 3.45,
    "cc_voltage_high": 3.58,
    "cv_current_low": 14.0,
    "cv_current_high": 70.0,
    "cc_current_ref": 140.0,
    "cv_voltage_ref": 3.6,
    "cv_voltage_scale": 0.01,
    "cc_current_mode": "physical",
    "cv_voltage_mode": "physical",
    "current_use_abs": True,
    "temp_room": 25.0,
    "temp_abs_scale": 20.0,
    "temp_delta_scale": 10.0,
}


def build_default_domain_registry() -> BatteryDomainRegistry:
    """Return the Paper-v1 public-domain registry.

    SmartHealth entries refer only to its generated canonical RAW/feature
    products.  They never make the direct GB18030 source a model input.
    """

    registry = BatteryDomainRegistry()
    registry.register(
        BatteryDomainSpec(
            domain_id="xjtu",
            paper_alias="A",
            source="XJTU public cycling source",
            manufacturer="not declared in the currently tracked source metadata",
            battery_model="single XJTU public battery family",
            chemistry="not declared in the current Paper-v1 manifest",
            nominal_capacity_ah=2.0,
            voltage_range_v=(4.0, 4.2),
            operating_conditions=("2C", "3C", "R2.5", "R3", "RW", "satellite"),
            data_root="UnifiedRawSOH/datasets/XJTU_raw",
            feature_data_root="UnifiedRawSOH/datasets/XJTU_features",
            adapter_id="xjtu",
            split_file="UnifiedRawSOH/splits/xjtu/paper_v1_mixed_split.json",
            normalization=XJTU_NORMALIZATION,
            notes=("Operating conditions are conditions inside the XJTU domain.",),
        )
    )
    registry.register(
        BatteryDomainSpec(
            domain_id="mit",
            paper_alias="B",
            source="MIT/Severson public cycling source",
            manufacturer="not declared in the canonical physical-cell export",
            battery_model="single MIT public battery family",
            chemistry="not declared in the current Paper-v1 manifest",
            nominal_capacity_ah=1.1,
            voltage_range_v=(3.45, 3.6),
            operating_conditions=("2017-05-12", "2017-06-30", "2018-04-12"),
            data_root="UnifiedRawSOH/datasets/MIT_raw",
            feature_data_root="UnifiedRawSOH/datasets/MIT_features",
            adapter_id="mit_raw",
            split_file="UnifiedRawSOH/splits/mit/mit_paper_physical124_v2_split.json",
            normalization=MIT_NORMALIZATION,
            notes=(
                "Uses continuation-aware mit_p### physical-cell identities.",
                "Proposed raw uses phase-aware CC 3.45--3.60 V and nominal-capacity "
                "CV 0.25C--0.05C; canonical Only-F statistics use the same selected rows.",
            ),
        )
    )
    for domain_id, alias, manufacturer, model, capacity, normalization, audit_note in (
        (
            "smarthealth_lishen40",
            "C1",
            "LISHEN",
            "40 Ah LFP",
            40.0,
            SMARTHEALTH_LISHEN_NORMALIZATION,
            "v2 canonical export requires finite selected-CC/CV source temperature and records all exclusions.",
        ),
        (
            "smarthealth_catl280",
            "C2",
            "CATL",
            "280 Ah LFP",
            280.0,
            SMARTHEALTH_280AH_NORMALIZATION,
            "v2 canonical export uses the source current-taper CC/CV policy.",
        ),
        (
            "smarthealth_eve280",
            "C3",
            "EVE",
            "280 Ah LFP",
            280.0,
            SMARTHEALTH_280AH_NORMALIZATION,
            "Source chunks without temp1_1 are audited and excluded unless a duplicate finite-T candidate wins.",
        ),
    ):
        registry.register(
            BatteryDomainSpec(
                domain_id=domain_id,
                paper_alias=alias,
                source="SmartHealth public cycling source",
                manufacturer=manufacturer,
                battery_model=model,
                chemistry="LFP",
                nominal_capacity_ah=capacity,
                voltage_range_v=(3.45, 3.58),
                operating_conditions=(
                    "C-rate and DOD are in-domain operating/aging conditions, not domain IDs.",
                ),
                data_root="UnifiedRawSOH/datasets/SmartHealth_raw",
                feature_data_root="UnifiedRawSOH/datasets/SmartHealth_features",
                adapter_id="smarthealth",
                split_file=f"UnifiedRawSOH/splits/smarthealth/{domain_id}_cell_split.json",
                normalization=normalization,
                availability="available",
                notes=(
                    "Processed GB18030 source cycles use an auditable persistent current-taper "
                    "boundary inside the combined charge step.",
                    "Canonical model windows are inferred-CC 3.45--3.58 V and "
                    "inferred-CV nominal 0.25C--0.05C with a +/-0.002C sampling tolerance.",
                    "SOH is calibration-direct or calibration-interpolated; partial-DOD "
                    "source discharge capacity is never divided by 40/280 Ah as a label.",
                    "Each logical sequence is domain + source serial + C-rate + DOD; "
                    "conditions remain within the family and are never cross-merged.",
                    "Cell and source-cycle provenance live in SmartHealth_raw/audit.",
                    audit_note,
                ),
            )
        )
    # SMVIC is kept as six physical battery families rather than one broad
    # vendor bucket.  Its confidential point data and model-ready arrays stay
    # outside this repository; only stable semantic metadata is registered.
    for domain_id, alias, model, capacity, voltage_range, condition in (
        ("smvic_e72_69ah", "SMVIC1", "E72 69.4 Ah", 69.4, (2.80, 4.20), "T25_S1N1"),
        ("smvic_s5e891_51ah", "SMVIC2", "S5E891 51 Ah", 51.0, (2.80, 4.18), "T25_S1N1"),
        ("smvic_type1_18ah", "SMVIC3", "type1 18 Ah", 18.0, (2.50, 4.40), "T25 cycling"),
        ("smvic_type2_150ah_t40", "SMVIC4", "type2 150 Ah", 150.0, (3.50, 4.85), "T40 aging"),
        ("smvic_type3_108ah", "SMVIC5", "type3 108 Ah", 108.0, (3.50, 4.78), "T25 cycling"),
        ("smvic_type4_11ah", "SMVIC6", "type4 11.4 Ah", 11.4, (3.00, 4.20), "T25 cycling"),
    ):
        registry.register(
            BatteryDomainSpec(
                domain_id=domain_id,
                paper_alias=alias,
                source="SMVIC enterprise normalized cell CSV v2",
                manufacturer="confidential enterprise source",
                battery_model=model,
                chemistry="ternary" if domain_id in {"smvic_e72_69ah", "smvic_s5e891_51ah"} else "not declared",
                nominal_capacity_ah=capacity,
                voltage_range_v=voltage_range,
                operating_conditions=(condition,),
                data_root=None,
                feature_data_root=None,
                adapter_id="smvic_preprocessed_only",
                split_file=None,
                normalization=None,
                availability="external_preprocessed_only",
                notes=(
                    "Model input is the locally generated, Git-ignored datasets/SMVIC_preprocessed_v2_128x128 product.",
                    "SOH is same-cycle discharge capacity divided by fixed nominal capacity.",
                ),
            )
        )
    return registry


def canonical_domain_id(value: str) -> str:
    """Resolve stable IDs while retaining source-name compatibility for checkpoints."""

    key = str(value).strip().lower()
    legacy = {
        "xjtu_raw": "xjtu",
        "xjtu_raw_t_v1": "xjtu",
        "xjtu_features": "xjtu",
        "xjtu_t_v1": "xjtu",
        "mit_raw": "mit",
        "mit_features": "mit",
    }
    key = legacy.get(key, key)
    return build_default_domain_registry().canonical_id(key)
