"""Dataset adapters and the unified CC/CV sample contract."""

from .splits import (
    apply_battery_roles,
    get_development_protocol,
    load_battery_roles,
    load_invalid_cycles,
    load_split_spec,
    load_test_batteries,
    resolve_test_batteries,
    split_mixed_cycle_records,
    split_records_from_spec,
)
from .mit import (
    MITFeatureAdapter,
    MITRawAdapter,
    inspect_mit_raw_inventory,
    load_mit_raw_records,
    read_mit_raw_file,
    validate_mit_physical_cohort,
)
from .domains import (
    BatteryDomainRegistry,
    BatteryDomainSpec,
    build_default_domain_registry,
    canonical_domain_id,
)
from .registry import DatasetRegistry, build_default_registry
from .smarthealth import (
    SmartHealthRawAdapter,
    audit_smarthealth_source,
    list_smarthealth_raw_files,
    load_smarthealth_raw_records,
    read_smarthealth_raw_file,
)
from .xjtu import (
    UnifiedCCCVSampleDataset,
    XJTURawAdapter,
    build_full_life_cycle_metadata,
    load_xjtu_records,
)

__all__ = [
    "UnifiedCCCVSampleDataset",
    "XJTURawAdapter",
    "apply_battery_roles",
    "build_full_life_cycle_metadata",
    "get_development_protocol",
    "load_battery_roles",
    "load_invalid_cycles",
    "load_split_spec",
    "load_test_batteries",
    "resolve_test_batteries",
    "load_xjtu_records",
    "split_mixed_cycle_records",
    "split_records_from_spec",
    "MITFeatureAdapter",
    "MITRawAdapter",
    "inspect_mit_raw_inventory",
    "load_mit_raw_records",
    "read_mit_raw_file",
    "validate_mit_physical_cohort",
    "DatasetRegistry",
    "build_default_registry",
    "BatteryDomainRegistry",
    "BatteryDomainSpec",
    "build_default_domain_registry",
    "canonical_domain_id",
    "SmartHealthRawAdapter",
    "audit_smarthealth_source",
    "list_smarthealth_raw_files",
    "load_smarthealth_raw_records",
    "read_smarthealth_raw_file",
]
