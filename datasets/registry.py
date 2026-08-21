"""Dataset registry used by single-domain and future unified loaders."""

from __future__ import annotations

from pathlib import Path

from .mit import MITFeatureAdapter, MITRawAdapter
from .smarthealth import SmartHealthRawAdapter
from .xjtu import XJTURawAdapter


class DatasetRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, dataset_id, factory):
        key = str(dataset_id)
        if key in self._factories:
            raise ValueError(f"Dataset id is already registered: {key}")
        self._factories[key] = factory
        return self

    def create(self, dataset_id, **kwargs):
        key = str(dataset_id)
        if key not in self._factories:
            raise KeyError(f"Unknown dataset id {key!r}; available={sorted(self._factories)}")
        return self._factories[key](**kwargs)

    def available(self):
        return tuple(sorted(self._factories))


def build_default_registry():
    registry = DatasetRegistry()
    registry.register(
        "xjtu",
        lambda data_root, **kwargs: XJTURawAdapter(
            data_root,
            nominal_capacity=kwargs.get("nominal_capacity", 2.0),
            label_scale_mode=kwargs.get("label_scale_mode", "auto_capacity_to_soh"),
        ),
    )
    def mit_factory(data_root, **kwargs):
        root_name = Path(data_root).name.lower()
        if root_name.startswith("mit_raw"):
            return MITRawAdapter(
                data_root,
                nominal_capacity=kwargs.get("nominal_capacity", 1.1),
                label_scale_mode=kwargs.get("label_scale_mode", "none"),
            )
        return MITFeatureAdapter(data_root)

    registry.register("mit", mit_factory)
    registry.register(
        "mit_raw",
        lambda data_root, **kwargs: MITRawAdapter(
            data_root,
            nominal_capacity=kwargs.get("nominal_capacity", 1.1),
            label_scale_mode=kwargs.get("label_scale_mode", "none"),
        ),
    )
    registry.register(
        "smarthealth",
        lambda data_root, **kwargs: SmartHealthRawAdapter(
            data_root,
            domain_id=kwargs.get("domain_id"),
            nominal_capacity=kwargs.get("nominal_capacity"),
            label_scale_mode=kwargs.get("label_scale_mode", "label_capacity_to_nominal"),
        ),
    )
    return registry
