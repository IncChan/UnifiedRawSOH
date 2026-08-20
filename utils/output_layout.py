"""Canonical Paper-v1 output namespace and provenance manifest helpers."""

from __future__ import annotations

from pathlib import Path


_OUTPUT_FIELDS = ("paper_version", "experiment_id", "model_id", "data_id")


def _path_component(field, value):
    value = str(value).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"output.{field} must be one safe directory component; got {value!r}")
    return value


def output_identity(config):
    """Return the four explicit components of a Paper-v1 result namespace."""

    output = config.get("output", {})
    missing = [field for field in _OUTPUT_FIELDS if field not in output]
    if missing:
        source = config.get("_config_path", "<resolved config>")
        raise ValueError(f"{source} is missing required output fields: {', '.join(missing)}")
    return {field: _path_component(field, output[field]) for field in _OUTPUT_FIELDS}


def output_namespace(config):
    identity = output_identity(config)
    return tuple(identity[field] for field in _OUTPUT_FIELDS)


def build_batch_output_dir(output_root, config, run_time):
    """Build ``Paper-v1/e*/model/data/runtime_*`` without creating it."""

    return Path(output_root).joinpath(*output_namespace(config), str(run_time))


def build_seed_output_dir(output_root, config, run_time, seed):
    return build_batch_output_dir(output_root, config, run_time) / f"seed_{int(seed)}"


def build_run_manifest(config, output_root, run_time, seed=None):
    """Small, stable provenance record stored next to a batch or seed result."""

    experiment = config.get("experiment", {})
    data = config.get("data", {})
    payload = {
        "output": output_identity(config),
        "runtime": str(run_time),
        "output_directory": str(build_batch_output_dir(output_root, config, run_time)),
        "experiment": {
            key: experiment[key]
            for key in (
                "name",
                "task",
                "domain_id",
                "domain_ids",
                "dataset_id",
                "dataset_ids",
                "source_domain_ids",
                "target_domain_id",
                "target_domain_ids",
                "source_dataset_id",
                "target_dataset_id",
                "loader",
            )
            if key in experiment and experiment[key] is not None
        },
        "data": {
            key: data[key]
            for key in (
                "data_root",
                "data_roots",
                "dataset",
                "split_file",
                "split_files",
                "batches_by_domain",
            )
            if key in data and data[key] is not None
        },
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if "reusability" in config:
        payload["reusability"] = dict(config["reusability"])
    return payload
