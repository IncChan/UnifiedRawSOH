"""Independent PINN4SOH-noLeak-OnlyF baseline.

This is a direct Paper-v1 migration of the validated F-only statistical
feature path: the original 16 PINN electrical statistics, eight temperature
statistics, all-column 3-sigma cleaning, adjacent-x1 sampling, pooled
train/validation min-max normalization, and the sinusoidal encoder/predictor
MLP.  The legacy cycle index is retained only as a source-row identifier; it
is never a model input or a physical-cycle matching key.
"""

from __future__ import annotations

import copy
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from UnifiedRawSOH.datasets.domains import canonical_domain_id
from UnifiedRawSOH.datasets.splits import (
    get_development_protocol,
    load_split_spec,
    resolve_test_batteries,
)
from UnifiedRawSOH.datasets.mit import validate_mit_physical_cohort
from UnifiedRawSOH.datasets.smarthealth import (
    SMARTHEALTH_CANONICAL_POLICY_VERSION,
    SMARTHEALTH_NOMINAL_CAPACITY_AH,
)
from UnifiedRawSOH.datasets.soh_labels import (
    BOL_LABEL_MODE,
    BOL_RULE_VERSION,
    apply_bol_relative_soh,
    build_bol_reference,
    frozen_smarthealth_bol_references,
    is_bol_label_mode,
)


PINN16_FEATURE_COLUMNS = [
    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
    "CC Q", "CC charge time", "voltage slope", "voltage entropy",
    "current mean", "current std", "current kurtosis", "current skewness",
    "CV Q", "CV charge time", "current slope", "current entropy",
]
TEMPERATURE_FEATURE_COLUMNS = [
    "T_CC_mean", "T_CC_max", "T_CC_delta", "T_CC_slope",
    "T_CV_mean", "T_CV_max", "T_CV_delta", "T_CV_slope",
]
MIT_PHYSICAL_FILENAME_RE = re.compile(
    r"MIT_\d{4}-\d{2}-\d{2}_physical-\d+\.csv$"
)
MIT_LEGACY_FILENAME_RE = re.compile(
    r"MIT_\d{4}-\d{2}-\d{2}_.*_cell-\d+\.csv$"
)


def _smarthealth_feature_metadata(path):
    """Read canonical SmartHealth identity metadata, not a filename guess."""

    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"domain_id", "condition", "battery_id", "cycle", "capacity"}
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Canonical SmartHealth feature file {path} is missing: {missing}")
        first = next(reader, None)
    if first is None:
        raise ValueError(f"Canonical SmartHealth feature file is empty: {path}")
    domain_id = str(first["domain_id"]).strip()
    condition = str(first["condition"]).strip()
    battery_id = str(first["battery_id"]).strip()
    if not domain_id or not condition or not battery_id:
        raise ValueError(f"Canonical SmartHealth feature file lacks identity metadata: {path}")
    return domain_id, condition, battery_id


def _smarthealth_feature_identity(path):
    _, condition, battery_id = _smarthealth_feature_metadata(path)
    return condition, battery_id


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class PINNEncoderMLP(nn.Module):
    def __init__(self, input_dim=24, output_dim=32, layers_num=3, hidden_dim=60, dropout=0.2):
        super().__init__()
        if int(layers_num) < 2:
            raise ValueError("layers_num must be >= 2")
        modules = []
        for idx in range(int(layers_num)):
            if idx == 0:
                modules.extend([nn.Linear(input_dim, hidden_dim), Sin()])
            elif idx == int(layers_num) - 1:
                modules.append(nn.Linear(hidden_dim, output_dim))
            else:
                modules.extend([nn.Linear(hidden_dim, hidden_dim), Sin(), nn.Dropout(p=dropout)])
        self.net = nn.Sequential(*modules)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.net(x)


class PINNPredictor(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, hidden_dim),
            Sin(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class PINNFOnlyMLP(nn.Module):
    """The historical Solution_u/comparison F-only supervised regressor."""

    def __init__(
        self,
        input_dim=24,
        encoder_hidden_dim=60,
        encoder_output_dim=32,
        encoder_layers_num=3,
        predictor_hidden_dim=32,
        dropout=0.2,
    ):
        super().__init__()
        self.encoder = PINNEncoderMLP(
            input_dim=input_dim,
            output_dim=encoder_output_dim,
            layers_num=encoder_layers_num,
            hidden_dim=encoder_hidden_dim,
            dropout=dropout,
        )
        self.predictor = PINNPredictor(
            input_dim=encoder_output_dim,
            hidden_dim=predictor_hidden_dim,
            dropout=dropout,
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, features):
        return self.predictor(self.encoder(features))


def parse_file_identity(path):
    stem = Path(path).stem
    if Path(path).name.startswith("smarthealth_"):
        return _smarthealth_feature_identity(path)
    mit_physical_match = re.match(
        r"MIT_(?P<date>\d{4}-\d{2}-\d{2})_physical-(?P<physical>\d+)$",
        stem,
    )
    if mit_physical_match is not None:
        return (
            mit_physical_match.group("date"),
            f"mit_p{int(mit_physical_match.group('physical')):03d}",
        )
    mit_match = re.match(
        r"MIT_(?P<date>\d{4}-\d{2}-\d{2})_.*_cell-(?P<cell>\d+)$",
        stem,
    )
    if mit_match is not None:
        condition = mit_match.group("date")
        return condition, f"{condition}_battery-{int(mit_match.group('cell'))}"
    battery_match = re.match(r"(?P<condition>.+)_battery-(?P<cell>\d+)$", stem)
    if battery_match is None:
        return stem, stem
    condition = battery_match.group("condition")
    if condition == "Sim_satellite":
        condition = "satellite"
    return condition, f"{condition}_battery-{int(battery_match.group('cell'))}"


def list_feature_csv_files(data_root, batch=None, domain_id=None):
    root = Path(data_root)
    if not root.is_dir():
        raise ValueError(f"F-only feature root does not exist: {root}")
    smarthealth_domain = (
        str(domain_id) if domain_id is not None and str(domain_id).startswith("smarthealth_")
        else None
    )
    # v2 canonical SmartHealth products are intentionally namespaced by family
    # below the common feature root.  Other historical feature layouts remain
    # flat, so do not recursively scan those directories.
    if smarthealth_domain is not None:
        search_root = root / smarthealth_domain
        if not search_root.is_dir():
            search_root = root
        candidates = sorted(search_root.rglob("smarthealth_*.csv"))
    else:
        candidates = sorted(root.glob("*.csv"))
    files = []
    for path in candidates:
        # Canonical MIT physical products keep provenance CSVs in the same
        # directory.  A generic *.csv scan must not interpret those as a
        # feature battery.
        if path.name.startswith("MIT_") and not (
            MIT_PHYSICAL_FILENAME_RE.match(path.name)
            or MIT_LEGACY_FILENAME_RE.match(path.name)
        ):
            continue
        smarthealth_metadata = None
        if path.name.startswith("smarthealth_"):
            smarthealth_metadata = _smarthealth_feature_metadata(path)
            if domain_id is not None and smarthealth_metadata[0] != str(domain_id):
                continue
        elif domain_id is not None and str(domain_id).startswith("smarthealth_"):
            continue
        if batch is not None:
            if smarthealth_metadata is not None:
                matches = smarthealth_metadata[1] == str(batch)
            else:
                matches = (
                    path.name.startswith("Sim_satellite")
                    if batch == "satellite"
                    else path.name.startswith(f"{batch}_") or path.name.startswith(f"MIT_{batch}_")
                )
            if not matches:
                continue
        files.append(path)
    if not files:
        raise ValueError(f"No F-only feature files found under {root} for batch={batch!r}")
    return files


def split_feature_files_by_battery(files, test_batteries=None):
    """Split files by exact battery identity.

    The caller must provide the test IDs resolved from the dataset split JSON.
    This keeps the feature baseline independent of dataset-specific battery
    naming and prevents a hidden XJTU-only fallback.
    """

    if test_batteries is None:
        raise ValueError("F-only split requires test_batteries from a dataset split JSON")
    configured = {str(item) for item in test_batteries}
    train_val, test = [], []
    for path in files:
        battery_id = parse_file_identity(path)[1]
        is_test = battery_id in configured
        if is_test:
            test.append(path)
        else:
            train_val.append(path)
    if not train_val or not test:
        raise ValueError("F-only battery split produced an empty development or test set")
    return train_val, test


def delete_3sigma(frame, columns=None):
    values = frame.replace([np.inf, -np.inf], np.nan)
    # SmartHealth canonical products intentionally leave optional audit fields
    # such as ``split_issue`` blank.  ``columns`` is the statistical cleaning
    # contract; metadata NaN values must not delete otherwise valid samples.
    cleaning_columns = (
        list(values.columns)
        if columns is None
        else [column for column in columns if column in values.columns]
    )
    values = values.dropna(subset=cleaning_columns).reset_index(drop=True)
    outlier_indices = set()
    for column in cleaning_columns:
        if column not in values.columns:
            continue
        # Canonical physical products add string provenance columns.  They are
        # identifiers, not statistics, and should neither crash nor influence
        # the historical numeric all-column 3-sigma rule.
        series = pd.to_numeric(values[column], errors="coerce")
        if not series.notna().all():
            continue
        std = series.std()
        if not np.isfinite(std) or std == 0:
            continue
        rule = (series.mean() - 3 * std > series) | (series.mean() + 3 * std < series)
        outlier_indices.update(np.flatnonzero(rule.to_numpy()).tolist())
    if outlier_indices:
        values = values.drop(sorted(outlier_indices), axis=0)
    return values.reset_index(drop=True)


def load_feature_file(path, config):
    data_cfg = config["data"]
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(
            f"{path} contains only a header and cannot be used for PINN4SOH-noLeak-OnlyF. "
            "Regenerate/copy the paired canonical feature export; no legacy feature fallback is used."
        )
    missing = [column for column in PINN16_FEATURE_COLUMNS + TEMPERATURE_FEATURE_COLUMNS + ["capacity"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing F-only columns: {missing}")
    cycle_column = str(data_cfg.get("cycle_column", "cycle index"))
    is_physical = "physical_cell_id" in frame.columns and "cycle" in frame.columns
    observed_domains = (
        {str(value).strip() for value in frame["domain_id"].dropna()}
        if "domain_id" in frame.columns
        else set()
    )
    is_smarthealth = (
        "logical_sequence_id" in frame.columns
        and "cycle" in frame.columns
        and bool(observed_domains)
        and all(value.startswith("smarthealth_") for value in observed_domains)
    )
    source_row_column = "__source_row_index"
    frame[source_row_column] = np.arange(len(frame), dtype=np.int64)
    raw_cycle_id = None
    if is_physical:
        physical_ids = {str(item).strip() for item in frame["physical_cell_id"].dropna()}
        condition, filename_battery_id = parse_file_identity(path)
        if physical_ids != {filename_battery_id}:
            raise ValueError(
                f"Physical MIT feature filename/id mismatch in {path}: "
                f"filename={filename_battery_id}, rows={sorted(physical_ids)}"
            )
        raw_cycle_id = pd.to_numeric(frame["cycle"], errors="raise").to_numpy(dtype=np.int64)
        cycle_values = raw_cycle_id
    elif is_smarthealth:
        required_lineage = {
            "strategy_version",
            "source_cycle",
            "source_absolute_start_time",
            "source_absolute_end_time",
        }
        if is_bol_label_mode(config):
            required_lineage.update(
                {"bol_q_ref_Ah", "bol_q_ref_rule", "bol_q_ref_source"}
            )
        missing_lineage = sorted(required_lineage - set(frame.columns))
        if missing_lineage:
            raise ValueError(
                f"Canonical SmartHealth feature file {path} is missing v3 chronology lineage: "
                f"{missing_lineage}"
            )
        policy_versions = {str(value).strip() for value in frame["strategy_version"].dropna()}
        if policy_versions != {SMARTHEALTH_CANONICAL_POLICY_VERSION}:
            raise ValueError(
                f"Canonical SmartHealth feature policy mismatch in {path}: {sorted(policy_versions)}"
            )
        logical_ids = {str(item).strip() for item in frame["logical_sequence_id"].dropna()}
        condition, filename_battery_id = parse_file_identity(path)
        if logical_ids != {filename_battery_id}:
            raise ValueError(
                f"Canonical SmartHealth feature filename/id mismatch in {path}: "
                f"filename={filename_battery_id}, rows={sorted(logical_ids)}"
            )
        raw_cycle_id = pd.to_numeric(frame["cycle"], errors="raise").to_numpy(dtype=np.int64)
        if (
            np.any(raw_cycle_id <= 0)
            or len(np.unique(raw_cycle_id)) != len(raw_cycle_id)
        ):
            raise ValueError(
                f"Canonical SmartHealth feature cycle IDs must be unique positive chronological IDs in {path}"
            )
        frame = frame.sort_values("cycle", kind="stable").reset_index(drop=True)
        source_start = pd.to_datetime(
            frame["source_absolute_start_time"], errors="coerce"
        )
        source_end = pd.to_datetime(
            frame["source_absolute_end_time"], errors="coerce"
        )
        if source_start.isna().any() or source_end.isna().any() or (source_end < source_start).any():
            raise ValueError(
                f"Canonical SmartHealth feature source-time provenance is invalid in {path}"
            )
        intervals = list(zip(source_start.tolist(), source_end.tolist()))
        if any(current < previous for previous, current in zip(intervals, intervals[1:])):
            raise ValueError(
                f"Canonical SmartHealth feature chronology regresses in {path}"
            )
        raw_cycle_id = frame["cycle"].to_numpy(dtype=np.int64)
        cycle_values = raw_cycle_id
    else:
        cycle_values = np.arange(len(frame), dtype=np.int64)
    if cycle_column in frame.columns:
        raise ValueError(f"Configured cycle column already exists in {path}: {cycle_column}")

    configured_domain = config.get("experiment", {}).get(
        "domain_id", config.get("experiment", {}).get(
            "dataset_id", data_cfg.get("domain_id", data_cfg.get("dataset_id", "xjtu"))
        )
    )
    resolved_domain = next(iter(observed_domains)) if observed_domains else canonical_domain_id(configured_domain)
    label_provenance = None
    if is_bol_label_mode(config):
        condition, filename_battery_id = parse_file_identity(path)
        label_records = []
        for row_index in range(len(frame)):
            source = {
                "domain_id": resolved_domain,
                "battery_id": filename_battery_id,
                "cycle_id": int(cycle_values[row_index]),
                "raw_cycle_order_index": int(row_index),
            }
            capacity = frame.iloc[row_index]["capacity"]
            if resolved_domain == "mit":
                source["capacity_Ah"] = capacity
            elif str(resolved_domain).startswith("smarthealth_"):
                source["label_capacity_Ah"] = frame.iloc[row_index].get("label_capacity_Ah", capacity)
                source["label_source"] = frame.iloc[row_index].get("label_source", "")
                source["bol_q_ref_Ah"] = frame.iloc[row_index].get("bol_q_ref_Ah")
                source["bol_q_ref_rule"] = frame.iloc[row_index].get("bol_q_ref_rule")
                source["bol_q_ref_source"] = frame.iloc[row_index].get("bol_q_ref_source")
            else:
                source["SOH"] = capacity
            label_records.append(source)
        if str(resolved_domain).startswith("smarthealth_"):
            frozen = frozen_smarthealth_bol_references(
                label_records, domain_id=resolved_domain
            )
            label_provenance = frozen[filename_battery_id]
        else:
            label_provenance = build_bol_reference(
                label_records, domain_id=resolved_domain
            )
        labeled = apply_bol_relative_soh(
            label_records, label_provenance, domain_id=resolved_domain
        )
        label_by_source_row = {
            int(row["raw_cycle_order_index"]): float(row["soh_bol"])
            for row in labeled
        }
        frame["__soh_bol"] = [
            label_by_source_row[int(value)] for value in range(len(frame))
        ]
    frame.insert(frame.shape[1] - 1, cycle_column, cycle_values)
    if bool(data_cfg.get("drop_3sigma_outliers", True)):
        # "All-column" is the historical statistical feature table plus its
        # capacity label and legacy cycle identifier.  The v3 lineage fields
        # are audit metadata, not statistics, and must not alter cleaning.
        cleaning_columns = [
            *PINN16_FEATURE_COLUMNS,
            *TEMPERATURE_FEATURE_COLUMNS,
            "capacity",
            cycle_column,
        ]
        frame = delete_3sigma(frame, columns=cleaning_columns)
    frame = frame.reset_index(drop=True)
    feature_columns = [*PINN16_FEATURE_COLUMNS, *TEMPERATURE_FEATURE_COLUMNS]
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    target_mode = str(data_cfg.get("feature_target_mode", "")).strip()
    label_column = str(data_cfg.get("feature_label_column", "")).strip()
    if is_bol_label_mode(config):
        soh = frame["__soh_bol"].to_numpy(dtype=np.float32).reshape(-1, 1)
    elif target_mode == "capacity_to_nominal":
        # SmartHealth and the raw model now share the same fixed-nominal
        # capacity target.  ``capacity`` is the calibration-derived physical
        # capacity exported by the matching feature preprocessor, never a
        # handcrafted feature or an inference input.
        nominal = float(data_cfg.get("nominal_capacity", 2.0))
        if not np.isfinite(nominal) or nominal <= 0.0:
            raise ValueError(f"{path}: nominal_capacity must be finite and positive")
        if is_smarthealth:
            expected_nominals = {
                SMARTHEALTH_NOMINAL_CAPACITY_AH[domain]
                for domain in observed_domains
            }
            if len(expected_nominals) != 1 or not np.isclose(nominal, next(iter(expected_nominals))):
                raise ValueError(
                    f"{path}: SmartHealth nominal_capacity={nominal} conflicts with "
                    f"domain metadata={sorted(expected_nominals)}"
                )
        soh = (frame["capacity"].to_numpy(dtype=np.float32) / nominal).reshape(-1, 1)
    elif target_mode:
        raise ValueError(
            f"Unsupported feature_target_mode={target_mode!r}; expected 'capacity_to_nominal' or empty."
        )
    elif label_column:
        if label_column not in frame.columns:
            raise ValueError(f"{path} is missing configured feature label column {label_column!r}")
        soh = frame[label_column].to_numpy(dtype=np.float32).reshape(-1, 1)
    else:
        nominal = float(data_cfg.get("nominal_capacity", 2.0))
        soh = (frame["capacity"].to_numpy(dtype=np.float32) / nominal).reshape(-1, 1)
    condition, battery_id = parse_file_identity(path)
    if is_physical:
        observed_ids = {str(item).strip() for item in frame["physical_cell_id"].dropna()}
        if observed_ids and observed_ids != {battery_id}:
            raise ValueError(
                f"Physical MIT feature identity changed after cleaning in {path}: {sorted(observed_ids)}"
            )
    elif is_smarthealth:
        observed_ids = {str(item).strip() for item in frame["logical_sequence_id"].dropna()}
        if observed_ids and observed_ids != {battery_id}:
            raise ValueError(
                f"SmartHealth feature identity changed after cleaning in {path}: {sorted(observed_ids)}"
            )
    return {
        "features": features,
        "soh": soh,
        "soh_bol": soh if is_bol_label_mode(config) else None,
        "soh_label_mode": BOL_LABEL_MODE if is_bol_label_mode(config) else "rated_relative",
        "label_provenance": label_provenance,
        "battery_id": battery_id,
        "condition": condition,
        "cycle_id": frame[cycle_column].to_numpy(dtype=np.int64),
        # These identify the original feature-source row for provenance-only
        # tasks such as matched-cycle evaluation.  They are never model input.
        "source_file": str(path),
        "source_row_index": frame[source_row_column].to_numpy(dtype=np.int64),
        "raw_cycle_id": (
            frame["cycle"].to_numpy(dtype=np.int64) if is_physical or is_smarthealth else None
        ),
        "feature_columns": feature_columns,
    }


def build_adjacent_first_samples(payloads):
    rows = []
    for payload in payloads:
        if len(payload["features"]) <= 1:
            continue
        for index in range(len(payload["features"]) - 1):
            rows.append(
                {
                    "features": payload["features"][index],
                    "soh": payload["soh"][index],
                    "soh_bol": (
                        None if payload.get("soh_bol") is None else payload["soh_bol"][index]
                    ),
                    "soh_label_mode": payload.get("soh_label_mode", "rated_relative"),
                    "label_provenance": payload.get("label_provenance"),
                    "battery_id": payload["battery_id"],
                    "condition": payload["condition"],
                    "cycle_id": int(payload["cycle_id"][index]),
                    "source_file": payload["source_file"],
                    "source_row_index": int(payload["source_row_index"][index]),
                    "raw_cycle_id": (
                        None
                        if payload.get("raw_cycle_id") is None
                        else int(payload["raw_cycle_id"][index])
                    ),
                }
            )
    if not rows:
        raise ValueError("No adjacent-x1 F-only samples were built")
    return rows


def fit_feature_minmax(rows, eps=1e-8):
    features = np.stack([row["features"] for row in rows], axis=0).astype(np.float32)
    minimum = features.min(axis=0)
    maximum = features.max(axis=0)
    return {
        "mode": "train_valid_minmax",
        "feature_names": None,
        "min": minimum.astype(float).tolist(),
        "max": maximum.astype(float).tolist(),
        "eps": float(eps),
        "formula": "2 * (x - min) / max(max - min, eps) - 1",
    }


def apply_feature_minmax(rows, normalizer):
    minimum = np.asarray(normalizer["min"], dtype=np.float32)
    maximum = np.asarray(normalizer["max"], dtype=np.float32)
    denom = np.maximum(maximum - minimum, float(normalizer.get("eps", 1e-8)))
    output = []
    for row in rows:
        item = dict(row)
        item["features"] = (2.0 * (np.asarray(row["features"], dtype=np.float32) - minimum) / denom - 1.0).astype(np.float32)
        output.append(item)
    return output


class StatFeatureDataset(Dataset):
    def __init__(self, rows, dataset_id="xjtu_features", domain_id="xjtu"):
        self.rows = list(rows)
        self.dataset_id = str(dataset_id)
        self.domain_id = str(domain_id)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        output = {
            "features": torch.from_numpy(np.asarray(row["features"], dtype=np.float32)),
            "soh": torch.from_numpy(np.asarray(row["soh"], dtype=np.float32)),
            "soh_label_mode": row.get("soh_label_mode", "rated_relative"),
            "battery_id": row["battery_id"],
            "cycle_id": int(row["cycle_id"]),
            "condition": row["condition"],
            "batch_name": row["condition"],
            "split": row.get("split", ""),
            "dataset_id": self.dataset_id,
            "domain_id": self.domain_id,
        }
        if row.get("soh_bol") is not None:
            output["soh_bol"] = torch.from_numpy(
                np.asarray(row["soh_bol"], dtype=np.float32)
            )
        return output


def build_feature_lodo_loaders(config, repo_root, seed=42):
    """Build zero-shot Feature MLP LODO loaders from the shared single-domain path.

    Each source domain is split with its own canonical split JSON. Only source
    train/validation datasets are concatenated; the target domain contributes
    only its original test dataset. This keeps the optional Feature MLP LODO
    interface honest without duplicating feature-file or BOL-label code.
    """

    from UnifiedRawSOH.trainers.reusability import parse_reusability_protocol

    protocol = parse_reusability_protocol(config)
    if protocol["protocol"] != "leave_one_domain_out":
        raise ValueError("Feature LODO requires reusability.protocol=leave_one_domain_out")
    source_domain_ids = protocol["source_domain_ids"]
    target_domain_ids = protocol["target_domain_ids"]
    if len(target_domain_ids) != 1:
        raise ValueError("Feature LODO requires exactly one target domain")
    target_domain_id = target_domain_ids[0]
    configured = [
        canonical_domain_id(value)
        for value in config.get("experiment", {}).get("domain_ids", [])
    ]
    expected = source_domain_ids + [target_domain_id]
    if set(configured) != set(expected):
        raise ValueError(
            "Feature LODO experiment.domain_ids must equal source domains plus "
            f"target domain; configured={configured}, expected={expected}"
        )

    data_cfg = config["data"]
    roots = data_cfg.get("data_roots", {})
    split_files = data_cfg.get("split_files", {})
    nominal_capacities = data_cfg.get("nominal_capacities", {})
    domain_loaders = {}
    domain_info = {}
    for index, domain_id in enumerate(expected):
        domain_config = copy.deepcopy(config)
        domain_config.setdefault("experiment", {})["loader"] = "feature_single_domain"
        domain_config["experiment"]["domain_id"] = domain_id
        domain_config["experiment"].pop("source_domain_ids", None)
        domain_config["experiment"].pop("target_domain_id", None)
        domain_config["experiment"].pop("target_domain_ids", None)
        batches_by_domain = data_cfg.get("batches_by_domain", {})
        if domain_id in batches_by_domain:
            domain_config["experiment"]["batches"] = list(batches_by_domain[domain_id])
        domain_config["data"] = copy.deepcopy(data_cfg)
        root_value = roots.get(domain_id)
        if not root_value:
            raise ValueError(f"Feature LODO config has no data root for {domain_id!r}")
        split_file = split_files.get(domain_id)
        if not split_file:
            raise ValueError(f"Feature LODO config has no split file for {domain_id!r}")
        domain_config["data"]["data_root"] = root_value
        domain_config["data"]["split_file"] = split_file
        if domain_id in nominal_capacities:
            domain_config["data"]["nominal_capacity"] = nominal_capacities[domain_id]
        if domain_id == target_domain_id:
            source_normalizers = [
                domain_info[source_domain]["normalization"]
                for source_domain in source_domain_ids
            ]
            minima = np.stack(
                [np.asarray(item["min"], dtype=np.float64) for item in source_normalizers],
                axis=0,
            )
            maxima = np.stack(
                [np.asarray(item["max"], dtype=np.float64) for item in source_normalizers],
                axis=0,
            )
            domain_config["feature_normalizer_override"] = {
                "mode": "source_domains_train_val_minmax",
                "feature_names": list(source_normalizers[0].get("feature_names") or []),
                "min": np.min(minima, axis=0).tolist(),
                "max": np.max(maxima, axis=0).tolist(),
                "eps": float(source_normalizers[0].get("eps", 1e-8)),
                "formula": "2 * (x - min) / max(max - min, eps) - 1",
                "test_statistics_used": False,
            }
        domain_loaders[domain_id], domain_info[domain_id] = build_feature_loaders(
            domain_config, repo_root, seed=int(seed) + index * 10_000
        )

    train_dataset = ConcatDataset(
        [domain_loaders[domain_id]["train"].dataset for domain_id in source_domain_ids]
    )
    val_dataset = ConcatDataset(
        [domain_loaders[domain_id]["val"].dataset for domain_id in source_domain_ids]
    )
    target_test_loader = domain_loaders[target_domain_id]["test"]
    train_cfg = config["train"]
    data_cfg = config["data"]
    common = {
        "batch_size": int(train_cfg.get("batch_size", 64)),
        "num_workers": int(data_cfg.get("num_workers", 0)),
    }
    loaders = {
        "train": DataLoader(train_dataset, shuffle=True, **common),
        "val": DataLoader(val_dataset, shuffle=False, **common),
        "test": target_test_loader,
    }
    return loaders, {
        "loader_type": "feature_leave_one_domain_out",
        "source_domain_ids": source_domain_ids,
        "target_domain_id": target_domain_id,
        "domain_ids": expected,
        "domain_info": domain_info,
        "sample_counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(target_test_loader.dataset),
        },
        "target_train_validation_samples_not_emitted": {
            "train": len(domain_loaders[target_domain_id]["train"].dataset),
            "val": len(domain_loaders[target_domain_id]["val"].dataset),
        },
        "split_usage": {
            "train": "source domains' original train split only",
            "val": "source domains' original val split only",
            "test": "left-out domain's original test split only",
            "excluded": [
                "source domains' test splits",
                "left-out domain's train and val splits",
            ],
        },
        "cycle_index_used_as_model_input": False,
        "label": {
            "label_mode": BOL_LABEL_MODE if is_bol_label_mode(config) else "rated_relative",
            "label_field": "soh_bol" if is_bol_label_mode(config) else "soh",
            "reference_rule": BOL_RULE_VERSION if is_bol_label_mode(config) else None,
            "q_ref_is_model_input": False,
            "q_ref_in_normalization": False,
        },
    }

def build_feature_loaders(config, repo_root, seed=42):
    """Build the paired all-batch mixed-cycle F-only protocol."""

    if config.get("experiment", {}).get("loader") == "feature_leave_one_domain_out":
        return build_feature_lodo_loaders(config, repo_root, seed=seed)

    data_cfg = config["data"]
    train_cfg = config["train"]
    experiment_cfg = config.get("experiment", {})
    configured_domain_id = experiment_cfg.get(
        "domain_id",
        experiment_cfg.get(
            "dataset_id", data_cfg.get("domain_id", data_cfg.get("dataset_id", data_cfg.get("dataset", "xjtu_features")))
        ),
    )
    domain_id = canonical_domain_id(configured_domain_id)
    if domain_id == "mit":
        dataset_id = "mit_features"
    elif domain_id == "xjtu":
        dataset_id = "xjtu_features"
    elif domain_id.startswith("smarthealth_"):
        dataset_id = "smarthealth_features"
    else:
        raise ValueError(
            f"PINN4SOH-noLeak-OnlyF has no validated statistical feature source for domain {domain_id!r}"
        )
    data_root = Path(data_cfg["data_root"])
    if not data_root.is_absolute():
        data_root = (Path(repo_root) / data_root).resolve()
    batches = list(config.get("experiment", {}).get("batches", []))
    if not batches:
        if dataset_id == "mit_features":
            batches = ["2017-05-12", "2017-06-30", "2018-04-12"]
        elif dataset_id == "smarthealth_features":
            batches = sorted({
                parse_file_identity(path)[0]
                for path in list_feature_csv_files(data_root, domain_id=domain_id)
            })
        else:
            batches = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
    split_file = data_cfg.get("split_file") or config.get("experiment", {}).get("split_file")
    if not split_file:
        raise ValueError(
            "F-only configuration must provide data.split_file or "
            "experiment.split_file; dataset split policy belongs in JSON."
        )
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = (Path(repo_root) / split_path).resolve()
    split_spec = load_split_spec(split_path)
    development_protocol = get_development_protocol(split_spec)
    if domain_id == "mit":
        # The baseline must use the same canonical physical cohort and JSON
        # test rule as RawMamba, even when invoked outside the shell launcher.
        mit_files = list_feature_csv_files(data_root, domain_id=domain_id)
        validate_mit_physical_cohort(
            (parse_file_identity(path)[1] for path in mit_files),
            split_spec,
            require_full_physical_cohort=bool(
                data_cfg.get("require_full_physical_cohort", False)
            ),
        )
    raw_rows = {"train": [], "val": [], "test": []}
    per_batch = {}
    feature_names = None
    label_provenance = {}

    def load_payloads(paths):
        payloads = [load_feature_file(path, config) for path in paths]
        for payload in payloads:
            if payload.get("label_provenance") is not None:
                label_provenance[str(payload["battery_id"])] = payload["label_provenance"]
        return payloads

    if dataset_id == "smarthealth_features":
        # v2 protocol: choose test logical sequences condition-by-condition,
        # then pool *all* development cycles in the family before one seed-420
        # mixed-cycle train/validation shuffle.  Do not turn C-rate/DOD into
        # separate baseline domains merely because their test cells differ.
        development_pool, all_test_rows = [], []
        for batch in batches:
            files = list_feature_csv_files(data_root, batch=batch, domain_id=domain_id)
            observed_batteries = [parse_file_identity(path)[1] for path in files]
            configured_test_batteries = resolve_test_batteries(
                split_spec,
                observed_battery_ids=observed_batteries,
                condition=batch,
            )
            train_val_files, test_files = split_feature_files_by_battery(
                files, test_batteries=configured_test_batteries
            )
            pool = build_adjacent_first_samples(
                load_payloads(train_val_files)
            )
            test_rows = build_adjacent_first_samples(
                load_payloads(test_files)
            )
            development_pool.extend(pool)
            all_test_rows.extend(test_rows)
            per_batch[batch] = {
                "development_samples_before_family_pool": len(pool),
                "test_samples": len(test_rows),
                "development_batteries": sorted({row["battery_id"] for row in pool}),
                "test_batteries": sorted({row["battery_id"] for row in test_rows}),
            }
        if len(development_pool) < 2 or not all_test_rows:
            raise ValueError("SmartHealth family-level split produced an empty development or test set")
        permutation = np.random.RandomState(
            development_protocol["random_state"]
        ).permutation(len(development_pool))
        n_val = max(
            1,
            int(np.ceil(development_protocol["val_ratio"] * len(development_pool))),
        )
        n_val = min(n_val, len(development_pool) - 1)
        raw_rows["train"] = [
            dict(development_pool[int(index)], split="train")
            for index in permutation[n_val:]
        ]
        raw_rows["val"] = [
            dict(development_pool[int(index)], split="val")
            for index in permutation[:n_val]
        ]
        raw_rows["test"] = [dict(row, split="test") for row in all_test_rows]
        feature_names = [*PINN16_FEATURE_COLUMNS, *TEMPERATURE_FEATURE_COLUMNS]
    else:
        for batch in batches:
            files = list_feature_csv_files(data_root, batch=batch, domain_id=domain_id)
            observed_batteries = [parse_file_identity(path)[1] for path in files]
            configured_test_batteries = resolve_test_batteries(
                split_spec,
                observed_battery_ids=observed_batteries,
                condition=batch,
            )
            train_val_files, test_files = split_feature_files_by_battery(
                files, test_batteries=configured_test_batteries
            )
            pool = build_adjacent_first_samples(load_payloads(train_val_files))
            test_rows = build_adjacent_first_samples(load_payloads(test_files))
            permutation = np.random.RandomState(
                development_protocol["random_state"]
            ).permutation(len(pool))
            n_val = max(
                1,
                int(np.ceil(development_protocol["val_ratio"] * len(pool))),
            )
            n_val = min(n_val, len(pool) - 1)
            val_indices = permutation[:n_val]
            train_indices = permutation[n_val:]
            current = {
                "train": [dict(pool[int(index)], split="train") for index in train_indices],
                "val": [dict(pool[int(index)], split="val") for index in val_indices],
                "test": [dict(row, split="test") for row in test_rows],
            }
            for split_name in raw_rows:
                raw_rows[split_name].extend(current[split_name])
            feature_names = feature_names or [*PINN16_FEATURE_COLUMNS, *TEMPERATURE_FEATURE_COLUMNS]
            per_batch[batch] = {
                "train_samples": len(current["train"]),
                "val_samples": len(current["val"]),
                "test_samples": len(current["test"]),
                "train_val_batteries": sorted({row["battery_id"] for row in pool}),
                "test_batteries": sorted({row["battery_id"] for row in test_rows}),
            }

    normalizer_override = (
        config.get("feature_normalizer_override")
        or data_cfg.get("feature_normalizer_override")
    )
    if normalizer_override is None:
        normalizer = fit_feature_minmax(
            raw_rows["train"] + raw_rows["val"],
            eps=float(data_cfg.get("normalization_eps", 1e-8)),
        )
        normalizer["fit_scope"] = "pooled_non_test_train_val_cycles"
    else:
        normalizer = copy.deepcopy(normalizer_override)
        normalizer["fit_scope"] = "source_domains_train_val_only"
    normalizer["feature_names"] = feature_names
    normalizer["test_statistics_used"] = False
    rows = {name: apply_feature_minmax(values, normalizer) for name, values in raw_rows.items()}
    debug_n = int(config.get("debug", {}).get("debug_num_samples", 0) or 0)
    if debug_n > 0:
        rows = {name: values[:debug_n] for name, values in rows.items()}
    datasets = {
        name: StatFeatureDataset(values, dataset_id=dataset_id, domain_id=domain_id)
        for name, values in rows.items()
    }
    batch_size = int(train_cfg.get("batch_size", 64))
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=int(data_cfg.get("num_workers", 0))),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=int(data_cfg.get("num_workers", 0))),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=int(data_cfg.get("num_workers", 0))),
    }
    return loaders, {
        "domain_id": domain_id,
        "dataset_id": dataset_id,
        "data_root": str(data_root),
        "data_mode": "all_batch_pooled",
        "batches": batches,
        "per_batch": per_batch,
        "feature_set": "pinn16_plus_temperature",
        "feature_names": feature_names,
        "input_dim": len(feature_names),
        "cycle_index_used_as_model_input": False,
        "sample_mode": "adjacent_x1",
        "drop_3sigma_outliers": bool(data_cfg.get("drop_3sigma_outliers", True)),
        "normalization": normalizer,
        "record_counts": {name: len(values) for name, values in rows.items()},
        "sample_counts": {name: len(datasets[name]) for name in datasets},
        "validation_split_mode": "mixed_cycle",
        "validation_split_scope": development_protocol["scope"],
        "val_ratio": development_protocol["val_ratio"],
        "val_random_state": development_protocol["random_state"],
        "test_battery_rule": split_spec.get("name", "dataset_split_spec"),
        "split_file": str(split_path) if split_path is not None else None,
        "invalid_cycle_policy": data_cfg.get(
            "invalid_cycle_policy", "source_native_all_column_3sigma"
        ),
        "label": {
            "label_mode": BOL_LABEL_MODE if is_bol_label_mode(config) else "rated_relative",
            "label_field": "soh_bol" if is_bol_label_mode(config) else "soh",
            "reference_rule": BOL_RULE_VERSION if is_bol_label_mode(config) else None,
            "reference_provenance": label_provenance,
            "q_ref_is_model_input": False,
            "q_ref_in_normalization": False,
        },
        "train_val_battery_overlap_expected": development_protocol[
            "train_val_battery_overlap_expected"
        ],
    }
