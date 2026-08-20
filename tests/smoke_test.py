#!/usr/bin/env python3
"""Low-cost Paper-v1 adapter/forward/train-step smoke test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.xjtu import UnifiedCCCVSampleDataset, build_full_life_cycle_metadata, read_xjtu_file  # noqa: E402
from UnifiedRawSOH.models.c5b_model import build_c5b_model, get_mamba_backend_info  # noqa: E402
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


def main():
    torch.set_num_threads(1)
    config_path = PROJECT_ROOT / "UnifiedRawSOH/configs/e1_raw_soh_learning/benchmark/raw_mamba_xjtu.json"
    config = load_config(config_path)
    data_root = PROJECT_ROOT / "UnifiedRawSOH/datasets/XJTU_raw"
    if not data_root.is_dir() or not list(data_root.glob("*.csv")):
        raise RuntimeError("Paper-v1 smoke requires the XJTU_raw copy; run copy_datasets.sh first.")
    first_file = sorted(path for path in data_root.glob("*.csv") if not path.name.endswith("_report.csv"))[0]
    records = read_xjtu_file(first_file, nominal_capacity=2.0, label_scale_mode="auto_capacity_to_soh")
    cycle_metadata = build_full_life_cycle_metadata(records)
    dataset = UnifiedCCCVSampleDataset(
        records[:1],
        config["data"],
        config["normalization"],
        split_name="smoke",
        cycle_metadata=cycle_metadata,
    )
    item = dataset[0]
    batch = {key: value.unsqueeze(0) for key, value in item.items() if torch.is_tensor(value)}
    model = build_c5b_model(config["model"], backend_override="torch_reference")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    aux = model.forward_with_aux(
        cc_signal=batch["cc_signal"],
        cv_signal=batch["cv_signal"],
        cc_mask=batch["cc_mask"],
        cv_mask=batch["cv_mask"],
        cc_time=batch["cc_time"],
        cv_time=batch["cv_time"],
        cc_temperature=batch["cc_temperature"],
        cv_temperature=batch["cv_temperature"],
        t0_temperature_norm=batch["t0_temperature_norm"],
    )
    criterion = torch.nn.MSELoss()
    soh_loss = criterion(aux["soh_pred"], batch["soh"])
    cycle_loss = criterion(aux["cycle_life_hat"], batch["cycle_life_norm_target"])
    loss = soh_loss + 0.0035 * cycle_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    result = {
        "status": "PASS",
        "backend_used": "torch_reference",
        "official_backend_info": get_mamba_backend_info(),
        "source_file": str(first_file),
        "sample_contract": ["cc", "cv", "t0", "soh", "battery_id", "dataset_id", "domain_id"],
        "cc_shape": list(item["cc"].shape),
        "cv_shape": list(item["cv"].shape),
        "z_health_shape": list(aux["z_health"].shape),
        "signal_feature_shape": list(aux["signal_feature"].shape),
        "soh_prediction_shape": list(aux["soh_pred"].shape),
        "cycle_prediction_shape": list(aux["cycle_life_hat"].shape),
        "loss": float(loss.detach().item()),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
