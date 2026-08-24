# E2 Unified Multi-domain Learning

E2 asks whether heterogeneous **battery domains**, rather than merely
different operating conditions, can share one raw representation learner.

- `unified/public_xjtu_mit.json` is E2-Pilot: the runnable XJTU+MIT path.
- `unified/public_all_domains.json` is E2-Full: XJTU, MIT, LISHEN40, CATL280,
  and EVE280 in one shared model.
- `separate/` points back to the E1 domain-specific benchmark configs instead
  of copying training logic.

Both configurations inherit `raw_mamba_xjtu.json`, so the RawMamba structure,
optimizer, scheduler, batch size, epoch limit, patience, and cycle auxiliary
loss are identical to E1. They override only experiment composition,
per-domain data contracts, output identity, and the checkpoint monitor
(`valid_domain_macro_rmse`). Each domain keeps its E1 split and fixed
physical normalization. The corresponding launcher performs a read-only E1
raw-data readiness check for every participating domain before training.

The model exposes `encode()`/`z_health`; result aggregation stores
`per_domain` metrics and domain-macro validation RMSE for this stage.

Each completed seed also writes `test_metrics_by_domain.json`. Its `loss`,
`soh_loss`, `mae`, `mape`, `mse`, and `rmse` fields use the same test-time
meaning as E1. The batch aggregator writes
`summary_per_domain_mean_std.json` and `.csv`, giving one mean/std row per
domain and metric; the overall summary remains available separately.
