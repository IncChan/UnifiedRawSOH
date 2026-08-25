# E2 Unified Multi-domain Learning

E2 asks whether heterogeneous **battery domains**, rather than merely
different operating conditions, can share one raw representation learner.

- `unified/public_xjtu_mit.json` is E2-Pilot: the runnable XJTU+MIT path.
- `unified/public_all_domains.json` is E2-Full: XJTU, MIT, LISHEN40, CATL280,
  and EVE280 in one shared model.
- `unified/public_xjtu_mit_domain_balanced_no_cycle_aux.json` is
  E2-Pilot-D w/o cycle auxiliary.
- `unified/public_all_domains_domain_balanced_no_cycle_aux.json` is
  E2-FULL-D w/o cycle auxiliary.
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

## Preprocessed cycle cache

The two `*_domain_balanced_no_cycle_aux.json` launch targets enable a
versioned cache for model-ready cycle samples. Each domain writes beneath its
own canonical data root by default:

```text
<dataset-root>/.cache/unified_cccv/<domain>-<fingerprint>.pt
```

The first seed holding the per-cache file lock parses the canonical CSVs,
applies the split/filter contract, and performs CC/CV interpolation. Concurrent
seeds wait for that atomic cache publication and then load the same immutable
samples. Model initialization and sampler randomness remain per-seed.

The fingerprint includes raw file path/size/mtime, the exact split and
preprocessing implementation contents, normalization, resampling, filtering,
and debug-sample settings. A changed input creates a new cache file instead of
silently reusing stale samples. Runtime controls live under
`data.preprocessed_cache`:

- `enabled`: use the cache when true;
- `directory`: absolute path, or a path relative to each dataset root;
- `rebuild`: ignore an existing matching entry and rebuild it. Use this with a
  single seed to avoid intentionally rebuilding once per concurrent process.

Cache files are trusted local PyTorch artifacts. Old fingerprints may be
removed when no training process is using the cache.
