# Paper-Backup offline preprocessing

This namespace materializes model-ready data without changing Paper V1 or V2.
Its versioned contract is `paper_backup_preprocessed_v1`.

Schema v2 is reserved for the isolated shared C-rate/FullVI experiment and is
written to a separate root. See
[`CRATE_V2_RUNBOOK.md`](../../docs/paper_backup/CRATE_V2_RUNBOOK.md).

For every canonical model-eligible Terminal cycle it writes fixed-length CC
and CV arrays, the 24 electrical/thermal statistics calculated from the
unresampled physical Terminal points, SOH, identity metadata, and an exclusion
audit. The rich sequence channel order is:

1. globally normalized voltage;
2. globally normalized absolute current;
3. normalized physical relative time;
4. absolute temperature normalization;
5. temperature delta from the cycle's first Terminal CC point;
6. phase-specific signal normalization (CC voltage or CV current);
7. phase coordinate `tau` in `[-1, 1]`.

Normalization uses fixed configuration constants and is not clipped. Feature
mean/std standardization is deliberately absent from the product and is fit by
the loader from the training split only.

FULL support in v1 covers the current E2 domains XJTU, LISHEN40 and CATL280.
FULL means the complete observed principal charging event split at its inferred
CC-to-CV boundary. It does not claim an unobserved 0–100% SOC interval. FULL
labels and identities are joined from the canonical Terminal cohort; a
Terminal record can never be promoted to FULL. SmartHealth source files are
scanned once per linked chunk, not once per cycle. Completely empty,
comma-only vendor padding rows are ignored; malformed rows containing any
source value still fail closed instead of being silently discarded.
SmartHealth FULL extraction streams one completed source file at a time and
uses bounded process parallelism, so unresampled long curves are not retained
for the entire domain. Progress is reported by completed source files and
materialized cycles. `PAPER_BACKUP_WORKERS` overrides `PREPROCESS_WORKERS`
from `preprocess/paths.env`; begin with 4 workers on shared storage.

Generated files live under
`datasets/PaperBackup_preprocessed/<domain>/` and are Git-ignored. Each domain
contains `manifest.json`, mmap-compatible `.npy` arrays, Terminal/FULL cycle
indices, `cohorts/full_matched_keys.csv`, and audit files.

## Commands

Configure the canonical and vendor roots in `preprocess/paths.env`, then run:

```bash
# Five-domain Terminal arrays and features only.
bash preprocess/paper_backup/run_preprocess.sh terminal

# Complete product: five-domain Terminal/features plus current E2 FULL domains.
bash preprocess/paper_backup/run_preprocess.sh all

# Validate shapes, finite values, checksums, identities and FULL subset linkage.
bash preprocess/paper_backup/run_preprocess.sh validate
```

Existing products are protected by default. To regenerate:

```bash
OVERWRITE=1 bash preprocess/paper_backup/run_preprocess.sh all
```

Configure SmartHealth FULL parallelism explicitly when needed:

```bash
PAPER_BACKUP_WORKERS=4 OVERWRITE=1 \
  bash preprocess/paper_backup/run_preprocess.sh all
```

Bounded diagnostics can set `MAX_RECORDS`, `CC_LEN` and `CV_LEN`; those products
are smoke artifacts and must not be used for the paper experiment:

```bash
MAX_RECORDS=8 CC_LEN=8 CV_LEN=8 OVERWRITE=1 \
  PAPER_BACKUP_PREPROCESSED_ROOT=/tmp/paper_backup_smoke \
  bash preprocess/paper_backup/run_preprocess.sh terminal
```

The paper contract uses 128 CC points and 256 CV points. E2 automatically uses
the exact FULL-matched physical-cycle cohort for all Terminal and FULL views.
