# Paper-Backup data views

`preprocessed.py` is the default v1 reader. It memory-maps the offline arrays
written by `preprocess/paper_backup`, copies only the requested row, and does
not interpolate, normalize, or open vendor files during training.
The production loaders use one persistent worker, pinned host memory and a
prefetch factor of two by default. `NUM_WORKERS=0` on the E1 launcher provides
a conservative synchronous fallback.

`sequence_views.py` retains the explicit Paper-Backup batch schema. Terminal joint/CC/CV/phase views all preserve
the same battery, cycle, label and split identity. The Transformer and
Vanilla/single-stream models receive only normalized voltage/current, relative
time and temperature tensors.

`full_cccv.py` is now primarily a preprocessing-time source linker and remains
intentionally fail-closed. A terminal product is never
treated as full data. E2 full runs require an explicitly configured source and
pair it with terminal cycles using `(physical battery ID, cycle ID)`, recording
missing and label-mismatch exclusions. The module supports the source-linked
XJTU/SmartHealth formats and an explicit normalized full-CSV export.

The old runtime path remains available only through
`data.source_mode=legacy_runtime` for numerical regression. Production configs
use `preprocessed_v1` and never fall back silently.

`strategy_pooling.py` obtains strategy IDs from canonical `condition` metadata.
It builds independent strategy splits and a pooled union. Pooled training uses
equal strategy mass, then equal battery mass, then equal cycle mass; strategy
metadata remains outside model forward.
