# Paper-Backup data views

`sequence_views.py` consumes the existing canonical terminal RAW adapters and
emits a separate batch schema. Terminal joint/CC/CV/phase views all preserve
the same battery, cycle, label and split identity. The Transformer and
Vanilla/single-stream models receive only normalized voltage/current, relative
time and temperature tensors.

`full_cccv.py` is intentionally fail-closed. A terminal product is never
treated as full data. E2 full runs require an explicitly configured source and
pair it with terminal cycles using `(physical battery ID, cycle ID)`, recording
missing and label-mismatch exclusions. The module supports the source-linked
XJTU/SmartHealth formats and an explicit normalized full-CSV export.

`strategy_pooling.py` obtains strategy IDs from canonical `condition` metadata.
It builds independent strategy splits and a pooled union. Pooled training uses
equal strategy mass, then equal battery mass, then equal cycle mass; strategy
metadata remains outside model forward.
