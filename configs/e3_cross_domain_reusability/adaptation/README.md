# E3 adaptation

Zero-shot, few-shot, and few-cell are one reusability framework.  A target
budget is explicit and uses one of:

- `cycle_fraction` — a fraction of the target development samples;
- `physical_cell_count` — a number of target cells/batteries.

Every adaptation configuration must also compare `pretrained + target budget`
with `scratch + the same target budget`.  The two supplied MIT examples are
planned interfaces; no adaptation trainer or result is claimed yet.
