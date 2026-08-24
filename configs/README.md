# Configuration versions

- `paper_v1/` contains the existing E1-E4 experiment contracts and conditional
  extensions. Moving them here changes no dataset, split, label, or output ID.
- `paper_v2/` is reserved for health-aligned pretraining and few-cell transfer.

Shared raw products remain under `datasets/`, and split provenance remains
under `splits/`; neither is duplicated between paper versions.
