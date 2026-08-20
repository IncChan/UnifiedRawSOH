# Data preprocessing boundary

Normalization in `normalization.py` is fixed physical/protocol normalization.
It does not fit statistics from any split. Dataset-specific parsing, raw source
provenance, segment handling, resampling, and label conventions belong in the
dataset adapters. The raw model receives only the common sample contract and
does not branch on dataset names. The cycle-life coordinate is an auxiliary
training target, never an inference input.
