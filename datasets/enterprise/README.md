# Enterprise battery-domain entrance

E1/E2/E3 here mean future enterprise battery families, not the paper
experiment labels.  Add one stable semantic domain ID per real battery family
(for example enterprise_companyx_modely), never a vague E1 as the only code
identifier.

Before an enterprise domain can enter E4, provide its real raw adapter,
metadata, normalization specification, physical-cell split JSON, label policy,
and provenance audit.  The model inference contract remains current-cycle raw
CC/CV/T only.  No enterprise data, synthetic raw sequence, split, or
placeholder result is stored in this repository.

SMVIC is implemented as a stream-built preprocessed-v2 product under the
repository `datasets/` namespace.
See [`preprocess/SMVIC_MODEL_READY.md`](../../preprocess/SMVIC_MODEL_READY.md)
for its protocol audit, label contract, physical-cell splits, preprocessing,
validation, and training commands. Generated arrays and audits are Git-ignored;
only their code, semantic configs, and split specifications are versioned.
