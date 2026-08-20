# E1 benchmark configurations

E1 evaluates one **battery domain** at a time. XJTU and SmartHealth C1/C2/C3
have canonical raw and Only-F products. MIT uses the same config interface,
but its phase-aware paired physical124 raw/feature exports must pass the
launcher readiness check before a run is started. The official MIT configs
reject a partial cell copy rather than silently changing the published cohort
or its `%5` test rule:

- `raw_mamba_xjtu.json`, `raw_mamba_mit.json`: the Paper-v1 phase-specific
  raw CC/CV/T model;
- `raw_mamba_smarthealth_lishen40.json`,
  `raw_mamba_smarthealth_catl280.json`, and
  `raw_mamba_smarthealth_eve280.json`: the same model with each family's fixed
  physical normalization;
- `pinn4soh_onlyf_xjtu.json`, `pinn4soh_onlyf_mit.json`: the independent
  handcrafted statistical-feature reference.
- `pinn4soh_onlyf_smarthealth_lishen40.json`,
  `pinn4soh_onlyf_smarthealth_catl280.json`, and
  `pinn4soh_onlyf_smarthealth_eve280.json`: canonical SmartHealth feature
  references using the exported, per-series SOH label rather than nominal-Ah
  renormalization.

The benchmark matrix also reserves LSTM, GRU, TCN, and a true joint-sequence
Vanilla Mamba control.  Those implementations are not present in this
repository yet, so no configuration pretends they are runnable.  Add each
only after its preprocessing and checkpoint-selection protocol are verified
against this E1 contract.

SmartHealth v2 raw input is emitted by its matching
`process_smarthealth_<family>_raw.py`; the matching
`extract_smarthealth_<family>_features.py` consumes canonical RAW/provenance
only. Its contract records the CC/CV boundary, calibration SOH (no EOL),
source-file overlap choice, logical-sequence identity, and the per-condition
2-development/1-test split. Enterprise domains are added only once real data
are available.
