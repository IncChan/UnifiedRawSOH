# Paper-Backup models

`model_factory.py` is the only Paper-Backup dispatch surface. Unknown model
types fail immediately; a configuration name never aliases another model.

E1 currently exposes only:

* `HI-MLP`: the existing independent PINN4SOH-noLeak-OnlyF statistical-feature
  reference;
* `Transformer`: a masked encoder-only joint raw-sequence baseline;
* `Ours`: the existing phase-specific CC/CV model with cycle prediction,
  predicted-cycle injection, and cycle auxiliary supervision disabled.

`VanillaMamba` and `SingleStreamMamba` are implemented for the E2 charging-view
comparison. `RawCNN` and `LSTM` are implemented as honest reusable controls but
are not included in the current E1 configuration matrix.

The model forward methods receive tensors only. Battery, strategy, domain and
cycle identity are evaluation/sampling provenance and are never forwarded.
