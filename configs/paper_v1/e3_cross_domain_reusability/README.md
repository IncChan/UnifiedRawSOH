# E3 Cross-domain Reusability

E3 evaluates a representation on training-unseen battery domains:

1. `leave_one_domain_out/` — runnable five-fold no-cycle LODO, with four
   source domains for training/validation and one untouched target test domain;
2. `cross_dataset_holdout/` — planned A+B pretraining with all SmartHealth
   domains held out;
3. `adaptation/` — planned zero/few-shot/few-cell target reuse versus scratch
   at the identical target budget.

The LODO implementation has a dedicated split-routing loader and reuses the
Paper-v1 RawMamba trainer. It preserves every domain's existing split JSON:
source train/val are used for fitting and checkpoint selection, while only the
left-out target test split is used for final evaluation. Target train/val and
source test data are excluded.
