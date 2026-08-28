# SmartHealth v6 legacy extractor

This directory freezes the superseded `smarthealth_cccv_calibration_v6` phase
detector. It is retained for ablation and result reproduction; canonical
preprocessing does not import it.

The old policy required 60 inferred CC points and 60 inferred CV points by
default. EVE used a 30-point CV override, and taper onset itself had to persist
for 30 points. Those counts depend on logger cadence rather than battery
physics, so valid EVE tails containing 5–29 samples could disappear.

`extraction.py` contains the runnable old detector and its exact defaults. The
v6 label behavior also allowed a partial-DOD full-capacity calibration cycle to
be exported when its charge window passed. v7 deliberately does not preserve
that behavior in the canonical path: such cycles are anchor-only.

For a v6/v7 phase comparison, construct the normal SmartHealth parser
namespace, replace its four phase-count settings with the `LEGACY_DEFAULT_*`
constants, resolve the EVE CV threshold with `legacy_min_cv_points`, and call
`split_combined_charge_v6`.
