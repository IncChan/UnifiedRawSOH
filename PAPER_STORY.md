# Paper-v1: Raw → Unified → Reusable

## One sentence

Only the current charging cycle's terminal raw CC/CV electrical signal and
temperature evolution are inference inputs. The aim is to learn a
handcrafted-indicator-free SOH representation that first works within a
battery domain, then is shared across heterogeneous domains, and finally is
reused on an unseen domain with little or no target data.

The paper unit is a battery domain, not a generic dataset or each individual
operating condition.

| Domain ID | Paper alias | Battery family | Conditions within the same domain |
|---|---|---|---|
| xjtu | A | XJTU public single family | 2C, 3C, R2.5, R3, RW, satellite |
| mit | B | MIT public single family | source date groups |
| smarthealth_lishen40 | C1 | LISHEN 40 Ah LFP | C-rate / DOD |
| smarthealth_catl280 | C2 | CATL 280 Ah LFP | C-rate / DOD |
| smarthealth_eve280 | C3 | EVE 280 Ah LFP | C-rate / DOD |

Future enterprise E1/E2/E3 labels mean separate enterprise battery families.
Their code IDs must be semantic, not merely E1/E2/E3.

## Leakage boundary

At inference, the model receives only the current cycle's CC/CV signal,
relative time, and temperature. It never receives a true normalized cycle
coordinate, future cycles, an observed lifetime, or an handcrafted health
indicator.

The default raw model may use a full-life degradation coordinate as
training-only auxiliary supervision. That coordinate is constructed from the
complete training record to encourage a degradation-aware encoder, enters only
the auxiliary loss, and is never a model input. Its predicted coordinate can
be injected into the SOH head with the inherited no-detach C5B semantics.
This is a conditional training aid, not a claimed core contribution; E1
includes an SOH-only ablation.

## Paper model

The proposed Paper-v1 model retains the verified C5B implementation:

    phase-specific CC Mamba and CV Mamba
        → zero-initialized CC→CV bridge
        → last/mean pooled health feature
        → post-fusion T0 metadata
        → SOH head

ΔT is concatenated into phase tokens and T0 is post-fusion metadata. The
stable encode() interface returns z_health for E2 aggregation, visualization,
and future diagnostic methods. Default state-dict names and tensor shapes
remain compatible with existing C5B Paper-v1 checkpoints.

## E1 — Raw SOH Learning

Question: can raw CC/CV plus temperature learn SOH competitively, and is the
phase-specific design necessary?

Every domain is trained and tested independently:

    A→A, B→B, C1→C1, C2→C2, C3→C3

The handcrafted reference is PINN4SOH-noLeak-OnlyF. It remains isolated from
the raw pipeline, uses source-native statistical features, does not use cycle
as a model input, and does not use the PINN G/PDE branch. Planned benchmark
controls are LSTM, GRU, TCN, and true joint-sequence Vanilla Mamba. They are
not fabricated as runnable implementations before their protocol is verified.

E1 also owns all ablations:

    Temperature: V/I, V/I+ΔT, V/I+T0, V/I+T0+ΔT
    CC/CV: joint, independent phases, phase-specific + CC→CV bridge
    Backbone: Vanilla Mamba, proposed model
    Supervision: SOH only, SOH + degradation auxiliary

The currently implemented controls faithfully reuse the same raw model for
temperature removal/isolation, bridge-off independent phases, and SOH-only
supervision. Joint-sequence and true Vanilla controls stay marked planned
until implemented honestly.

## E2 — Unified Multi-domain Learning

Question: can one raw representation learner serve multiple heterogeneous
battery domains?

    Separate: A→Model-A, B→Model-B, C1→Model-C1, ...
    Unified: A+B+C1+C2+C3 → Model-U

E2 has explicit domain IDs in loader composition, split provenance,
normalization, balancing, and result aggregation. C-rate/DOD are conditions,
not implicit domains. Unified training can balance domains and batteries so
the largest domain or longest-lived cell does not dominate a batch. Report
Separate(domain) versus Unified(domain), not only a pooled-cycle score.

## E3 — Cross-domain Reusability

Question: does the unified representation help a battery domain completely
unseen in pretraining?

Two protocols are supported by configuration:

    leave-one-domain-out:
    A+B+C1+C2 → C3 (and any other source/target list)

    whole-source holdout:
    A+B → C1/C2/C3

For every unseen target, compare zero-shot and adaptation against scratch with
the identical target budget. A budget is explicitly either a cycle/sample
fraction or a physical-cell count; the choice is made per target domain rather
than forced globally.

## E4 — Industrial External Validation

Question: can the final public unified representation transfer to real
enterprise battery families?

    A+B+C1+C2+C3 → final public unified model → enterprise family

E4 supports zero-shot and 1-cell/2-cell/etc. adaptation versus scratch using
the same target cells. No enterprise adapter, data, split, or result is
invented before real source data are available.

## Conditional contrastive extension

Health-aware contrastive alignment is not a fifth paper experiment. It is a
conditional method extension only if E2 is materially worse than Separate or
z_health is observed to cluster mainly by domain. Its reserved location is
configs/conditional/health_aware_contrastive and
scripts/conditional/health_aware_contrastive.

## Current source status

XJTU and the continuation-aware 124 physical-cell MIT protocol use the same
common raw-cycle contract and retain fixed physical normalization and split
provenance. MIT's launcher rejects an incomplete local v3 phase-aware export;
it never falls back to a historical raw/feature-aligned table. Their paired
Only-F implementation remains independent, while its MIT statistics are
calculated from the same phase-aware accepted CC/CV points as the raw model.

SmartHealth has generated canonical raw and Only-F feature products with the
underlying GB18030 point-level source and representative CC boundary audited.
The source labels charge as one combined
constant-current/constant-voltage step. Its v2 code now infers the boundary,
selects CC 3.45–3.58 V and CV nominal 0.25C–0.05C, requires selected-point
temperature, derives calibration-only SOH, preserves logical sequence identity,
and emits split JSON. The adapter validates these exported products before it
builds samples; no fallback reads direct source CSVs.

## Training protocol

The Paper-v1 main protocol inherits the verified mixed C5B semantics: test
batteries/cells are independent; development cycles are mixed into train and
validation according to JSON-owned split provenance. Strict train/validation
battery separation is not a paper contribution. E1 uses condition-macro
validation RMSE for checkpoint selection; E2 uses domain-macro validation RMSE
while retaining per-domain and per-battery diagnostics.

Historical outputs remain in their original output namespace so their saved
checkpoints stay readable. New configs use the four paper experiment groups;
old paths are not used by active code.
