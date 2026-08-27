# Paper-v2 hierarchy and sampling

The hierarchy is fixed as:

```text
dataset/domain → strategy group → physical cell → cycle
```

Strategy metadata is read from explicit `strategy_group`, `condition`, or
`batch_name` fields. The implementation never guesses strategy from cycle
order or a filename. Missing metadata is a hard error.

`HierarchicalReplacementSampler` uniformly chooses each level in sequence and
uses a private seeded generator. It is used only for train; validation and test
DataLoaders are deterministic sequential loaders. `set_epoch(epoch)` makes a
`(seed, epoch)` pair reproducible independent of worker scheduling. Its audit
contains the source inventory and sampled domain/strategy/cell counts.

`SourceEpisodeBuilder` supports dataset-level pseudo-LODO and strategy-level
pseudo-domain episodes. The selected dataset/environment is held out as a
whole. If a physical cell spans multiple strategies, all of that cell's
cycles are held out and `cell_disjoint_expansion` records the stricter range.
E3 builders accept source train only and reject a target/undeclared domain.

| Component | Status | Config | Command | Output | Tests | Last verified | Limitations |
|---|---|---|---|---|---|---|---|
| HierarchyIndex | smoke-tested | E2/E3 sampler config | unit-test API | `sampling_audit.json` | `test_hierarchical_sampler.py` | 2026-08-27 | metadata must be present in the dataset |
| Four-level sampler | smoke-tested | `data.sampler.kind=hierarchical` | E2/E3 launcher | `sampling_audit.json` | reproducibility/imbalance tests | 2026-08-27 | replacement sampling is intentionally train-only |
| Pseudo-LODO episodes | smoke-tested | DG trainer config | E3 launcher | `episode_audit.json` | `test_episodic_dg.py` | 2026-08-27 | only source-internal episodes are implemented |
