# Metaorder Start Event-Study

This document covers `scripts/run_analysis.py metaorders start-event`, which measures
whether high-participation metaorders tend to start in locally crowded
neighborhoods and whether their starts are followed by elevated nearby start
intensity.

The workflow is phrased as an event-study of start intensity, not as a causal
claim about who triggered whom.

## Inputs and config

Default config:

- `config_ymls/metaorder_start_event_study.yml`

Main environment-free entrypoint:

```bash
python scripts/run_analysis.py metaorders start-event
```

Alternate YAML:

```bash
python scripts/run_analysis.py metaorders start-event --config config_ymls/metaorder_start_event_study.yml
```

By default the workflow loads the filtered proprietary and client metaorder
tables from `out_files/{DATASET_NAME}/`, but both input paths can be overridden
through CLI flags or YAML.

## Design

The current workflow runs separately for proprietary and client metaorders.

Core design choices:

- anchor events are metaorders in the top `HIGH_ETA_QUANTILE` of
  `Participation Rate` within each group
- optional threshold sweeps can run over `HIGH_ETA_QUANTILES` or
  `--high-eta-quantiles`
- neighboring starts are restricted to the same `ISIN` and `Date`
- controls are matched exactly on:
  - `ISIN`
  - `Date`
  - clock bucket of width `MATCHING_BUCKET_MINUTES`
- event time spans `+- EVENT_WINDOW_MINUTES`
- bins have width `BIN_MINUTES`
- outcomes are split into:
  - `same_sign`
  - `opposite_sign`

## Event Metrics

For each group and variant, the workflow builds:

- event-time curves of treated and control start intensity
- diagnostics on matching, simultaneous starts, and boundary truncation

The primary variant is:

- `all_others`

Optional robustness variant:

- `exclude_same_actor`

The same-actor exclusion uses:

- `Member` when member-level tables are available
- otherwise `Client`
- or no exclusion if `SAME_ACTOR_KEY=none`

## Inference

Two resampling schemes are used:

- date-cluster bootstrap for confidence intervals
- within-stratum permutation of the high-`eta` label for p-values

The workflow then applies:

- Benjamini-Hochberg adjustment to the bin-by-bin curve tests, separately
  within each `(group, variant, sign relation)` family

For the event-time curves, the permutation test is carried out bin by bin on the
treated-minus-control excess rate. Those bin-level tests are two-sided so the
output table can flag both excesses and deficits relative to matched controls.

Main knobs:

- `BOOTSTRAP_RUNS`
- `PERMUTATION_RUNS`
- `N_JOBS`
- `ALPHA`
- `SEED`

See [`bootstrap_methods.md`](bootstrap_methods.md) for the resampling details.

For performance, the workflow parallelizes the independent `(ISIN, Date)` event
windows and the bootstrap/permutation replicate batches when `N_JOBS` is
greater than 1 or set to `0` for the auto mode. The event-window counting loop
also uses `numba` when it is available in the active Python environment.

When multiple `high_eta` quantiles are requested, the workflow reuses the
materialized event-window metrics and matching strata across thresholds. In
that sweep mode it switches to outer threshold-level parallelism and keeps the
per-threshold bootstrap/permutation workers serial to avoid nested executor
contention.

## Outputs

Tables are written under:

- `out_files/{DATASET_NAME}/{ANALYSIS_TAG}/`

Figures are written under:

- `images/{DATASET_NAME}/{ANALYSIS_TAG}/html/`
- `images/{DATASET_NAME}/{ANALYSIS_TAG}/png/`

Canonical table outputs:

- `event_study_curves.csv`
- `event_study_diagnostics.csv`
- `event_study_curves_threshold_sweep.csv` for multi-quantile runs
- `event_study_diagnostics_threshold_sweep.csv` for multi-quantile runs
- Parquet versions of those tables when `WRITE_PARQUET=true`
- `run_manifest.json`

The curve table includes the treated, control, and excess rates for each event
bin, bootstrap confidence intervals for the excess, permutation-based
`p_raw` / `p_adjusted` columns for the bin-level excess tests, and per-bin
effective support columns. It now also carries:

- `high_eta_quantile`
- `eta_threshold`
- `n_high_eta`

Per-bin support columns:

- `n_treated_effective`
- `n_control_effective`
- `n_valid_strata`

These support columns reflect the number of treated observations, control
observations, and matching strata that are actually informative for that bin
after exposure truncation and finite-value filtering.

Canonical figure stems:

- `event_curve_prop_all_others`
- `event_curve_client_all_others`
- `event_curve_prop_exclude_same_actor`
- `event_curve_client_exclude_same_actor`
- `event_heatmap_effect_{group}_{variant}_{sign_relation}`
- `event_heatmap_padj_{group}_{variant}_{sign_relation}`
- `event_heatmap_support_{group}_{variant}_{sign_relation}`

When plots are enabled, bins with adjusted permutation `p < ALPHA` are marked
with a star above the corresponding panel.

For multi-quantile runs, the heatmap figures are the primary visual output:

- effect heatmaps show `excess_rate` over `(high-eta quantile, event-time bin)`
- adjusted-p heatmaps show the within-threshold BH-adjusted p-values
- support heatmaps show `n_valid_strata`, with hover text exposing the
  effective treated/control counts

Heatmap families with no finite values are skipped. In particular, when
`PERMUTATION_RUNS=0`, the adjusted-p heatmaps are omitted rather than exported
as blank figures.

## Useful CLI overrides

The command exposes a full CLI. Common overrides include:

- `--dataset-name`
- `--analysis-tag`
- `--event-window-minutes`
- `--bin-minutes`
- `--matching-bucket-minutes`
- `--high-eta-quantile`
- `--high-eta-quantiles`
- `--bootstrap-runs`
- `--permutation-runs`
- `--n-jobs`
- `--seed`
- `--same-actor-key`
- `--plots/--no-plots`
- `--write-parquet/--no-write-parquet`
- `--progress/--no-progress`
- `--dry-run`

Example:

```bash
python scripts/run_analysis.py metaorders start-event \
  --analysis-tag metaorder_start_event_study \
  --high-eta-quantiles 0.5,0.6,0.7,0.8,0.9 \
  --event-window-minutes 20 \
  --n-jobs 0 \
  --bootstrap-runs 1000 \
  --permutation-runs 1000
```

## Related docs

- [`imbalance_and_crowding.md`](imbalance_and_crowding.md)
- [`bootstrap_methods.md`](bootstrap_methods.md)
