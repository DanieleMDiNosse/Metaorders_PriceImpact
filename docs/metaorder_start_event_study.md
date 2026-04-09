# Metaorder Start Event-Study

This document covers `scripts/metaorder_start_event_study.py`, which measures
whether high-participation metaorders tend to start in locally crowded
neighborhoods and whether their starts are followed by elevated nearby start
intensity.

The script is phrased as an event-study of start intensity, not as a causal
claim about who triggered whom.

## Inputs and config

Default config:

- `config_ymls/metaorder_start_event_study.yml`

Main environment-free entrypoint:

```bash
python scripts/metaorder_start_event_study.py
```

By default the script loads the filtered proprietary and client metaorder
tables from `out_files/{DATASET_NAME}/`, but both input paths can be overridden
through CLI flags or YAML.

## Design

The current workflow runs separately for proprietary and client metaorders.

Core design choices:

- anchor events are metaorders in the top `HIGH_ETA_QUANTILE` of
  `Participation Rate` within each group
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

## Outputs

For each group and variant, the script builds:

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

The script then applies:

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

For performance, the script parallelizes the independent `(ISIN, Date)` event
windows and the bootstrap/permutation replicate batches when `N_JOBS` is
greater than 1 or set to `0` for the auto mode. The event-window counting loop
also uses `numba` when it is available in the active Python environment.

## Outputs

Tables are written under:

- `out_files/{DATASET_NAME}/{ANALYSIS_TAG}/`

Figures are written under:

- `images/{DATASET_NAME}/{ANALYSIS_TAG}/html/`
- `images/{DATASET_NAME}/{ANALYSIS_TAG}/png/`

Canonical table outputs:

- `event_study_curves.csv`
- `event_study_diagnostics.csv`
- Parquet versions of those tables when `WRITE_PARQUET=true`
- `run_manifest.json`

The curve table includes the treated, control, and excess rates for each event
bin, bootstrap confidence intervals for the excess, and permutation-based
`p_raw` / `p_adjusted` columns for the bin-level excess tests.

Canonical figure stems:

- `event_curve_prop_all_others`
- `event_curve_client_all_others`
- `event_curve_prop_exclude_same_actor`
- `event_curve_client_exclude_same_actor`

When plots are enabled, bins with adjusted permutation `p < ALPHA` are marked
with a star above the corresponding panel.

## Useful CLI overrides

The script exposes a full CLI. Common overrides include:

- `--dataset-name`
- `--analysis-tag`
- `--event-window-minutes`
- `--bin-minutes`
- `--matching-bucket-minutes`
- `--high-eta-quantile`
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
python scripts/metaorder_start_event_study.py \
  --analysis-tag metaorder_start_event_study \
  --high-eta-quantile 0.9 \
  --event-window-minutes 20 \
  --n-jobs 0 \
  --bootstrap-runs 1000 \
  --permutation-runs 1000
```

## Related docs

- [`imbalance_and_crowding.md`](imbalance_and_crowding.md)
- [`bootstrap_methods.md`](bootstrap_methods.md)
