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

## Summary statistics

For each group and variant, the script builds:

- summary rows with treated-minus-control effects
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

- Holm adjustment to the primary same-sign tests
- Benjamini-Hochberg adjustment to the opposite-sign secondary tests

Main knobs:

- `BOOTSTRAP_RUNS`
- `PERMUTATION_RUNS`
- `ALPHA`
- `SEED`

See [`bootstrap_methods.md`](bootstrap_methods.md) for the resampling details.

## Outputs

Tables are written under:

- `out_files/{DATASET_NAME}/{ANALYSIS_TAG}/`

Figures are written under:

- `images/{DATASET_NAME}/{ANALYSIS_TAG}/html/`
- `images/{DATASET_NAME}/{ANALYSIS_TAG}/png/`

Canonical table outputs:

- `event_study_summary.csv`
- `event_study_curves.csv`
- `event_study_diagnostics.csv`
- `robustness_same_actor_exclusion_summary.csv` when the robustness run is
  active
- Parquet versions of those tables when `WRITE_PARQUET=true`
- `run_manifest.json`

Canonical figure stems:

- `event_curve_prop_all_others`
- `event_curve_client_all_others`
- `event_curve_prop_exclude_same_actor`
- `event_curve_client_exclude_same_actor`

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
- `--seed`
- `--same-actor-key`
- `--plots/--no-plots`
- `--write-parquet/--no-write-parquet`
- `--show-progress/--no-show-progress`
- `--dry-run`

Example:

```bash
python scripts/metaorder_start_event_study.py \
  --analysis-tag metaorder_start_event_study \
  --high-eta-quantile 0.9 \
  --bootstrap-runs 1000 \
  --permutation-runs 1000
```

## Related docs

- [`imbalance_and_crowding.md`](imbalance_and_crowding.md)
- [`bootstrap_methods.md`](bootstrap_methods.md)
