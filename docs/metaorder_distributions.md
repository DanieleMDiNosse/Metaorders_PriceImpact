# Metaorder Distributions

This document covers `scripts/metaorder_distributions.py`, which builds the
combined client-vs-proprietary distribution diagnostics and the optional
tail-model overlays.

## Purpose

The script loads the canonical proprietary and client metaorder dictionaries,
reconstructs comparable samples from the per-ISIN trade tapes, and exports:

- one multi-panel comparison figure
- machine-readable fit summaries
- plot data that can regenerate the figure without rereading raw tapes
- a review table in CSV and Markdown form

## Inputs

Default config:

- `config_ymls/metaorder_distributions.yml`

Environment-variable override:

- `METAORDER_DISTRIBUTIONS_CONFIG`

Main required inputs:

- proprietary dictionary:
  `metaorders_dict_all_{LEVEL}_proprietary[ _member_nationality_{...} ].pkl`
- client dictionary:
  `metaorders_dict_all_{LEVEL}_non_proprietary[ _member_nationality_{...} ].pkl`
- per-ISIN trade tapes under `PARQUET_PATH`

The script shares the filtering logic in
`moimpact/metaorder_distribution_samples.py`, including:

- trading-hours filtering
- proprietary-vs-client filtering
- optional member-nationality filtering

## Metrics shown

The combined figure has five panels:

1. metaorder duration
2. inter-arrival times within metaorders
3. metaorder volume `Q`
4. relative size `Q/V`
5. participation rate `eta`

Each panel compares proprietary and client samples on the same layout.

## Tail-model fitting

When `POWERLAW_FIT_ENABLED=true`, the script fits a power-law tail to each
panel and can compare it against:

- `lognormal`
- `exponential`
- `truncated_power_law`

The implementation uses shared helpers in `moimpact/power_law_fits.py`.

Relevant config keys:

- `POWERLAW_FIT_METHOD`
- `POWERLAW_MIN_TAIL`
- `POWERLAW_NUM_CANDIDATES`
- `POWERLAW_REFINE_WINDOW`
- `POWERLAW_COMPARE_ENABLED`
- `POWERLAW_COMPARE_ALTERNATIVES`

Two fitting modes are supported:

- `approx`: repository coarse-to-local search for `xmin`
- `powerlaw`: delegate `xmin` search to the `powerlaw` package

## Bootstrap summaries

When `POWERLAW_FULL_BOOTSTRAP_ENABLED=true`, the script reruns the full
tail-fitting pipeline on bootstrap resamples of each panel.

These bootstrap results feed:

- annotation intervals in the figure
- the compact fit-summary table
- the Markdown review note

Key config keys:

- `POWERLAW_FULL_BOOTSTRAP_RUNS`
- `POWERLAW_FULL_BOOTSTRAP_ALPHA`
- `POWERLAW_FULL_BOOTSTRAP_RANDOM_STATE`
- `BOOTSTRAP_DISTRIBUTION`
- `POWERLAW_FIT_MAX_WORKERS`

Parallelism:

- panel fits can run through `ProcessPoolExecutor`
- worker count is capped conservatively in code when full bootstrap is enabled

## Saved artifacts

The figure is written under:

- `images/{DATASET_NAME}/{LEVEL}_metaorder_distributions/html/`
- `images/{DATASET_NAME}/{LEVEL}_metaorder_distributions/png/`

The summary tables live under:

- `out_files/{DATASET_NAME}/{LEVEL}_metaorder_distributions/`

Canonical stems:

- figure:
  `metaorder_distributions_prop_vs_client[ _member_nationality_{...} ]`
- fit summary:
  `metaorder_distribution_fit_summary_prop_vs_client[ _member_nationality_{...} ]`
- review:
  `metaorder_distribution_fit_review_prop_vs_client[ _member_nationality_{...} ]`
- saved plot data:
  `metaorder_distribution_plot_data_prop_vs_client[ _member_nationality_{...} ]`

The script writes:

- fit summary CSV and Parquet
- plot data CSV and Parquet
- review CSV
- review Markdown

## Load mode

The script supports a saved-output regeneration mode:

```bash
python scripts/metaorder_distributions.py --load
```

In that mode it rebuilds the Plotly figure from the saved fit-summary and
plot-data artifacts only. This is useful when the raw dictionaries or tapes are
slow to reload and the goal is only to regenerate the figure.

## Typical usage

From the repo root:

```bash
source /home/danielemdn/miniconda3/etc/profile.d/conda.sh
conda activate main
python scripts/metaorder_distributions.py
```

To point at another YAML:

```bash
METAORDER_DISTRIBUTIONS_CONFIG=config_ymls/metaorder_distributions.yml \
python scripts/metaorder_distributions.py
```

## Related docs

- [`metaorder_summary_statistics.md`](metaorder_summary_statistics.md)
- [`market_impact.md`](market_impact.md)
- [`bootstrap_methods.md`](bootstrap_methods.md)
