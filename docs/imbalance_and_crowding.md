# Imbalance and Crowding

This document describes the crowding workflows implemented in:

- `scripts/crowding_analysis.py`
- `scripts/crowding_vs_part_rate.py`
- shared helpers in `moimpact/stats/correlation.py`

The focus is the current code path: what imbalance variables are attached, how
the main correlations are defined, what is bootstrapped or permuted, and which
tables and figures are written.

## Inputs and outputs

`scripts/crowding_analysis.py` expects the filtered per-metaorder tables
produced by `scripts/metaorder_computation.py`. By default it loads:

- `out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_member_proprietary.parquet`
- `out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_member_non_proprietary.parquet`

Main outputs:

- figures under `images/{DATASET_NAME}/prop_vs_nonprop/`
- log file under `out_files/{DATASET_NAME}/logs/`
- optional participation-conditioned outputs under
  `out_files/{DATASET_NAME}/crowding_vs_part_rate/` and
  `images/{DATASET_NAME}/crowding_vs_part_rate/`

Important implementation detail:

- if a required imbalance or return column is missing, the script computes it
  and may rewrite the input parquet files to persist the new columns

## Core metaorder columns

The crowding code works on per-metaorder rows with fields such as:

- `ISIN`
- `Date`
- `Direction`
- `Q`
- `Period`
- `Member`
- `Participation Rate`

The key signed-volume object is:

`Q_i * Direction_i`

and the generic imbalance template is:

`sum(Q_i * Direction_i) / sum(Q_i)`

with `NaN` when the denominator is zero.

## Within-group imbalance

The within-group crowding variable is a leave-one-out imbalance computed on
`(ISIN, Date)` within the same flow group.

Interpretation:

- positive correlation between `Direction` and the leave-one-out imbalance
  means the metaorder tends to align with the same-group environment
- negative correlation means it tends to trade against the same-group
  environment

This is the default "local" crowding notion used in the repo.

## Cross-group environment imbalance

The cross-group crowding variable is built from the other group on the same
`(ISIN, Date)`:

- proprietary rows can receive `imbalance_client_env`
- client rows can receive `imbalance_prop_env`

This answers whether proprietary directions align with client-side signed
imbalance, and vice versa.

## All-others imbalance

`scripts/crowding_vs_part_rate.py` can also compute an all-others imbalance on
the concatenated proprietary + client sample, then remove the focal metaorder's
own contribution.

This is available through the `all` imbalance kind in the participation-rate
workflow.

## Member-level client environment

`scripts/crowding_analysis.py` includes an optional member-level measure that
aggregates client flow at `(Member, Date)` and relates it to the proprietary
directions of the same member.

This is a member-day analysis, not an ISIN-day analysis.

Config keys that govern this block include:

- `MIN_METAORDERS_PER_MEMBER`
- `N_MIN_PER_MEMBER_CLIENT`
- `MEMBER_WINDOW_DAYS`
- `BOOTSTRAP_HEATMAP`
- `P_VALUE_THRESHOLD`

## Correlation and inference

The shared inference helpers live in `moimpact/stats/correlation.py`.

The repository uses:

- Pearson correlations as the default alignment statistic
- date-cluster bootstrap confidence intervals
- permutation p-values in workflows that explicitly request them

For the main crowding script, the important config keys are:

- `ALPHA`
- `BOOTSTRAP_RUNS`
- `MIN_N`
- `SMOOTHING_DAYS`

The main analysis reports both:

- pooled/global correlations across all usable observations
- mean daily correlations after grouping by `Date`

## Daily returns and diagnostics

When `ATTACH_DAILY_RETURNS=true`, `scripts/crowding_analysis.py` derives
close-to-close daily log returns from the per-ISIN trade tapes and attaches the
column named by `DAILY_RETURN_COL`.

Optional diagnostics controlled by the YAML include:

- `PLOT_IMBALANCE_VS_RETURNS`
- `ACF_IMBALANCE`
- `DISTRIBUTIONS_IMBALANCE`
- histogram and KDE settings

## Crowding vs participation rate

The participation-conditioned analysis is implemented in
`scripts/crowding_vs_part_rate.py`. It can be run directly or triggered from
`scripts/crowding_analysis.py`.

It bins metaorders by `Participation Rate` and estimates bin-level statistics
such as:

- mean alignment `E[Direction * imbalance]`
- mean absolute imbalance
- top-minus-bottom effects
- monotonic trend summaries

It supports:

- local imbalance
- cross-group imbalance
- all-others imbalance
- optional regressions
- optional 2D `Q/V x eta` heatmaps
- optional within-date permutation tests

Important config keys exposed through `crowding_analysis.yml`:

- `CROWDING_VS_PART_RATE_ANALYSIS_TAG`
- `CROWDING_VS_PART_RATE_IMBALANCE_KIND`
- `CROWDING_VS_PART_RATE_ETA_BINS`
- `CROWDING_VS_PART_RATE_ETA_BINNING`
- `CROWDING_VS_PART_RATE_MIN_N_BIN`
- `CROWDING_VS_PART_RATE_BOOTSTRAP_RUNS`
- `CROWDING_VS_PART_RATE_CLUSTER_CI`
- `CROWDING_VS_PART_RATE_PERMUTATION_RUNS`
- `CROWDING_VS_PART_RATE_RUN_REGRESSIONS`
- `CROWDING_VS_PART_RATE_RUN_2D`

The script writes a `run_manifest.json` for traceability.

## Common artifacts

`scripts/crowding_analysis.py` writes figures such as:

- daily crowding time series
- rolling crowding curves
- cross-group crowding plots
- all-vs-all crowding plots
- imbalance distributions
- imbalance-vs-return scatter plots
- member-level crowding histogram and heatmap

`scripts/crowding_vs_part_rate.py` writes tables such as:

- `bin_summary_prop_{imbalance_kind}.csv`
- `bin_summary_client_{imbalance_kind}.csv`
- `daily_bin_panel_prop_{imbalance_kind}.csv`
- `daily_bin_panel_client_{imbalance_kind}.csv`
- `effect_sizes_{imbalance_kind}.csv`
- `regression_results_{imbalance_kind}.csv` when regressions are enabled
- `heatmap_align_qv_eta_{group}_{imbalance_kind}.csv` when 2D mode is enabled

## Running

From the repo root:

```bash
source /home/danielemdn/miniconda3/etc/profile.d/conda.sh
conda activate main
python scripts/crowding_analysis.py
```

Direct participation-rate workflow:

```bash
python scripts/crowding_vs_part_rate.py --analysis-tag crowding_vs_part_rate
```

## Related docs

- [`bootstrap_methods.md`](bootstrap_methods.md)
- [`metaorder_start_event_study.md`](metaorder_start_event_study.md)
- [`market_impact.md`](market_impact.md)
- [`PLOTTING_GUIDE.md`](PLOTTING_GUIDE.md)
