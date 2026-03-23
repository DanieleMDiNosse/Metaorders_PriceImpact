# Market Impact Workflow

This document covers the impact-analysis code paths centered on:

- `scripts/metaorder_computation.py`
- `scripts/plot_prop_nonprop_fits.py`
- `scripts/metaorder_intraday_analysis.py`
- shared fitting helpers in `moimpact/impact_fits.py`

The emphasis here is on the live implementation: how metaorders are built,
which normalizations are available, how the one-dimensional and bivariate fits
are estimated, and which artifacts are exported.

## Inputs and main outputs

`scripts/metaorder_computation.py` can start from:

- raw CSVs in `CSV_LOAD_PATH`
- per-ISIN trade tapes in `PARQUET_PATH`

Current defaults are defined in
`config_ymls/metaorder_computation.yml`.

The script writes, under `out_files/{DATASET_NAME}/`:

- `metaorders_dict_all_{LEVEL}_{group}[ _member_nationality_{it|foreign} ].pkl`
- `metaorders_info_sameday_{...}.parquet`
- `metaorders_info_sameday_filtered_{...}.parquet`

and, under `images/{DATASET_NAME}/{LEVEL}_{group}/`:

- power-law and logarithmic fit figures
- buy-vs-sell fit figures when `SPLIT_BY_SIDE=true`
- impact-surface heatmaps and 3D HTML surfaces
- normalized impact-path figures when `COMPUTE_IMPACT_PATHS=true`

The optional member-nationality filter is controlled by
`MEMBER_NATIONALITY` and propagates into filenames through the suffix
`_member_nationality_it` or `_member_nationality_foreign`.

## Metaorder reconstruction

At a fixed `LEVEL` (`member` or `client`), the code works ISIN by ISIN and day
by day on aggressive trades only.

High-level reconstruction logic:

1. map raw codes and build the canonical trade view with `utils.map_trade_codes`
   and `utils.build_trades_view`
2. filter to the configured trading session, default `09:30:00` to `17:30:00`
3. select proprietary or client flow using `PROPRIETARY`
4. build same-agent signed activity through `utils.agents_activity_sparse`
5. detect same-sign runs with `utils.find_metaorders`
6. split runs when inter-trade gaps exceed `MAX_GAP`
7. keep only same-day, single-client runs with at least `MIN_TRADES` child
   trades and duration at least `SECONDS_FILTER`

The main reconstruction knobs live in `metaorder_computation.yml`:

- `LEVEL`
- `PROPRIETARY`
- `MAX_GAP`
- `MIN_TRADES`
- `SECONDS_FILTER`
- `TRADING_HOURS`
- `MEMBER_NATIONALITY`

## Per-metaorder quantities

For each metaorder `i`, the pipeline builds the usual size and impact fields:

- `Q`: metaorder volume
- `Q/V`: relative size, where the denominator is configurable
- `Participation Rate`: metaorder volume divided by traded volume during the
  execution window
- signed impact horizons such as `Impact_1m`, `Impact_3m`, `Impact_10m`, ...
- packed path columns used by the impact-path plots and retention bootstrap

The daily-volume denominator for `Q/V` is controlled by:

- `Q_V_DENOMINATOR_MODE = same_day | prev_day | avg_5d`

The daily volatility normalization is controlled by:

- `DAILY_VOL_MODE = same_day | prev_day | avg_5d`

These daily caches are built from the filtered per-ISIN trade tapes through
helpers in `utils.py`.

## One-dimensional impact fits

The main one-dimensional fit is the power-law relation

`E[I | phi] = Y * phi^gamma`

with `phi = Q/V`.

The current implementation is:

1. filter the per-metaorder table with
   `moimpact.impact_fits.filter_metaorders_info_for_fits`
2. keep only rows that satisfy at least:
   - finite `Q/V`
   - `Q/V >= MIN_QV`
   - finite impact
   - `Participation Rate <= MAX_PARTICIPATION_RATE` when configured
3. log-bin `Q/V` into `N_LOGBIN` bins
4. compute bin-level means and uncertainties
5. fit the binned relation in log space with weighted least squares through
   `fit_power_law_logbins_wls_new`

Related config knobs:

- `MIN_QV`
- `MAX_PARTICIPATION_RATE`
- `N_LOGBIN`
- `MIN_COUNT`
- `RUN_SQL_FITS`
- `RUN_WLS`

The plotting helper `plot_fit` is shared through `moimpact/impact_fits.py`.

## Logarithmic overlay

The code can also compare the power-law fit against a logarithmic overlay of
the form

`E[I | phi] = a * log10(1 + b * phi)`

The overlay is fitted on the same binned statistics through
`fit_logarithmic_from_binned`.

`scripts/plot_prop_nonprop_fits.py` reloads the filtered proprietary and client
tables and exports direct overlays into:

- `images/{DATASET_NAME}/prop_vs_nonprop/`
- `out_files/{DATASET_NAME}/prop_vs_nonprop/`

Its config lives in `config_ymls/plot_prop_nonprop_fits.yml`.

## Bivariate impact surface

`scripts/metaorder_computation.py` also exports a bivariate surface over
relative size and participation rate. The fitted relation is multiplicative in
the same variables used by the one-dimensional workflow, and the script writes:

- a static heatmap
- an interactive 3D Plotly surface

Key config knobs:

- `MIN_COUNT_SURFACE`
- `N_PR_BINS_SURFACE`

## Normalized impact paths

When `COMPUTE_IMPACT_PATHS=true`, the script stores the in-execution and
aftermath paths needed for normalized impact-path plots.

Important controls:

- `AFTERMATH_DURATION_MULTIPLIER`
- `AFTERMATH_NUM_SAMPLES`
- `RUN_IMPACT_PATH_PLOT`
- `SPLIT_BY_SIDE`

These path columns are later reused by:

- `scripts/plot_prop_nonprop_fits.py` for the retention bootstrap
- `moimpact/stats/impact_paths.py` for the underlying cluster bootstrap logic

## Intraday split

`scripts/metaorder_intraday_analysis.py` reuses the saved per-metaorder tables
and splits each metaorder into named intraday sessions using the start and end
timestamps in `Period`.

Current defaults in `config_ymls/metaorder_intraday_analysis.yml` are:

- `morning = [09:30:00, 13:30:00)`
- `evening = [13:30:00, 17:30:00]`

The script writes:

- `out_files/{DATASET_NAME}/{LEVEL}_metaorder_intraday_analysis/`
- `images/{DATASET_NAME}/{LEVEL}_metaorder_intraday_analysis/`

with:

- session summary CSV and Parquet
- counts figure
- per-group session fit comparison figures

## Reproducible usage

Run the main impact pipeline from the repo root:

```bash
source /home/danielemdn/miniconda3/etc/profile.d/conda.sh
conda activate main
python scripts/metaorder_computation.py
```

Common follow-up analyses:

```bash
python scripts/plot_prop_nonprop_fits.py
python scripts/metaorder_intraday_analysis.py
```

To override the default YAML file:

```bash
METAORDER_COMP_CONFIG=config_ymls/metaorder_computation.yml python scripts/metaorder_computation.py
```

## Related docs

- [`metaorder_distributions.md`](metaorder_distributions.md)
- [`metaorder_summary_statistics.md`](metaorder_summary_statistics.md)
- [`bootstrap_methods.md`](bootstrap_methods.md)
- [`PLOTTING_GUIDE.md`](PLOTTING_GUIDE.md)
