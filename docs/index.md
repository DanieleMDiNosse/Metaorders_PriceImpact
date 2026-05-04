# Metaorders, Price Impact, and Crowding

This repository reconstructs metaorders from CONSOB trade data, measures
volatility-normalized price impact, and studies how proprietary and client
aggressive flow align with signed imbalances. The docs in this folder describe
the current code paths and output conventions rather than hard-coding one
historical run.

## Repository Map

The stable command surface is the unified CLI `scripts/run_analysis.py`.
Implementation code lives under `moimpact/workflows/`, while shared numerical,
plotting, config, and inference helpers live under `moimpact/`.

Main CLI commands:

- `scripts/run_analysis.py metaorders compute`
  - canonical preprocessing, metaorder reconstruction, per-metaorder tables,
    one-dimensional impact fits, impact surfaces, and normalized impact paths
  - config: `config_ymls/metaorder_computation.yml`
- `scripts/run_analysis.py metaorders distributions`
  - combined client-vs-proprietary distributions figure and optional tail-model
    comparison with bootstrap summaries
  - config: `config_ymls/metaorder_distributions.yml`
- `scripts/run_analysis.py metaorders summary`
  - nationality shares, nationality context tables, member profiles, and mean
    daily metaorder-volume share by ISIN
  - config: `config_ymls/metaorder_summary_statistics.yml`
- `scripts/run_analysis.py crowding daily`
  - within-group, cross-group, all-others, and member-level crowding analyses;
    can also delegate the participation-conditioned analysis
  - config: `config_ymls/crowding_analysis.yml`
- `scripts/run_analysis.py crowding eta`
  - crowding as a function of participation rate `eta`, with date-cluster
  bootstrap, optional placebo/permutation tests, regressions, and optional
  2D `Q/V x eta` heatmaps
  - config: `config_ymls/crowding_analysis.yml`
- `scripts/run_analysis.py crowding impact`
  - crowding-conditioned impact curves, benchmark-impact contrasts, and
    participation-bin robustness for client vs proprietary metaorders
  - config: `config_ymls/crowding_impact_analysis.yml`
- `scripts/run_analysis.py crowding intraday`
  - intraday start-bin profile of aligned all-vs-all crowding
  - config: `config_ymls/crowding_intraday_profile.yml`
- `scripts/run_analysis.py crowding overlap`
  - interval-level active-overlap features and overlap-conditioned impact
    regressions
  - config: `config_ymls/crowding_overlap_analysis.yml`
- `scripts/run_analysis.py crowding member-overlap`
  - member-level proprietary/client active-overlap correlations
  - config: `config_ymls/member_active_overlap_crowding.yml`
- `scripts/run_analysis.py metaorders start-event`
  - matched start-intensity event-study around high-participation anchors
  - config: `config_ymls/metaorder_start_event_study.yml`
- `scripts/run_analysis.py metaorders start-time`
  - intraday start-time distributions for proprietary and client metaorders
  - config: `config_ymls/metaorder_start_time_distribution.yml`
- `scripts/run_analysis.py execution schedule`
  - proprietary-vs-client execution-schedule curves, heatmaps, and scalar
    inference
  - config: `config_ymls/metaorder_execution_schedule.yml`
- `scripts/run_analysis.py execution typology`
  - pooled execution typology with behavior-led clustering, type shares,
    within-type impact profiles, and within-type execution schedule profiles
  - config: `config_ymls/metaorder_execution_typology.yml`
- `scripts/run_analysis.py metaorders intraday-impact`
  - morning-vs-evening session split on existing metaorder tables
  - config: `config_ymls/metaorder_intraday_analysis.yml`
- `scripts/run_analysis.py impact overlay`
  - proprietary-vs-client fit overlays and optional retention bootstrap
  - config: `config_ymls/plot_prop_nonprop_fits.yml`
- `scripts/run_analysis.py members stats`
  - descriptive member and ISIN plots from per-ISIN trade tapes
- `scripts/run_analysis.py execution cluster`
  - PCA + k-means clustering on metaorder features with silhouette-based `k`
    selection
  - config: `config_ymls/metaorder_computation.yml`
- `scripts/run_analysis.py paper figures`
  - paper-oriented runner that reads `paper/main.tex`, selects figures, and
    drives the workflow commands with temporary YAML overrides

Shared modules:

- `moimpact/workflows/`: workflow modules behind the unified CLI
- `utils.py`: trade-schema mapping, canonical trade view, realized-volatility
  helpers, metaorder detection helpers
- `moimpact/config.py`: YAML loading, path templating, repo-relative resolution
- `moimpact/impact_fits.py`: shared impact filtering and fitting helpers
- `moimpact/metaorder_distribution_samples.py`: canonical sample extraction for
  the distributions and summary-statistics workflows
- `moimpact/stats/correlation.py`: correlation, cluster bootstrap, and
  permutation helpers
- `moimpact/stats/impact_paths.py`: retention bootstrap for impact-path
  comparisons

## End-to-end workflow

The typical member-level workflow is:

1. Run `scripts/run_analysis.py metaorders compute` for proprietary and
   non-proprietary flow.
2. Use the resulting dictionaries and filtered per-metaorder tables in:
   - `scripts/run_analysis.py metaorders distributions`
   - `scripts/run_analysis.py metaorders summary`
   - `scripts/run_analysis.py crowding daily`
   - `scripts/run_analysis.py metaorders start-event`
   - `scripts/run_analysis.py metaorders intraday-impact`
   - `scripts/run_analysis.py execution typology`
   - `scripts/run_analysis.py impact overlay`
   - `scripts/run_analysis.py execution cluster`
3. Use `scripts/run_analysis.py paper figures` when the target is the figure set
   referenced in `paper/main.tex`.

The convenience wrapper `run_all_pipelines.sh` runs the core production path:

- activates the `main` conda environment
- runs `metaorders compute` for proprietary and client flow
- runs `metaorders distributions`
- runs `metaorders summary`
- runs `crowding daily`
- runs `members stats`

## Inputs

The repository expects proprietary trade data that is not shipped with the
codebase. Current defaults are:

- raw CSVs: `data/csv/*.csv`
- per-ISIN trade tapes: `data/parquet/*.parquet`
- member nationality metadata: `data/members_nationality.parquet`

The canonical trade preprocessing path uses `utils.map_trade_codes` followed by
`utils.build_trades_view`. The working schema described by the workflows includes
fields such as:

- `MIC`
- `TRADING_DAY`
- `TRADETIME`
- `TRADED_QUANTITY`
- `TRADED_PRICE`
- `COD_BUY`, `COD_SELL`
- `CLIENT_IDENTIFIC_SHORT_CODE_BUY`, `CLIENT_IDENTIFIC_SHORT_CODE_SELL`
- `PASSIVE_ORDER_INDICATOR_BUY`
- `TRADING_CAPACITY_BUY`, `TRADING_CAPACITY_SELL`

All main analyses default to the continuous session
`09:30:00` to `17:30:00`.

## Core output layout

Tables and serialized objects live under:

- `out_files/{DATASET_NAME}/`

Figures live under:

- `images/{DATASET_NAME}/`
- `paper/images/` when generated through `scripts/run_analysis.py paper figures`

Common artifacts written by `scripts/run_analysis.py metaorders compute`:

- `metaorders_dict_all_{LEVEL}_{group}[ _member_nationality_{it|foreign} ].pkl`
- `metaorders_info_sameday_{...}.parquet`
- `metaorders_info_sameday_filtered_{...}.parquet`

Common figure roots:

- `images/{DATASET_NAME}/{LEVEL}_{group}/`
- `images/{DATASET_NAME}/{LEVEL}_metaorder_distributions/`
- `images/{DATASET_NAME}/{LEVEL}_metaorder_summary_statistics/`
- `images/{DATASET_NAME}/prop_vs_nonprop/`
- `images/{DATASET_NAME}/crowding_vs_part_rate/`
- `images/{DATASET_NAME}/crowding_impact/`
- `images/{DATASET_NAME}/crowding_intraday_profile/`
- `images/{DATASET_NAME}/crowding_overlap_analysis/`
- `images/{DATASET_NAME}/member_active_overlap_crowding/`
- `images/{DATASET_NAME}/metaorder_start_event_study/`
- `images/{DATASET_NAME}/metaorder_start_time_distribution/`
- `images/{DATASET_NAME}/member_metaorder_execution_schedule/`
- `images/{DATASET_NAME}/{LEVEL}_metaorder_intraday_analysis/`

Most Plotly outputs are split into `html/` and `png/` subfolders via the shared
`moimpact.plotting.make_plot_output_dirs` helper.

## Configuration contract

The YAML files under `config_ymls/` are the main source of default settings.
Important patterns shared across workflows:

- `DATASET_NAME` is used in path templates and output tags.
- `OUTPUT_FILE_PATH` and `IMG_OUTPUT_PATH` commonly accept
  `{DATASET_NAME}` placeholders.
- several workflows also support member-nationality suffixing through
  `_member_nationality_{it|foreign}` in stems and filenames.
- commands accept a uniform `--config PATH` alias when the workflow exposes a
  YAML path, and many workflows also accept an environment variable such as
  `METAORDER_COMP_CONFIG` or `CROWDING_CONFIG`.

The main config files are:

- `config_ymls/metaorder_computation.yml`
- `config_ymls/metaorder_distributions.yml`
- `config_ymls/metaorder_summary_statistics.yml`
- `config_ymls/crowding_analysis.yml`
- `config_ymls/crowding_impact_analysis.yml`
- `config_ymls/crowding_intraday_profile.yml`
- `config_ymls/crowding_overlap_analysis.yml`
- `config_ymls/member_active_overlap_crowding.yml`
- `config_ymls/metaorder_start_event_study.yml`
- `config_ymls/metaorder_start_time_distribution.yml`
- `config_ymls/metaorder_execution_schedule.yml`
- `config_ymls/metaorder_execution_typology.yml`
- `config_ymls/metaorder_intraday_analysis.yml`
- `config_ymls/plot_prop_nonprop_fits.yml`
- `config_ymls/paper_figures.yml`
- `config_ymls/plot_style.yml`

## Reproducible execution

From the repo root:

```bash
source /home/danielemdn/miniconda3/etc/profile.d/conda.sh
conda activate main
```

Core pipeline:

```bash
bash run_all_pipelines.sh
```

Paper-oriented figure generation:

```bash
python scripts/run_analysis.py paper figures --targets all
```

Workflows that create run manifests include:

- `scripts/run_analysis.py crowding eta`
- `scripts/run_analysis.py crowding impact`
- `scripts/run_analysis.py crowding intraday`
- `scripts/run_analysis.py crowding overlap`
- `scripts/run_analysis.py crowding member-overlap`
- `scripts/run_analysis.py execution schedule`
- `scripts/run_analysis.py execution typology`
- `scripts/run_analysis.py metaorders start-event`
- `scripts/run_analysis.py metaorders start-time`
- `scripts/run_analysis.py paper figures`

Those manifests record timestamps, paths, selected arguments, and when
available the current git hash.

## Method docs

- [`market_impact.md`](market_impact.md)
- [`imbalance_and_crowding.md`](imbalance_and_crowding.md)
- [`crowding_impact_analysis.md`](crowding_impact_analysis.md)
- [`crowding_impact_results_summary.md`](crowding_impact_results_summary.md)
- [`member_active_overlap_crowding.md`](member_active_overlap_crowding.md)
- [`bootstrap_methods.md`](bootstrap_methods.md)
- [`metaorder_distributions.md`](metaorder_distributions.md)
- [`metaorder_summary_statistics.md`](metaorder_summary_statistics.md)
- [`metaorder_start_event_study.md`](metaorder_start_event_study.md)
- [`metaorder_start_event_study_threshold_sweep_perm35.md`](metaorder_start_event_study_threshold_sweep_perm35.md)
- [`execution_typology.md`](execution_typology.md)
- [`PLOTTING_GUIDE.md`](PLOTTING_GUIDE.md)

## Notes

- The docs intentionally avoid baking in one fixed set of numeric results.
  Use the generated tables and logs under `out_files/` for run-specific values.
- `docs/crowding_review.md` is a paper-writing checklist, not the canonical
  description of the implemented crowding code.
