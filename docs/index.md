# Metaorders, Price Impact, and Crowding

This repository reconstructs metaorders from CONSOB trade data, measures
volatility-normalized price impact, and studies how proprietary and client
aggressive flow align with signed imbalances. The docs in this folder describe
the current code paths and output conventions rather than hard-coding one
historical run.

## Repository map

The main scripts are:

- `scripts/metaorder_computation.py`
  - canonical preprocessing, metaorder reconstruction, per-metaorder tables,
    one-dimensional impact fits, impact surfaces, and normalized impact paths
  - config: `config_ymls/metaorder_computation.yml`
- `scripts/metaorder_distributions.py`
  - combined client-vs-proprietary distributions figure and optional tail-model
    comparison with bootstrap summaries
  - config: `config_ymls/metaorder_distributions.yml`
- `scripts/metaorder_summary_statistics.py`
  - nationality shares, nationality context tables, member profiles, and mean
    daily metaorder-volume share by ISIN
  - config: `config_ymls/metaorder_summary_statistics.yml`
- `scripts/crowding_analysis.py`
  - within-group, cross-group, all-others, and member-level crowding analyses;
    can also delegate the participation-conditioned analysis
  - config: `config_ymls/crowding_analysis.yml`
- `scripts/crowding_vs_part_rate.py`
  - crowding as a function of participation rate `eta`, with date-cluster
    bootstrap, optional placebo/permutation tests, regressions, and optional
    2D `Q/V x eta` heatmaps
- `scripts/metaorder_start_event_study.py`
  - matched start-intensity event-study around high-participation anchors
  - config: `config_ymls/metaorder_start_event_study.yml`
- `scripts/metaorder_intraday_analysis.py`
  - morning-vs-evening session split on existing metaorder tables
  - config: `config_ymls/metaorder_intraday_analysis.yml`
- `scripts/plot_prop_nonprop_fits.py`
  - proprietary-vs-client fit overlays and optional retention bootstrap
  - config: `config_ymls/plot_prop_nonprop_fits.yml`
- `scripts/member_statistics.py`
  - descriptive member and ISIN plots from per-ISIN trade tapes
- `scripts/metaorder_clustering.py`
  - PCA + k-means clustering on metaorder features with silhouette-based `k`
    selection
- `scripts/generate_paper_figures.py`
  - paper-oriented runner that reads `paper/main.tex`, selects figures, and
    drives the existing scripts with temporary YAML overrides
- `scripts/export_paper_appendix_it.py`
  - appendix table export for the Italian-member subset

Shared modules:

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

1. Run `scripts/metaorder_computation.py` for proprietary and
   non-proprietary flow.
2. Use the resulting dictionaries and filtered per-metaorder tables in:
   - `scripts/metaorder_distributions.py`
   - `scripts/metaorder_summary_statistics.py`
   - `scripts/crowding_analysis.py`
   - `scripts/metaorder_start_event_study.py`
   - `scripts/metaorder_intraday_analysis.py`
   - `scripts/plot_prop_nonprop_fits.py`
   - `scripts/metaorder_clustering.py`
3. Use `scripts/generate_paper_figures.py` when the target is the figure set
   referenced in `paper/main.tex`.

The convenience wrapper `run_all_pipelines.sh` runs the core production path:

- activates the `main` conda environment
- runs `metaorder_computation.py` for proprietary and client flow
- runs `metaorder_distributions.py`
- runs `metaorder_summary_statistics.py`
- runs `crowding_analysis.py`
- runs `member_statistics.py`

## Inputs

The repository expects proprietary trade data that is not shipped with the
codebase. Current defaults are:

- raw CSVs: `data/csv/*.csv`
- per-ISIN trade tapes: `data/parquet/*.parquet`
- member nationality metadata: `data/members_nationality.parquet`

The canonical trade preprocessing path uses `utils.map_trade_codes` followed by
`utils.build_trades_view`. The working schema described by the scripts includes
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
- `paper/images/` when generated through `scripts/generate_paper_figures.py`

Common artifacts written by `metaorder_computation.py`:

- `metaorders_dict_all_{LEVEL}_{group}[ _member_nationality_{it|foreign} ].pkl`
- `metaorders_info_sameday_{...}.parquet`
- `metaorders_info_sameday_filtered_{...}.parquet`

Common figure roots:

- `images/{DATASET_NAME}/{LEVEL}_{group}/`
- `images/{DATASET_NAME}/{LEVEL}_metaorder_distributions/`
- `images/{DATASET_NAME}/{LEVEL}_metaorder_summary_statistics/`
- `images/{DATASET_NAME}/prop_vs_nonprop/`
- `images/{DATASET_NAME}/crowding_vs_part_rate/`
- `images/{DATASET_NAME}/metaorder_start_event_study/`
- `images/{DATASET_NAME}/{LEVEL}_metaorder_intraday_analysis/`

Most Plotly outputs are split into `html/` and `png/` subfolders via the shared
`moimpact.plotting.make_plot_output_dirs` helper.

## Configuration contract

The YAML files under `config_ymls/` are the main source of default settings.
Important patterns shared across scripts:

- `DATASET_NAME` is used in path templates and output tags.
- `OUTPUT_FILE_PATH` and `IMG_OUTPUT_PATH` commonly accept
  `{DATASET_NAME}` placeholders.
- several workflows also support member-nationality suffixing through
  `_member_nationality_{it|foreign}` in stems and filenames.
- many scripts accept an environment variable that points to an alternate YAML
  file, for example `METAORDER_COMP_CONFIG` or `CROWDING_CONFIG`.

The main config files are:

- `config_ymls/metaorder_computation.yml`
- `config_ymls/metaorder_distributions.yml`
- `config_ymls/metaorder_summary_statistics.yml`
- `config_ymls/crowding_analysis.yml`
- `config_ymls/metaorder_start_event_study.yml`
- `config_ymls/metaorder_intraday_analysis.yml`
- `config_ymls/plot_prop_nonprop_fits.yml`
- `config_ymls/paper_figures.yml`

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
python scripts/generate_paper_figures.py --targets all
```

The scripts that create run manifests currently are:

- `scripts/crowding_vs_part_rate.py`
- `scripts/metaorder_start_event_study.py`
- `scripts/generate_paper_figures.py`

Those manifests record timestamps, paths, selected arguments, and when
available the current git hash.

## Method docs

- [`market_impact.md`](market_impact.md)
- [`imbalance_and_crowding.md`](imbalance_and_crowding.md)
- [`bootstrap_methods.md`](bootstrap_methods.md)
- [`metaorder_distributions.md`](metaorder_distributions.md)
- [`metaorder_summary_statistics.md`](metaorder_summary_statistics.md)
- [`metaorder_start_event_study.md`](metaorder_start_event_study.md)
- [`PLOTTING_GUIDE.md`](PLOTTING_GUIDE.md)

## Notes

- The docs intentionally avoid baking in one fixed set of numeric results.
  Use the generated tables and logs under `out_files/` for run-specific values.
- `docs/crowding_review.md` is a paper-writing checklist, not the canonical
  description of the implemented crowding code.
