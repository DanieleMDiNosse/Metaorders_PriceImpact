# Metaorders\_PriceImpact — Agent Guide (root)

## Repository goal (what this repo is for)
This repository is a research toolkit to reconstruct **metaorders** from CONSOB trade prints, estimate **volatility-normalized price impact**, and study **crowding / imbalance** by comparing **proprietary** vs **client (non-proprietary)** aggressive flow. It is designed to generate the *tables and figures* that back a scientific paper (see `paper/main.tex`).

Core questions the code supports:
- **Impact law:** how average impact scales with relative size (power-law “square-root” vs alternatives).
- **Heterogeneity:** differences between proprietary vs client metaorders.
- **Crowding:** whether metaorders align with within-group and cross-group imbalances, and how this varies with participation rate.

---

## Main entrypoints (what to run)
### Unified CLI dispatcher
Use `scripts/run_analysis.py <group> <command>` from the repository root. List the currently registered workflows with:
- `python scripts/run_analysis.py --list`

The dispatcher accepts `--config PATH` as a uniform alias. For legacy workflows it sets the appropriate environment variable before import; for newer workflows it also forwards `--config-path` when needed.

### One-command pipeline (standard analysis refresh)
- `bash run_all_pipelines.sh`
  - Activates conda env `main`.
  - Runs the CSV/parquet intro transform once, then parallel `(PROPRIETARY, MEMBER_NATIONALITY)` metaorder slices for `all`, `it`, and `foreign`.
  - For each metaorder slice it runs the full fit pass and a WLS-only buy/sell decomposition pass.
  - Runs `metaorders distributions` and `metaorders summary` for the nationality slices, then `crowding daily` and `members stats`.
  - Uses temporary YAML copies passed via environment variables; it does **not** edit `config_ymls/*.yml` in place.
  - Controls: `MAX_JOBS` (default 4), `DISABLE_PLOT_LEGENDS`, `IMG_OUTPUT_PATH_OVERRIDE`, `RUN_TAG`, `MIN_FREE_GB`.

### Core scripts (paper backend)
- `python scripts/run_analysis.py paper figures`
  - Regenerates the figures referenced by `paper/main.tex` into `paper/images/...`.
  - Config: `config_ymls/paper_figures.yml` (or `PAPER_FIGURES_CONFIG`).
  - Important knobs:
    - `DEFAULT_TARGETS`, `MAX_WORKERS`, `WRITE_PDF`
    - `IMPACT_FIT_FIGURE_WIDTH` / `IMPACT_FIT_FIGURE_HEIGHT` are forwarded to the relevant generator configs for paper impact-fit figures (currently Figures 6, 7, 8, 17, and 18).
  - It still relies on the generator scripts and their YAMLs; LaTeX should consume the copied snapshot under `paper/images/`.

- `python scripts/run_analysis.py metaorders compute`
  - Inputs: raw trades (`data/csv/*.csv`) and/or derived tapes (`data/parquet/*.parquet`) depending on config.
  - Enriches trade tapes with member nationality when `RUN_INTRO: true` and `data/members_nationality.parquet` is present (adds `Aggressive Member Nationality`).
  - Outputs (under `out_files/{DATASET_NAME}/`): metaorder dictionaries (`*.pkl`) and per-metaorder tables (`*.parquet`).
  - Outputs (under `images/{DATASET_NAME}/{LEVEL}_{proprietary_tag}/`): impact fits, surfaces, distributions, and (optionally) normalized impact paths. `png` and `html` subfolders contain static and interactive versions.
  - Config: `config_ymls/metaorder_computation.yml` (or `METAORDER_COMP_CONFIG`).
  - Key config knobs:
    - `LEVEL: member|client`, `PROPRIETARY: true|false`
    - `MEMBER_NATIONALITY: null|it|foreign` (optional filter applied on the tape column `Aggressive Member Nationality`)
    - `SPLIT_BY_SIDE`, `RUN_WLS`, `RUN_IMPACT_PATH_PLOT`, `IMPACT_FIT_FIGURE_WIDTH`, `IMPACT_FIT_FIGURE_HEIGHT`

- `python scripts/run_analysis.py metaorders distributions`
  - Combined distribution diagnostics + power-law overlays.
  - Inputs: proprietary and client metaorder dictionaries from `out_files/{DATASET_NAME}/...` plus per-ISIN trade tapes from `data/parquet/`.
  - Outputs: combined distributions figure under `images/{DATASET_NAME}/{LEVEL}_metaorder_distributions/{png,html}/`, plus fit-summary tables under `out_files/{DATASET_NAME}/{LEVEL}_metaorder_distributions/`.
  - Config: `config_ymls/metaorder_distributions.yml` (or `METAORDER_DISTRIBUTIONS_CONFIG`).

- `python scripts/run_analysis.py metaorders summary`
  - Member profile, nationality share, and mean daily metaorder-volume share.
  - Inputs: proprietary and client metaorder dictionaries from `out_files/{DATASET_NAME}/...` plus per-ISIN trade tapes from `data/parquet/`.
  - Outputs: combined summary figures under `images/{DATASET_NAME}/{LEVEL}_metaorder_summary_statistics/{png,html}/`.
  - Config: `config_ymls/metaorder_summary_statistics.yml` (or `METAORDER_SUMMARY_STATS_CONFIG`).

- `python scripts/run_analysis.py metaorders intraday-impact`
  - Splits impact fits by intraday session.
  - Config: `config_ymls/metaorder_intraday_analysis.yml` (or `METAORDER_INTRADAY_CONFIG`).

- `python scripts/run_analysis.py metaorders start-event`
  - Start-intensity event study around high-participation anchors.
  - Config: `config_ymls/metaorder_start_event_study.yml` via `--config`/`--config-path`.

- `python scripts/run_analysis.py metaorders start-time`
  - Intraday start-time distribution comparison.
  - Config: `config_ymls/metaorder_start_time_distribution.yml` (or `METAORDER_START_TIME_DISTRIBUTION_CONFIG`, also accepts `--config`).

- `python scripts/run_analysis.py impact overlay`
  - Overlays proprietary vs client impact fits and can run impact-retention bootstrap diagnostics.
  - Inputs: filtered per-metaorder parquets produced by `metaorders compute`.
  - Outputs: `images/{DATASET_NAME}/prop_vs_nonprop/{png,html}/` and summary tables/logs under `out_files/{DATASET_NAME}/prop_vs_nonprop/` or logs.
  - Config: `config_ymls/plot_prop_nonprop_fits.yml` (or `PLOT_PROP_NONPROP_FITS_CONFIG`).
  - Paper Figures 6 and 7 are generated here.

- `python scripts/run_analysis.py crowding daily`
  - Canonical “prop vs client” daily/cross/all-others crowding figures.
  - Inputs: filtered per-metaorder parquets produced by `metaorders compute`.
  - Outputs: `images/{DATASET_NAME}/prop_vs_nonprop/{png,html}/` and `crowding_analysis.log`.
  - Config: `config_ymls/crowding_analysis.yml` (or `CROWDING_CONFIG`).

- `python scripts/run_analysis.py crowding eta`
  - Crowding vs participation rate “η”.
  - Inputs: same metaorder parquets as crowding analysis.
  - Outputs:
    - tables/logs in `out_files/{DATASET_NAME}/{analysis_tag}/` (default `analysis_tag=crowding_vs_part_rate`)
    - figures in `images/{DATASET_NAME}/{analysis_tag}/{png,html}/`
    - `run_manifest.json` in the output folder (use this as a paper traceability template).
  - Config: `config_ymls/crowding_analysis.yml` via `--config`/`--config-path`.

- `python scripts/run_analysis.py crowding impact`
  - Crowding-conditioned impact curves, robustness by participation-rate bins, and cell-level regressions.
  - Inputs: filtered proprietary/client metaorder parquets.
  - Outputs: tables/logs under `out_files/{DATASET_NAME}/crowding_impact/` and figures under `images/{DATASET_NAME}/crowding_impact/{png,html}/`.
  - Config: `config_ymls/crowding_impact_analysis.yml` (or `CROWDING_IMPACT_CONFIG`, also accepts `--config`).
  - Paper Figures 17 and 18 are generated here.

- `python scripts/run_analysis.py crowding overlap`
  - Computes active-overlap crowding features used by later impact/crowding checks.
  - Config: `config_ymls/crowding_overlap_analysis.yml` (or `CROWDING_OVERLAP_ANALYSIS_CONFIG`, also accepts `--config`).

- `python scripts/run_analysis.py crowding member-overlap`
  - Active member-level prop-client alignment based on same-member overlapping client flow.
  - Config: `config_ymls/member_active_overlap_crowding.yml` (or `MEMBER_ACTIVE_OVERLAP_CROWDING_CONFIG`, also accepts `--config`).

- `python scripts/run_analysis.py crowding intraday`
  - Profiles crowding by intraday start bin.
  - Config: `config_ymls/crowding_intraday_profile.yml` (or `CROWDING_INTRADAY_PROFILE_CONFIG`, also accepts `--config`).

- `python scripts/run_analysis.py execution schedule`
  - Compares proprietary/client execution schedules.
  - Outputs: `images/{DATASET_NAME}/member_metaorder_execution_schedule/{png,html}/` and related tables where configured.
  - Config: `config_ymls/metaorder_execution_schedule.yml` (or `METAORDER_EXECUTION_SCHEDULE_CONFIG`).

- `python scripts/run_analysis.py execution typology`
  - Clusters execution typologies.
  - Config: `config_ymls/metaorder_execution_typology.yml` via `--config`/`--config-path`.

- `python scripts/run_analysis.py execution cluster`
  - PCA + k-means clustering on metaorder features; can run on proprietary, client, or both.
  - Outputs in `out_files/{DATASET_NAME}/kmeans_pca_clustering_{level}_{group}/` and `images/{DATASET_NAME}/kmeans_pca_clustering_{level}_{group}/{html,png}/`.

### Utilities / supporting scripts
- `python scripts/run_analysis.py members stats`: member/ISIN descriptive plots; outputs to `images/{DATASET_NAME}/member_statistics/{png,html}/` (reads per-ISIN tapes from `data/parquet/`).
  - If `Aggressive Member Nationality` is available in the tapes, member coverage bars are colored by inferred member nationality (IT vs FOREIGN; UNKNOWN/MIXED when needed).
- `utils.py`: schema mapping (`map_trade_codes`), canonical trade view (`build_trades_view`), realized volatility estimators, metaorder detection helpers.
- `moimpact/`: shared library used across scripts (YAML config loading, path templating, logging helpers, plot styling, figure export helpers, and correlation/bootstrap utilities).
- `docs/`: method notes aligned with the implementation (start at `docs/index.md`).

---

## Inputs and outputs (canonical locations)
### Data roots
These are config-driven (YAML), but the current defaults in this working tree are:
- Raw per-ISIN CSVs: `data/csv/*.csv`
- Derived per-ISIN tapes (parquet): `data/parquet/*.parquet`
- Member metadata used for nationality tags: `data/members_nationality.parquet` (columns: `FIRM_ID_MODIF`, `NAZIONALITA`)

### Generated artifacts (do not hand-edit)
- Tables / serialized objects: `out_files/{DATASET_NAME}/`
  - examples:
    - `metaorders_dict_all_{LEVEL}_{proprietary_tag}*.pkl`
    - `metaorders_info_sameday_{...}.parquet`
    - `metaorders_info_sameday_filtered_{...}.parquet`
    - analysis folders like `out_files/{DATASET_NAME}/crowding_vs_part_rate/`, `crowding_impact/`, `member_active_overlap_crowding/`
- Generated figures: `images/{DATASET_NAME}/...`
  - PNGs typically under `.../png/`, HTML under `.../html/`
  - Some Plotly workflows also emit PDF sidecars when requested (for example by `PLOTLY_WRITE_PDF=true` or paper-runner export settings).
- Paper-stable figure snapshot: `paper/images/...`
  - LaTeX should reference this snapshot rather than transient `images/{DATASET_NAME}/...` outputs.
- Logs:
  - sibling to scripts or output folders depending on the workflow
  - pipeline logs under `out_files/{DATASET_NAME}/logs/pipeline_{RUN_TAG}/`
  - analysis logs such as `out_files/{DATASET_NAME}/crowding_vs_part_rate/run_YYYYMMDD_HHMMSS.log`

---

## Configuration contract (what changes results)
Treat YAML files in `config_ymls/` as the single source of truth for defaults:
- `config_ymls/metaorder_computation.yml`
  - identification/filtering knobs: `LEVEL`, `PROPRIETARY`, `MEMBER_NATIONALITY`, `MIN_TRADES`, `SECONDS_FILTER`, `MAX_GAP`, `TRADING_HOURS`
  - normalization knobs: `Q_V_DENOMINATOR_MODE ∈ {same_day, prev_day, avg_5d}`, `DAILY_VOL_MODE ∈ {same_day, prev_day, avg_5d}`
  - estimation knobs: `N_LOGBIN`, `MIN_COUNT`, `MIN_QV`, `RESAMPLE_FREQ`, `MAX_PARTICIPATION_RATE`
  - impact-figure canvas knobs: `IMPACT_FIT_FIGURE_WIDTH`, `IMPACT_FIT_FIGURE_HEIGHT`
- `config_ymls/plot_prop_nonprop_fits.yml`
  - proprietary/client overlay inputs, fit filters, retention-bootstrap settings, and impact-overlay canvas knobs.
- `config_ymls/crowding_analysis.yml`
  - daily crowding, crowding-vs-η, bootstrap/permutation, and plotting controls.
- `config_ymls/crowding_impact_analysis.yml`
  - crowding-conditioned impact curves, joint/cell regression settings, bootstrap settings, and impact-figure canvas knobs.
- `config_ymls/crowding_overlap_analysis.yml` and `config_ymls/member_active_overlap_crowding.yml`
  - active-overlap feature and member-level prop-client alignment settings.
- `config_ymls/metaorder_distributions.yml`, `config_ymls/metaorder_summary_statistics.yml`, `config_ymls/metaorder_execution_schedule.yml`, and related workflow YAMLs
  - workflow-specific filters, bootstrap settings, output toggles, and plot controls.
- `config_ymls/plot_style.yml`
  - central generated-figure visual style: Plotly/Matplotlib fonts, tick/label/legend sizes, colorway, grid/background colors, default export size/scale, and DPI.
- `config_ymls/paper_figures.yml`
  - paper figure runner settings and paper-only forwarding of impact-fit canvas dimensions.

If a result is used in the paper, its generating YAML + CLI arguments must be archived alongside outputs or recorded in a manifest (see “paper-ready workflow” below).

---

## Figure style and paper snapshot rules
- Make visual changes in the generating script/config when the user asks to change generated figures. Do not use LaTeX `\includegraphics` resizing as a substitute unless the request is explicitly LaTeX-only.
- General generated-figure typography and color defaults come from `config_ymls/plot_style.yml` through `moimpact.plot_style`.
- Impact-fit canvas dimensions for paper Figures 6, 7, 8, 17, and 18 are controlled from `config_ymls/paper_figures.yml` and forwarded to:
  - `config_ymls/plot_prop_nonprop_fits.yml`
  - `config_ymls/metaorder_computation.yml`
  - `config_ymls/crowding_impact_analysis.yml`
- Some plot-specific colors/marker sizes/line widths are still hard-coded in the plotting functions; inspect the generator before promising that a style key is global.
- `paper/images/` is the LaTeX source of truth. When regenerating figures, copy only final publication-ready artifacts from `images/{DATASET_NAME}/...` into `paper/images/...`.

---

## Scientific paper backend: defensible workflow (paper-ready)
The goal is **traceability**: every figure/table in `paper/` must be reproducible from code + data + config.

### Required for any paper run
1. **Freeze configuration**
   - Copy the exact YAML(s) used for the run into the run output folder or store an equivalent structured manifest.
2. **Record provenance**
   - Record: git commit hash, command line, timestamp, dataset name, input paths, output paths, and key config values.
   - Prefer the manifest patterns already implemented in workflows such as `crowding eta`, `crowding impact`, and `paper figures`.
3. **Determinism**
   - Any randomness (bootstrap/permutation, k-means, t-SNE, sampling) must run with an explicit, recorded seed.
4. **No manual numbers**
   - Do not type statistics into `paper/main.tex` by hand unless they are copied from a generated, saved table and the source table/run is traceable.
5. **Robustness checks (minimum set)**
   - Metaorder definition sensitivity: vary `MIN_TRADES`, `MAX_GAP`, `SECONDS_FILTER`.
   - Normalization sensitivity: compare `same_day` vs `prev_day` vs `avg_5d` for both volume and volatility modes.
   - Subsample stability: check by time subperiod and by instrument subsets (large vs small names).
   - Dependence-aware inference: use day-cluster logic or another justified clustering unit for uncertainty around correlations/means.
6. **Build verification**
   - After editing `paper/main.tex`, compile with `latexmk -cd -pdf paper/main.tex` from the repo root.
   - Known current manuscript warnings include duplicate labels around the early member-statistics figures and an old float-size warning near the distributions figure; do not confuse those with new errors unless your edit changes them.

### Output organization for paper runs (recommended convention)
When producing final paper artifacts, write into a dedicated run folder when supported:
- `out_files/{DATASET_NAME}/runs/{YYYYMMDD_HHMMSS}_{tag}/...`
- `images/{DATASET_NAME}/runs/{YYYYMMDD_HHMMSS}_{tag}/...`
and keep a manifest in that folder. If a script does not support run folders, mimic the provenance manifest manually.

---

## Environment / execution rules
- Activate the main conda environment before running full data pipelines:
  - `conda activate main`
- Keep raw CONSOB inputs out of git; treat raw data and derived sensitive data as proprietary.
- Prefer running scripts from the repo root so relative paths resolve consistently.
- Use temporary configs or `--config`/environment variables for experiments; avoid hand-editing default YAMLs for one-off runs unless the default itself is intentionally changing.

---

## Documentation pointers (methods truth)
- Start here: `docs/index.md`
- Impact fits and filters: `docs/POWER_LAW_IMPACT_FITS.md`
- Crowding definitions and inference: `docs/prop_vs_nonprop.md`
- Metaorder distributions: `docs/metaorder_distributions.md`
- Metaorder summary statistics: `docs/metaorder_summary_statistics.md`
- Paper artifacts and figure syncing: see `paper/AGENTS.md`
