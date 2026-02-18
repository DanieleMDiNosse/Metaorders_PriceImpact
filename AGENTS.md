# Metaorders\_PriceImpact — Agent Guide (root)

## Repository goal (what this repo is for)
This repository is a research toolkit to reconstruct **metaorders** from CONSOB trade prints, estimate **volatility‑normalized price impact**, and study **crowding / imbalance** by comparing **proprietary** vs **client (non‑proprietary)** aggressive flow. It is designed to generate the *tables and figures* that back a scientific paper (see `paper/main.tex`).

Core questions the code supports:
- **Impact law:** how average impact scales with relative size (power‑law “square‑root” vs alternatives).
- **Heterogeneity:** differences between proprietary vs client metaorders.
- **Crowding:** whether metaorders align with within‑group and cross‑group imbalances, and how this varies with participation rate.

---

## Main entrypoints (what to run)
### One-command pipeline (recommended)
- `bash run_all_pipelines.sh`
  - Activates conda env `main`.
  - Runs `metaorder_computation.py` twice (`PROPRIETARY=true/false`) then `metaorder_statistics.py` twice, then `member_statistics.py`.
  - Temporarily edits YAML configs under `config_ymls/` and restores them on exit.

### Core scripts (paper backend)
- `metaorder_computation.py`
  - Inputs: raw trades (`data/csv/*.csv`) and/or derived tapes (`data/parquet/*.parquet`) depending on config.
  - Outputs (under `out_files/{DATASET_NAME}/`): metaorder dictionaries (`*.pkl`) and per‑metaorder tables (`*.parquet`).
  - Outputs (under `images/{DATASET_NAME}/{LEVEL}_{proprietary_tag}/`): impact fits, surfaces, distributions, and (optionally) normalized impact paths. `png` subfolder for the png image version, `html` for the html version.
  - Config: `config_ymls/metaorder_computation.yml`.
  - Optional CLI filters (affect output filenames via suffixes):
    - `--nationality {IT,ST}`
    - `--client-type {PG,PF}`

- `metaorder_statistics.py` (metaorder distributions + auxiliary diagnostics)
  - Inputs: per‑metaorder parquet(s) from `out_files/{DATASET_NAME}/...` and (optionally) trade tapes from `data/parquet/`.
  - Outputs: distribution figures and diagnostics under `images/{DATASET_NAME}/{METAORDER_STATS_LEVEL}_{proprietary_tag}/` plus `metaorder_statistics.log`.
  - Config: `config_ymls/metaorder_statistics.yml`.

- `crowding_analysis.py` (canonical “prop vs client” crowding figures)
  - Inputs: filtered per‑metaorder parquets produced by `metaorder_computation.py`.
  - Outputs: `images/{DATASET_NAME}/prop_vs_nonprop/` and `crowding_analysis.log`.
  - Config: `config_ymls/crowding_analysis.yml`.

- `crowding_vs_part_rate.py` (crowding vs participation rate “η”)
  - Inputs: same metaorder parquets as crowding analysis.
  - Outputs:
    - tables/logs in `out_files/{DATASET_NAME}/{analysis_tag}/` (default `analysis_tag=crowding_vs_part_rate`)
    - figures in `images/{DATASET_NAME}/{analysis_tag}/`
    - writes `run_manifest.json` in the output folder (use this as the paper traceability template).

- `metaorder_clustering.py`
  - PCA + k‑means clustering on metaorder features; can run on proprietary, client, or both.
  - Outputs in `out_files/{DATASET_NAME}/kmeans_pca_clustering_{level}_{group}/` and `images/{DATASET_NAME}/kmeans_pca_clustering_{level}_{group}/`.

### Utilities / supporting scripts
- `member_statistics.py`: member/ISIN descriptive plots; outputs to `images/{DATASET_NAME}/member_statistics/`.
- `plot_prop_nonprop_fits.py`: overlays proprietary vs client impact fits from filtered parquets.
- `utils.py`: schema mapping (`map_trade_codes`), canonical trade view (`build_trades_view`), realized volatility estimators, metaorder detection helpers.
- `docs/`: method notes aligned with the implementation (start at `docs/index.md`).

---

## Inputs and outputs (canonical locations)
### Data roots
These are *config-driven* (YAML), but the current defaults in this working tree are:
- Raw per‑ISIN CSVs: `data/csv/*.csv`
- Derived per‑ISIN “tapes” (parquet): `data/parquet/*.parquet`

### Generated artifacts (do not hand-edit)
- Tables / serialized objects: `out_files/{DATASET_NAME}/`
  - examples:
    - `metaorders_dict_all_{LEVEL}_{proprietary_tag}*.pkl`
    - `metaorders_info_sameday_{...}.parquet`
    - `metaorders_info_sameday_filtered_{...}.parquet`
    - analysis folders like `out_files/{DATASET_NAME}/crowding_vs_part_rate/`
- Figures: `images/{DATASET_NAME}/...`
  - PNGs typically under `.../png/`, HTML under `.../html/`
- Logs:
  - sibling to scripts (e.g., `metaorder_computation.log`, `crowding_analysis.log`)
  - or inside analysis folders (e.g., `out_files/{DATASET_NAME}/crowding_vs_part_rate/run_YYYYMMDD_HHMMSS.log`)

---

## Configuration contract (what changes results)
Treat YAML files in `config_ymls/` as the single source of truth for defaults:
- `config_ymls/metaorder_computation.yml`
  - identification/filtering knobs: `LEVEL`, `PROPRIETARY`, `MIN_TRADES`, `SECONDS_FILTER`, `MAX_GAP`, `TRADING_HOURS`
  - normalization knobs: `Q_V_DENOMINATOR_MODE ∈ {same_day, prev_day, avg_5d}`, `DAILY_VOL_MODE ∈ {same_day, prev_day, avg_5d}`
  - estimation knobs: `N_LOGBIN`, `MIN_COUNT`, `MIN_QV`, `RESAMPLE_FREQ`
- `config_ymls/crowding_analysis.yml` and `config_ymls/metaorder_statistics.yml`
  - inference knobs: `BOOTSTRAP_RUNS`, `ALPHA`, `MIN_N`, `SMOOTHING_DAYS`
  - plotting toggles and (if used) permutation/heatmap controls

If a result is used in the paper, its generating YAML + CLI arguments must be archived alongside outputs (see “paper‑ready workflow” below).

---

## Scientific paper backend: defensible workflow (paper-ready)
The goal is **traceability**: every figure/table in `paper/` must be reproducible from code + data + config.

### Required for any “paper” run
1. **Freeze configuration**
   - Copy the exact YAML(s) used for the run into the run output folder (or store an equivalent structured manifest).
2. **Record provenance**
   - Record: git commit hash, command line, timestamp, dataset name, and key config values.
   - Prefer the pattern already implemented in `crowding_vs_part_rate.py` (`run_manifest.json`).
3. **Determinism**
   - Any randomness (bootstrap/permutation, k‑means, t‑SNE) must run with an explicit, recorded seed.
4. **No manual numbers**
   - Do not type statistics into `paper/main.tex` by hand; generate and save tables under `out_files/` and reference them (or copy into paper with a traceable step).
5. **Robustness checks (minimum set)**
   - Metaorder definition sensitivity: vary `MIN_TRADES`, `MAX_GAP`, `SECONDS_FILTER`.
   - Normalization sensitivity: compare `same_day` vs `prev_day` vs `avg_5d` for both volume and volatility modes.
   - Subsample stability: check by time subperiod and by instrument subsets (large vs small names).
   - Dependence-aware inference: use day‑cluster logic (or another justified clustering unit) for uncertainty around correlations/means.

### Output organization for paper runs (recommended convention)
When producing “final” paper artifacts, write into a dedicated run folder:
- `out_files/{DATASET_NAME}/runs/{YYYYMMDD_HHMMSS}_{tag}/...`
- `images/{DATASET_NAME}/runs/{YYYYMMDD_HHMMSS}_{tag}/...`
and keep a manifest in that folder. (Some scripts already do this partially; if not, mimic it manually.)

---

## Environment / execution rules
- Activate the main conda environment before running Python:
  - `conda activate main`
- Keep raw data out of git; treat CONSOB inputs as proprietary.
- Prefer running scripts from the repo root so relative paths resolve consistently.

---

## Documentation pointers (methods truth)
- Start here: `docs/index.md`
- Impact fits and filters: `docs/POWER_LAW_IMPACT_FITS.md`
- Crowding definitions and inference: `docs/prop_vs_nonprop.md`
- Metaorder distributions: `docs/metaorder_distributions.md`
- Paper artifacts and figure syncing: see `paper/AGENTS.md`
