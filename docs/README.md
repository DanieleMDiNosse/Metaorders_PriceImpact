# Metaorders_PriceImpact

Toolkit for detecting metaorders in CONSOB trade data, measuring price impact, and studying crowding between proprietary and client flow.

## Main scripts
- `scripts/metaorder_computation.py`:
  - builds metaorders and per-metaorder tables in `out_files/{DATASET_NAME}/`
  - writes impact/fits/surfaces/paths in `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/` and `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/html/`
- `scripts/metaorder_statistics.py`:
  - computes metaorder-dictionary distribution diagnostics (durations, inter-arrivals, volumes, Q/V, participation, nationality share)
  - writes figures in `images/{DATASET_NAME}/{METAORDER_STATS_LEVEL}_{METAORDER_STATS_PROPRIETARY_TAG}/png/` and `.../html/`
- `scripts/crowding_analysis.py`:
  - runs proprietary vs client crowding analyses (within-group, cross-group, all-vs-all, member-level, diagnostics)
  - writes figures in `images/{DATASET_NAME}/prop_vs_nonprop/png/` and `.../html/`
- `scripts/plot_prop_nonprop_fits.py`:
  - overlays proprietary vs client impact fits from filtered parquet outputs
  - if `--out images/{DATASET_NAME}/prop_vs_nonprop/power_law_prop_vs_nonprop.png`, output files are written to `.../prop_vs_nonprop/png/` and `.../prop_vs_nonprop/html/`
- `scripts/member_statistics.py`:
  - computes member/ISIN descriptive plots in `images/ftsemib/member_statistics/png/` and `.../html/`
- `run_all_pipelines.sh`:
  - activates conda env `main`, runs computation/statistics for both groups, then crowding and member stats

## Data and outputs
- Inputs are config-driven. Current defaults are:
  - raw CSV: `data/csv/*.csv`
  - trade tapes: `data/parquet/*.parquet`
- Tables/serialized outputs: `out_files/{DATASET_NAME}/...`
- Figures: `images/{DATASET_NAME}/...` with canonical `png/` and `html/` subfolders.
- Logs are written under `out_files/{DATASET_NAME}/logs/` for pipeline scripts, plus script-level logs where applicable.

## Running
Typical usage from repo root:
- `conda activate main`
- `python scripts/metaorder_computation.py` (run once with `PROPRIETARY=true`, once with `PROPRIETARY=false`)
- `python scripts/metaorder_statistics.py` (run once with `METAORDER_STATS_PROPRIETARY=true`, once with `METAORDER_STATS_PROPRIETARY=false`)
- `python scripts/crowding_analysis.py`
- `python scripts/member_statistics.py`
- optional: `python scripts/plot_prop_nonprop_fits.py --out images/{DATASET_NAME}/prop_vs_nonprop/power_law_prop_vs_nonprop.png`
- one-shot: `bash run_all_pipelines.sh`
