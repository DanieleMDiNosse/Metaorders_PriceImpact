# Metaorders_PriceImpact

Toolkit for detecting metaorders in CONSOB trade data, measuring their price impact, and studying crowding between proprietary and client flow.

## Main scripts
- `metaorder_computation.py` - converts raw CSVs to parquet, detects metaorders (same-sign runs per member/client within a day with gap/duration filters), builds metaorder dictionaries, and writes per-metaorder summaries to `out_files/`. Config flags cover `LEVEL`, `PROPRIETARY`, dataset filter (MOT vs non-MOT), `Q_V_DENOMINATOR_MODE` and `DAILY_VOL_MODE` (same_day/prev_day/avg_5d), optional impact trajectories (`COMPUTE_IMPACT_PATHS`), and optional signature plots and impact fits (power-law plus logarithmic overlay).
- `metaorder_statistics.py` - consumes the filtered metaorder parquet files for proprietary vs non-proprietary flow, attaches imbalance columns (within-group, cross-group, all-others), optional daily returns, and generates crowding plots and diagnostics; it can also recompute raw metaorder distribution plots from `metaorders_dict_all_*.pkl` when `RUN_METAORDER_DICT_STATS` is true.
- `plot_prop_nonprop_fits.py` - reloads the filtered parquet outputs, reruns the same log-binned WLS fits, and overlays proprietary vs non-proprietary power-law curves on a single figure.
- `member_statistics.py` - summarises members per ISIN, proprietary vs client trade counts, ISIN coverage, and member activity heatmaps (HTML + PNG).
- `metaorders_pipeline.sh` - convenience wrapper that activates the `defi` conda environment and runs `metaorder_computation.py` followed by `metaorder_statistics.py`.

## Data and outputs
- Input CSV/parquet files are expected under `data/` by default; outputs are written to `out_files/` and `images/<level>_<proprietary_tag>/` plus `images/prop_vs_nonprop/` for crowding plots.
- Each script logs to a sibling `.log` file (e.g., `metaorder_computation.log`, `metaorder_statistics.log`, `member_statistics.log`).

## Running
Typical usage from the repo root:
- `python metaorder_computation.py` to rebuild metaorders and impact fits using the in-file flags.
- `python metaorder_statistics.py` to attach imbalance metrics and crowding plots from the saved parquet outputs.
- `python member_statistics.py` for member/ISIN activity summaries.
- `python plot_prop_nonprop_fits.py` to overlay proprietary vs non-proprietary power-law fits.
- `bash metaorders_pipeline.sh` to run computation and statistics back to back.
