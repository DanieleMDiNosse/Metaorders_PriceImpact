# Metaorders_PriceImpact

Code for studying how detected metaorders move prices in CONSOB trade data. The repository focuses on identifying same-sign, same-agent runs of trades, summarising their characteristics, and measuring price impact and crowding effects.

- `metaorders_pipeline.py` turns raw CSV trades into parquet caches, extracts metaorders at client/member level for different perspectives (`client`, `broker`, `proprietary`), computes per-metaorder summaries (size, duration, impact, Q/V, participation), and produces power-law impact plots. It exposes a CLI (`python metaorders_pipeline.py mode [--data-dir ... --cache-dir ... --output-dir ...]`) to rerun the pipeline with custom locations and settings.
- `metaorder_computation.py` offers a more configurable, research-oriented workflow to build metaorders, aggregate distributions (durations, inter-arrivals, volumes, Q/V, participation), compute normalized execution/aftermath impact paths, and fit impact curves with weighted least squares for both power-law and logarithmic specifications.
- `metaorder_statistics.py` (formerly `prop_vs_nonprop.py`) compares proprietary vs client flow: it constructs local and environmental imbalances, measures crowding correlations over time (within-group, cross-group, and all-vs-all), analyses daily metaorder count imbalances, and produces additional diagnostics such as imbalance distributions, participation vs |imbalance|, autocorrelation of metaorder signs, and imbalance vs daily returns.
- Supporting notes (`metaorder_distributions.md`, `POWER_LAW_IMPACT_FITS.md`, `prop_vs_nonprop.md`) document the methodology and the impact/crowding analyses aligned with these scripts.

Outputs such as cached metaorders, summary parquet files, and figures are written by the scripts to the configured `out_files/` and `images/` locations.
