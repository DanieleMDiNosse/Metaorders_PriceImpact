# Metaorders_PriceImpact

Code for studying how detected metaorders move prices in CONSOB trade data. The repository focuses on identifying same-sign, same-agent runs of trades, summarising their characteristics, and measuring price impact and crowding effects.

- `metaorders_pipeline.py` turns raw CSV trades into parquet caches, extracts metaorders at client/member level, computes per-metaorder summaries (size, duration, impact, Q/V, participation), and produces power-law impact plots.
- `metaorder_computation.py` offers a more configurable workflow to build metaorders, aggregate distributions (durations, inter-arrivals, volumes, Q/V, participation), and fit impact curves with weighted least squares.
- `prop_vs_nonprop.py` compares proprietary vs client flow: it builds local/environmental imbalances, measures crowding correlations over time, and can plot daily/rolling series.
- Supporting notes (`metaorder_distributions.md`, `POWER_LAW_IMPACT_FITS.md`, `prop_vs_nonprop.md`) document the methodology and the impact/crowding analyses.

Outputs such as cached metaorders, summary parquet files, and figures are written by the scripts to the configured `out_files/` and `images/` locations.
