# Metaorders_PriceImpact Docs

This folder documents the current research pipeline, not just one historical
run. The goal is to keep the markdown aligned with the live scripts under
`scripts/`, the shared helpers under `moimpact/`, and the YAML contracts under
`config_ymls/`.

Start here:

- [`index.md`](index.md): repository overview, workflow map, inputs, outputs,
  and the main entrypoints.
- [`market_impact.md`](market_impact.md): metaorder construction, impact
  normalization, power-law/log fits, surfaces, and impact-path outputs.
- [`imbalance_and_crowding.md`](imbalance_and_crowding.md): within-group,
  cross-group, all-others, and member-level crowding analyses.
- [`bootstrap_methods.md`](bootstrap_methods.md): bootstrap and permutation
  schemes used across crowding, event-study, retention, and distribution fits.
- [`metaorder_distributions.md`](metaorder_distributions.md): combined client
  vs proprietary distribution diagnostics and tail-model overlays.
- [`metaorder_summary_statistics.md`](metaorder_summary_statistics.md):
  nationality share, member profiles, and mean daily metaorder-volume share.
- [`metaorder_start_event_study.md`](metaorder_start_event_study.md): matched
  event-study of metaorder starts around high-participation anchors.
- [`PLOTTING_GUIDE.md`](PLOTTING_GUIDE.md): shared Plotly styling and export
  conventions.

Supporting notes:

- [`crowding_review.md`](crowding_review.md): editorial checklist for the
  crowding section in `paper/main.tex`. This is not the method source of truth;
  use the crowding and bootstrap docs above for the implemented workflow.

Quick start from the repo root:

```bash
source /home/danielemdn/miniconda3/etc/profile.d/conda.sh
conda activate main
bash run_all_pipelines.sh
```

Common script-level entrypoints:

```bash
python scripts/metaorder_computation.py
python scripts/metaorder_distributions.py
python scripts/metaorder_summary_statistics.py
python scripts/crowding_analysis.py
python scripts/metaorder_start_event_study.py
python scripts/metaorder_intraday_analysis.py
python scripts/plot_prop_nonprop_fits.py
python scripts/metaorder_clustering.py
python scripts/generate_paper_figures.py
```

Most scripts also support config overrides via environment variables. The main
ones are:

- `METAORDER_COMP_CONFIG`
- `METAORDER_DISTRIBUTIONS_CONFIG`
- `METAORDER_SUMMARY_STATS_CONFIG`
- `CROWDING_CONFIG`
- `METAORDER_INTRADAY_CONFIG`
- `PLOT_PROP_NONPROP_FITS_CONFIG`
- `PAPER_FIGURES_CONFIG`
