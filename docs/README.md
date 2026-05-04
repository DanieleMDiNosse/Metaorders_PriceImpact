# Metaorders_PriceImpact Docs

This folder documents the current research pipeline, not just one historical
run. The goal is to keep the markdown aligned with the unified CLI
`scripts/run_analysis.py`, the workflow modules under `moimpact/workflows/`,
the shared helpers under `moimpact/`, and the YAML contracts under
`config_ymls/`.

Start here:

- [`index.md`](index.md): repository overview, workflow map, inputs, outputs,
  and the main entrypoints.
- [`market_impact.md`](market_impact.md): metaorder construction, impact
  normalization, power-law/log fits, surfaces, and impact-path outputs.
- [`imbalance_and_crowding.md`](imbalance_and_crowding.md): within-group,
  cross-group, all-others, and member-level crowding analyses.
- [`crowding_impact_analysis.md`](crowding_impact_analysis.md):
  crowding-conditioned impact, participation, intraday, and overlap analyses.
- [`member_active_overlap_crowding.md`](member_active_overlap_crowding.md):
  member-level proprietary/client active-overlap correlations.
- [`bootstrap_methods.md`](bootstrap_methods.md): bootstrap and permutation
  schemes used across crowding, event-study, retention, and distribution fits.
- [`metaorder_distributions.md`](metaorder_distributions.md): combined client
  vs proprietary distribution diagnostics and tail-model overlays.
- [`metaorder_summary_statistics.md`](metaorder_summary_statistics.md):
  nationality share, member profiles, and mean daily metaorder-volume share.
- [`metaorder_start_event_study.md`](metaorder_start_event_study.md): matched
  event-study of metaorder starts around high-participation anchors.
- [`execution_typology.md`](execution_typology.md): pooled execution-type
  clustering and within-type impact/schedule profiles.
- [`PLOTTING_GUIDE.md`](PLOTTING_GUIDE.md): shared Plotly styling and export
  conventions.

Supporting notes:

- [`crowding_review.md`](crowding_review.md): editorial checklist for the
  crowding section in `paper/main.tex`. This is not the method source of truth;
  use the crowding and bootstrap docs above for the implemented workflow.
- [`crowding_impact_results_summary.md`](crowding_impact_results_summary.md):
  plain-language interpretation of one documented crowding-impact run.
- [`metaorder_start_event_study_threshold_sweep_perm35.md`](metaorder_start_event_study_threshold_sweep_perm35.md):
  archived note for a specific event-study threshold-sweep run.

Quick start from the repo root:

```bash
source /home/danielemdn/miniconda3/etc/profile.d/conda.sh
conda activate main
bash run_all_pipelines.sh
```

Common CLI commands:

```bash
python scripts/run_analysis.py metaorders compute
python scripts/run_analysis.py metaorders distributions
python scripts/run_analysis.py metaorders summary
python scripts/run_analysis.py crowding daily
python scripts/run_analysis.py crowding eta
python scripts/run_analysis.py crowding impact
python scripts/run_analysis.py crowding intraday
python scripts/run_analysis.py crowding overlap
python scripts/run_analysis.py crowding member-overlap
python scripts/run_analysis.py metaorders start-event
python scripts/run_analysis.py metaorders start-time
python scripts/run_analysis.py metaorders intraday-impact
python scripts/run_analysis.py impact overlay
python scripts/run_analysis.py execution schedule
python scripts/run_analysis.py execution cluster
python scripts/run_analysis.py execution typology
python scripts/run_analysis.py members stats
python scripts/run_analysis.py paper figures
```

Most commands also support config overrides via `--config PATH` or the
workflow-specific environment variables. The main environment variables are:

- `METAORDER_COMP_CONFIG`
- `METAORDER_DISTRIBUTIONS_CONFIG`
- `METAORDER_SUMMARY_STATS_CONFIG`
- `CROWDING_CONFIG`
- `CROWDING_IMPACT_CONFIG`
- `CROWDING_INTRADAY_PROFILE_CONFIG`
- `CROWDING_OVERLAP_ANALYSIS_CONFIG`
- `MEMBER_ACTIVE_OVERLAP_CROWDING_CONFIG`
- `METAORDER_INTRADAY_CONFIG`
- `METAORDER_START_TIME_DISTRIBUTION_CONFIG`
- `METAORDER_EXECUTION_SCHEDULE_CONFIG`
- `PLOT_PROP_NONPROP_FITS_CONFIG`
- `PAPER_FIGURES_CONFIG`

To inspect the current command surface:

```bash
python scripts/run_analysis.py --list
```
