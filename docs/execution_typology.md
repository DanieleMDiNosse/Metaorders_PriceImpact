# Execution Typology

This document covers `scripts/run_analysis.py execution typology`, which builds a
pooled proprietary-vs-client execution typology from the filtered metaorder
tables produced by `scripts/run_analysis.py metaorders compute`.

## Goal

The workflow clusters metaorders into interpretable execution types and then
compares:

- how proprietary and client metaorders populate those types
- how impact profiles differ within type
- how execution schedules differ within type

The default design clusters the pooled proprietary + client sample once so the
resulting types are directly comparable across groups.

## Inputs

Default config:

- `config_ymls/metaorder_execution_typology.yml`

Default inputs:

- `out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_{LEVEL}_proprietary.parquet`
- `out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_{LEVEL}_non_proprietary.parquet`

The workflow expects the filtered per-metaorder tables to contain:

- tabular execution context such as `Q`, `Q/V`, `Participation Rate`, `Vt/V`,
  `N Child`, `Daily Vol`, and `Period`
- packed execution paths:
  - `child_time_norm`
  - `child_volume_fraction`
- packed impact paths:
  - `partial_impact`
- scalar impact horizons:
  - `Impact`
  - `Impact_1m`
  - `Impact_10m`
  - `Impact_30m`
  - `Impact_60m`

## Feature set

The current implementation uses a behavior-led feature set combining:

- urgency/context features:
  - log size
  - log `Q/V`
  - log `eta`
  - log `Vt/V`
  - log duration
  - log child count
  - asinh daily volatility
- schedule-shape features:
  - front-25 share
  - front-50 share
  - back-25 share
  - execution center of mass
  - schedule HHI
  - L1 distance from TWAP
- impact-shape features:
  - absolute impact at end, 1m, 10m, 30m, and 60m
  - 30m and 60m retention relative to end impact
  - peak in-execution absolute impact
  - peak overshoot relative to end impact

## Clustering

The default clustering path is:

- robust scaling of the engineered features
- PCA reduction to the variance threshold set in the YAML
- pooled `MiniBatchKMeans`
- silhouette-based `k` selection on a stratified sample

The workflow then computes rule-based label metrics for each cluster and writes:

- `type_code`
- `auto_type_label`
- `type_label`

`type_label` can be overridden through the YAML after an initial run.

## Outputs

Tables are written under:

- `out_files/{DATASET_NAME}/{ANALYSIS_TAG}/`

Figures are written under:

- `images/{DATASET_NAME}/{ANALYSIS_TAG}/html/`
- `images/{DATASET_NAME}/{ANALYSIS_TAG}/png/`

Canonical outputs:

- `clustered_metaorders.parquet`
- `cluster_summary.csv`
- `group_type_shares.csv`
- `impact_profiles_by_type.csv`
- `schedule_profiles_by_type.csv`
- `silhouette_scores.csv`
- `pca_report.json`
- `run_manifest.json`

Canonical figure stems:

- `execution_typology_feature_heatmap`
- `execution_typology_group_shares`
- `execution_typology_pca_scatter`
- `execution_typology_impact_profiles`
- `execution_typology_schedule_profiles`

## Performance

The workflow is designed for the large pooled member-level sample:

- chunked packed-path feature extraction
- optional process-based parallelization via `N_JOBS`
- vectorized aggregation for summaries
- sampled silhouette scoring
- `MiniBatchKMeans` for the final clustering fit

## Running

From the repo root:

```bash
source /home/danielemdn/miniconda3/etc/profile.d/conda.sh
conda activate main
python scripts/run_analysis.py execution typology
```

Useful overrides:

- `--config PATH`
- `--analysis-tag`
- `--k-min`
- `--k-max`
- `--n-jobs`
- `--chunk-size`
- `--silhouette-sample-size`
- `--pca-scatter-sample-size`
- `--no-progress`
