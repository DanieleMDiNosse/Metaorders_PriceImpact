# Bootstrap and Permutation Methods

This repository uses several different resampling schemes. They are not
interchangeable: each script bootstraps the object that matches its dependence
structure and estimand.

This note summarizes the current implementations in:

- `scripts/crowding_analysis.py`
- `scripts/crowding_vs_part_rate.py`
- `scripts/plot_prop_nonprop_fits.py`
- `scripts/metaorder_start_event_study.py`
- `scripts/metaorder_distributions.py`

## Quick reference

| Workflow | Main estimand | Resampling unit | Seed control |
|---|---|---|---|
| `crowding_analysis.py` | pooled and mean-daily crowding correlations | trading date clusters | `SEED` is passed through delegated workflows; core helpers accept `random_state` |
| `crowding_vs_part_rate.py` | alignment curves vs `eta`, effect sizes, optional placebo tests | trading date clusters | `--seed` / `CROWDING_VS_PART_RATE_SEED` |
| `plot_prop_nonprop_fits.py` | retention difference `R_prop - R_client` | pooled date clusters | `RETENTION_RANDOM_STATE` |
| `metaorder_start_event_study.py` | treated-minus-control start-intensity effects | bootstrap by Date, permutation within matching strata | `SEED` |
| `metaorder_distributions.py` | tail-model parameter uncertainty and `xmin` uncertainty | iid resampling of panel values | `POWERLAW_FULL_BOOTSTRAP_RANDOM_STATE` |

## 1. Crowding correlations

The crowding code uses helpers from `moimpact/stats/correlation.py`.

The main cluster-robust helper is:

- `corr_with_cluster_bootstrap_ci_and_permutation_p(...)`

Its design is:

- observations are first aligned on valid `(x, y, cluster)` rows
- clusters are typically trading dates
- bootstrap draws resample clusters with replacement
- each replicate recomputes the correlation on the resampled frame
- confidence intervals are percentile intervals

Important points:

- this is a cluster bootstrap, not an iid bootstrap
- the p-value path is permutation-based when requested
- if a replicate produces an undefined correlation, it is dropped from the
  interval computation

## 2. `scripts/crowding_vs_part_rate.py`

This workflow estimates bin-level quantities after assigning each metaorder to
an `eta` bin.

What is bootstrapped:

- the full vector of bin-level statistics
- top-minus-bottom summaries
- monotonic trend summaries
- centered noise bands used in the Plotly and matplotlib curves

Resampling scheme:

- cluster unit: `Date`
- each bootstrap draw samples dates with replacement
- cluster-level sufficient statistics are aggregated first, then converted back
  into bin-level estimates

Optional placebo / p-value logic:

- within-date permutation can be run through `within_date_permutation_delta_align`
- the number of permutations is controlled by `--permutation-runs` or the
  corresponding YAML fields

Artifacts written by this workflow:

- bin summaries
- daily panels
- effect-size tables
- optional permutation summaries
- optional regression tables
- optional 2D heatmap tables
- `run_manifest.json`

## 3. `scripts/plot_prop_nonprop_fits.py`

The retention workflow uses `moimpact/stats/impact_paths.py` and centers on
the statistic:

- `R = mean(I(tau_end)) / mean(I(tau_start))`

for proprietary and client metaorders separately, plus

- `delta = R_prop - R_client`

Bootstrap scheme:

- derive a cluster label from `Period` when no explicit cluster column is given
- pool the distinct date clusters across both groups
- resample those pooled clusters with replacement
- recompute both retention ratios and their difference on each draw

The output object stores:

- point estimates
- percentile intervals for each group and for `delta`
- bootstrap draw arrays
- a centered-bootstrap p-value for `delta = 0`

This workflow is controlled by the `RETENTION_*` keys in
`config_ymls/plot_prop_nonprop_fits.yml`.

## 4. `scripts/metaorder_start_event_study.py`

The start event-study uses two different resampling ideas:

- confidence intervals: bootstrap by `Date`
- p-values: permutation of the high-`eta` label within exact matching strata

Matching strata are defined by:

- `ISIN`
- `Date`
- clock-time bucket of width `MATCHING_BUCKET_MINUTES`

Bootstrap logic:

- treated-minus-control summary statistics are first computed at the stratum
  level
- date bootstrap draws resample trading dates with replacement
- each draw aggregates the stratum-level summaries over the selected dates

Permutation logic:

- the high-`eta` label is shuffled only within admissible matching strata
- the null therefore preserves the same stratum structure used in the matched
  design

The script also applies multiple-testing adjustments after the raw p-values are
computed:

- Holm adjustment for the primary same-sign tests
- Benjamini-Hochberg for the opposite-sign secondary tests

## 5. `scripts/metaorder_distributions.py`

This workflow optionally runs a full Clauset-style tail-fit bootstrap on each
panel.

What is resampled:

- the raw one-dimensional sample values for one panel

What is recomputed on each draw:

- the candidate `xmin`
- the fitted parameters of each selected tail model
- the model-comparison diagnostics used in the panel annotations and review
  tables

This is not a cluster bootstrap. The panel values are treated as an iid sample
for tail-fitting purposes.

Main controls in `config_ymls/metaorder_distributions.yml`:

- `POWERLAW_FULL_BOOTSTRAP_ENABLED`
- `POWERLAW_FULL_BOOTSTRAP_RUNS`
- `POWERLAW_FULL_BOOTSTRAP_ALPHA`
- `POWERLAW_FULL_BOOTSTRAP_RANDOM_STATE`
- `BOOTSTRAP_DISTRIBUTION`

## 6. Terminology caveats

Some naming in the codebase is historically broader than the exact method:

- some helper names still mention "bootstrap" even when the p-value is
  permutation-based
- the crowding workflows mix bootstrap confidence intervals with permutation
  p-values depending on the question

When documenting or citing a result, state both:

- the estimand
- the resampling unit

## 7. Reproducibility

For paper-facing or long-running analyses:

- record the config file used
- record the seed
- archive the manifest or log produced by the script
- keep the exact output tables rather than only the figures

Scripts that already write a machine-readable manifest:

- `scripts/crowding_vs_part_rate.py`
- `scripts/metaorder_start_event_study.py`
- `scripts/generate_paper_figures.py`

## Related docs

- [`imbalance_and_crowding.md`](imbalance_and_crowding.md)
- [`metaorder_start_event_study.md`](metaorder_start_event_study.md)
- [`metaorder_distributions.md`](metaorder_distributions.md)
- [`market_impact.md`](market_impact.md)
