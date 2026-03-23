# Metaorder Summary Statistics

This document covers `scripts/metaorder_summary_statistics.py`, which produces
the non-distribution descriptive outputs built from the proprietary and client
metaorder dictionaries plus the filtered trade tapes.

## Scope

The script exports four summary families:

- aggressive-member nationality share
- nationality context versus overall traded volume
- member-level metaorder profiles
- mean daily metaorder-volume share by ISIN

It intentionally does not handle the distribution panels or tail-model fits;
those live in [`metaorder_distributions.md`](metaorder_distributions.md).

## Inputs

Default config:

- `config_ymls/metaorder_summary_statistics.yml`

Environment-variable override:

- `METAORDER_SUMMARY_STATS_CONFIG`

The script loads:

- proprietary and client metaorder dictionaries
- per-ISIN trade tapes under `PARQUET_PATH`
- optional member nationality metadata through
  `data/members_nationality.parquet`

## Split vs pooled mode

By default the script conditions on the proprietary/client split.

You can pool both groups into one "all metaorders" view with:

```bash
python scripts/metaorder_summary_statistics.py \
  --condition-on-client-proprietary false
```

This affects:

- figure stems
- saved table stems
- whether legends and traces are group-specific or pooled

## Nationality share

When `LEVEL="member"`, the script infers the aggressive-side member
nationality as `it`, `foreign`, `unknown`, or `mixed` using the shared sample
builder in `moimpact/metaorder_distribution_samples.py`.

Figure stems:

- split mode:
  `nationality_share_prop_vs_client[ _member_nationality_{...} ]`
- pooled mode:
  `nationality_share_all_metaorders[ _member_nationality_{...} ]`

This figure is written under:

- `images/{DATASET_NAME}/{LEVEL}_metaorder_summary_statistics/`

## Nationality context table

The nationality-context workflow compares:

- known trade volume by nationality
- detected metaorder counts by nationality
- detected metaorder volume by nationality
- the share of each nationality's traded volume that ends up in detected
  metaorders

Saved tables:

- `nationality_overall_context[ _member_nationality_{...} ].csv`
- `nationality_overall_context[ _member_nationality_{...} ].parquet`

Saved figure stems:

- split mode:
  `nationality_overall_context[ _member_nationality_{...} ]`
- pooled mode:
  `nationality_overall_context_all_metaorders[ _member_nationality_{...} ]`

## Member metaorder profiles

When `LEVEL="member"`, the script builds a member-level table with:

- number of detected metaorders
- total child orders across detected metaorders
- total filtered aggressive trades by member

Outputs:

- `member_metaorder_profiles[ _member_nationality_{...} ].parquet`
- `member_metaorder_profiles_all_metaorders[ _member_nationality_{...} ].parquet`

The figure combines:

- a descending rank plot of metaorder counts
- a scatter of total child orders vs total filtered trades

Members with filtered trades but zero detected metaorders remain in the member
table and in the scatter plot.

## Mean daily metaorder-volume share by ISIN

For each ISIN-day pair the script computes:

- total market volume from the filtered tape
- proprietary metaorder volume
- client metaorder volume
- the corresponding daily ratios

It then averages those daily ratios by ISIN and writes the figure:

- split mode:
  `mean_daily_metaorder_volume_share[ _member_nationality_{...} ]`
- pooled mode:
  `mean_daily_metaorder_volume_share_all_metaorders[ _member_nationality_{...} ]`

The underlying day-level table is built in memory through
`compute_daily_metaorder_share_table(...)`.

## Output layout

Figures:

- `images/{DATASET_NAME}/{LEVEL}_metaorder_summary_statistics/html/`
- `images/{DATASET_NAME}/{LEVEL}_metaorder_summary_statistics/png/`

Tables:

- `out_files/{DATASET_NAME}/{LEVEL}_metaorder_summary_statistics/`

Log files:

- `out_files/{DATASET_NAME}/logs/metaorder_summary_statistics_{LEVEL}_{...}.log`

## Typical usage

From the repo root:

```bash
source /home/danielemdn/miniconda3/etc/profile.d/conda.sh
conda activate main
python scripts/metaorder_summary_statistics.py
```

Pooled mode:

```bash
python scripts/metaorder_summary_statistics.py \
  --condition-on-client-proprietary false
```

## Related docs

- [`metaorder_distributions.md`](metaorder_distributions.md)
- [`market_impact.md`](market_impact.md)
- [`PLOTTING_GUIDE.md`](PLOTTING_GUIDE.md)
