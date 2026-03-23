# Crowding Paper Checklist

This file is an editorial checklist for the crowding discussion in
`paper/main.tex`. It is intentionally separate from the method docs:

- use [`imbalance_and_crowding.md`](imbalance_and_crowding.md) for the current
  implementation
- use [`bootstrap_methods.md`](bootstrap_methods.md) for the current inference
  logic

## What to verify before calling the crowding section final

- Terminology matches the code.
  - If the section says "co-impact", the text should actually estimate or show
    a co-impact object rather than only crowding correlations.
- Every imbalance object is defined before it is interpreted.
  - within-group
  - cross-group environment
  - all-others
  - member-level client environment
- The inference method is stated explicitly for every reported interval or
  p-value.
  - bootstrap unit
  - number of draws
  - any multiple-testing correction
- The zero-denominator and missing-environment edge cases are explained.
  - no source-group metaorders
  - single-metaorder days for leave-one-out objects
- Daily correlation thresholds are stated once and used consistently.
  - for example the `MIN_N` threshold in the crowding config
- Member-level results are described with the right aggregation level.
  - member-day is not the same object as ISIN-day
- Any claim of statistical significance is backed by a method the code actually
  computes.
  - especially for heatmaps and rolling plots
- Figure captions describe exactly what the plotted object is.
  - unconditional distribution
  - conditional-on-direction distribution
  - rolling average vs daily value

## Good references inside the repo

- `scripts/crowding_analysis.py`
- `scripts/crowding_vs_part_rate.py`
- `docs/imbalance_and_crowding.md`
- `docs/bootstrap_methods.md`

## Practical recommendation

When updating `paper/main.tex`, pull numbers from saved tables or logs under
`out_files/` rather than retyping them manually. The crowding section is easier
to keep consistent when the narrative follows the exact file names produced by
the scripts.
