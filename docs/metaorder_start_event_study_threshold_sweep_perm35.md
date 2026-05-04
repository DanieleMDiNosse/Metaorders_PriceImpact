# Metaorder Start Event-Study Threshold Sweep (35 Permutations)

This note summarizes the threshold-sweep rerun of
`scripts/run_analysis.py metaorders start-event` using five `high_eta` cutoffs and a
small permutation budget.

## Run setup

Command used:

```bash
conda activate main
python scripts/run_analysis.py metaorders start-event \
  --analysis-tag metaorder_start_event_study_threshold_sweep_perm35 \
  --high-eta-quantiles 0.5,0.6,0.7,0.8,0.9 \
  --bootstrap-runs 0 \
  --permutation-runs 35 \
  --n-jobs 0
```

Main outputs:

- `out_files/ftsemib/metaorder_start_event_study_threshold_sweep_perm35/event_study_curves_threshold_sweep.csv`
- `out_files/ftsemib/metaorder_start_event_study_threshold_sweep_perm35/event_study_diagnostics_threshold_sweep.csv`
- `out_files/ftsemib/metaorder_start_event_study_threshold_sweep_perm35/run_manifest.json`
- `images/ftsemib/metaorder_start_event_study_threshold_sweep_perm35/png/`
- `images/ftsemib/metaorder_start_event_study_threshold_sweep_perm35/html/`

The run keeps the exact same matched design as the baseline event-study:

- controls matched within `(ISIN, Date, clock bucket)`
- threshold sweep over `high_eta_quantile in {0.5, 0.6, 0.7, 0.8, 0.9}`
- baseline `all_others` and robustness `exclude_same_actor`
- no bootstrap confidence intervals
- 35 within-stratum permutations per threshold

## Important caveat

This run is useful as a screening exercise, not as final inference.

With only 35 permutations, the raw p-values move in steps of `1 / (35 + 1) =
0.0278`. After the within-curve BH correction, many significant bins pile up at
`0.0317` or `0.0370`. That means the sign/significance map is informative, but
the p-values are still coarse.

Because `BOOTSTRAP_RUNS=0`, the confidence-interval columns are not populated in
this run. Interpretation should therefore rely on:

- the sign and magnitude of `excess_rate`
- the threshold pattern across `high_eta_quantile`
- the permutation p-values as a rough robustness filter

## Main findings

### 1. Client baseline same-sign crowding is stable in the earlier pre bins

In the baseline `all_others` specification for client anchors, the three earlier
pre-start bins remain positive and permutation-significant for every threshold:

- `[-20,-15)`, `[-15,-10)`, `[-10,-5)` are all significant from `q=0.5` to
  `q=0.9`
- the excess attenuates as the threshold becomes stricter, but it stays clearly
  positive

Examples:

- at `q=0.5`, the same-sign excess is about `0.0090`, `0.0127`, `0.0128`
- at `q=0.9`, it is still about `0.0061`, `0.0084`, `0.0054`

The immediate pre-start bin `[-5,0)` is different:

- it is never significant
- it turns negative for thresholds above the median split

So the threshold sweep supports a persistent early pre-start same-sign crowding
pattern for client flow, but not a robust last-minute spike.

### 2. Proprietary baseline same-sign crowding is weaker and fades at the top threshold

For proprietary anchors in the baseline `all_others` specification:

- the earlier pre bins are positive and significant from `q=0.5` through `q=0.8`
- the magnitudes are much smaller than for client flow, typically around
  `0.001` to `0.003`
- at `q=0.9`, only `[-15,-10)` remains significant, while `[-20,-15)` and
  `[-10,-5)` lose significance

Again, the immediate pre-start bin `[-5,0)` is not robust:

- it is never significant
- it is positive at `q=0.5` and `q=0.6`
- it flips negative from `q=0.7` onward

This reinforces the earlier interpretation: proprietary same-sign clustering is
present, but much weaker than the client pattern and not concentrated in a
stable last-pre-start spike.

### 3. Same-actor exclusion removes the baseline same-sign pre-start excess

Once starts from the same actor are excluded, the earlier same-sign pre bins
flip negative for both groups across the whole threshold grid.

For client anchors:

- `[-20,-15)`, `[-15,-10)`, `[-10,-5)` are negative and significant at every
  threshold
- `[-5,0)` remains non-significant

For proprietary anchors:

- the same three earlier pre bins are also negative and significant at every
  threshold
- `[-5,0)` remains non-significant

This is the main robustness message of the sweep: the baseline same-sign
pre-start excess does not survive the removal of same-actor neighbors. The raw
clustering pattern is therefore not strong evidence of broad cross-actor
synchronization.

### 4. Opposite-sign behavior is asymmetric across groups in the baseline specification

For client anchors in the baseline `all_others` specification:

- opposite-sign excess is negative and significant in every bin except `[-5,0)`
- this holds across all five thresholds

For proprietary anchors in the baseline `all_others` specification:

- opposite-sign bins are mostly not significant
- the baseline proprietary signal is therefore much more one-sided than the
  client pattern

After same-actor exclusion, opposite-sign bins become negative and significant
for both groups across the whole threshold grid.

### 5. The post-start same-sign pattern is also threshold-stable

Client baseline:

- `(5,10]`, `(10,15]`, `(15,20]` are positive and significant at all thresholds
- `(0,5]` is weakly negative and only sometimes significant

Proprietary baseline:

- `(0,5]` is negative and significant
- the later post bins are positive and significant, although weaker than for
  client flow

After same-actor exclusion, the same-sign post bins are negative and significant
for both groups, again showing that same-actor repetition was an important part
of the raw local clustering.

## Interpretation

The threshold sweep does not overturn the core story from the single-threshold
event-study.

What remains stable across `q=0.5` to `q=0.9`:

- client flow shows a persistent same-sign excess in the earlier pre-start bins
- proprietary flow shows the same pattern, but more weakly
- the last pre-start bin `[-5,0)` is not robust
- removing same-actor neighbors removes the baseline same-sign pre-start excess

What changes with the threshold:

- client same-sign pre-start excess shrinks gradually but remains visible
- proprietary same-sign pre-start excess weakens faster and partly disappears at
  the strictest threshold

So the sweep points to a stable early pre-start clustering pattern for
high-participation client starts, but it also confirms that the broad
cross-actor interpretation should be treated cautiously because the same-actor
robustness check reverses the sign.
