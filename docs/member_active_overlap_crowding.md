# Member Active-Overlap Crowding

## Objective

This analysis studies whether specific market members display systematic
directional co-movement between their proprietary metaorders and the metaorders
executed on behalf of their own clients. The unit of analysis is therefore the
member, not the market as a whole. The empirical object of interest is whether a
given member repeatedly trades proprietary metaorders in the same direction as,
or in the opposite direction to, the client flow that is active during the same
execution interval.

The analysis uses the active-overlap construction introduced for the
crowding-impact exercise. For each proprietary metaorder, the client environment
is restricted to client metaorders executed through the same member and active
during the proprietary execution interval. The main specification further
requires the client metaorders to refer to the same ISIN. This produces a local
measure of prop-client alignment at the member-stock-day-interval level.

The workflow is implemented in:

- `scripts/run_analysis.py crowding member-overlap`
- `moimpact/stats/active_overlap.py`
- `config_ymls/member_active_overlap_crowding.yml`

## Construction of the Active Client Environment

For a proprietary target metaorder $i$ executed by member $m$, let
$\mathcal{C}_i$ denote the set of eligible client metaorders executed through
the same member. In the main specification, eligibility requires the same
member, same ISIN, same date, and a positive temporal overlap with the
proprietary execution interval.

For a target metaorder $i$ and an eligible client metaorder $j$, define the
active-overlap weight:

$$
\omega_{ij}
= \frac{
\left[
\min\!\left(t_i^{\mathrm{end}}, t_j^{\mathrm{end}}\right)
- \max\!\left(t_i^{\mathrm{start}}, t_j^{\mathrm{start}}\right)
\right]_+
}{
t_i^{\mathrm{end}} - t_i^{\mathrm{start}}
}.
$$

For zero-duration proprietary targets, the point-in-time convention is used:

$$
\omega_{ij}
=
\begin{cases}
1, & t_j^{\mathrm{start}} \le t_i^{\mathrm{start}} \le t_j^{\mathrm{end}}, \\
0, & \text{otherwise}.
\end{cases}
$$

The active client imbalance faced by proprietary target $i$ is:

$$
B_i
=
\frac{
\sum_{j \in \mathcal{C}_i} \omega_{ij} Q_j \varepsilon_j
}{
\sum_{j \in \mathcal{C}_i} \omega_{ij} Q_j
},
$$

where $Q_j$ is the size of client metaorder $j$ and $\varepsilon_j \in \{-1,+1\}$
is its direction. The proprietary target itself is never included in
$\mathcal{C}_i$. If the denominator is zero, $B_i$ is undefined and the
proprietary target is excluded from the corresponding correlation estimate.

Positive $B_i$ means that the active client environment is buy-dominated;
negative $B_i$ means that it is sell-dominated.

## Alignment and Correlation

Two related statistics are useful, but they answer different questions.

The target-level signed alignment is:

$$
a_i = \varepsilon_i^{\mathrm{prop}} B_i.
$$

It is positive when proprietary flow and active client imbalance have the same
sign, negative when they have opposite signs, and close to zero when the active
client environment is balanced. The cross-target average

$$
\bar{a}_m
=
\frac{1}{n_m}
\sum_{i:\operatorname{Member}(i)=m} a_i
$$

is an economically direct measure of signed exposure: it tells whether the
member's proprietary metaorders tend, on average, to face client flow in the same
or opposite direction.

The per-member correlation is:

$$
\rho_m
=
\operatorname{Corr}\!\left(
\varepsilon_i^{\mathrm{prop}}, B_i
\;\middle|\;
\operatorname{Member}(i)=m
\right).
$$

The correlation is a normalized co-movement measure. It subtracts the average
proprietary direction and the average client imbalance before measuring their
association. This is important because a positive mean alignment can arise
mechanically if a member tends to buy on its proprietary account and its client
environment is also buy-dominated, even without episode-by-episode co-movement.
The correlation is therefore stricter: it asks whether proprietary direction
varies with the active client imbalance across observations.

Thus $a_i$ and $\rho_m$ are complementary:

- $a_i$ and $\bar{a}_m$ measure signed alignment exposure;
- $\rho_m$ measures normalized directional co-movement;
- if both are positive for the same member, the evidence for repeated
  prop-client alignment is stronger;
- if $\bar{a}_m$ is positive but $\rho_m$ is near zero, the signal may mostly
  reflect unconditional directional bias rather than local co-movement.

The result tables emphasize $\rho_m$ for inference and report signed alignment
as a descriptive diagnostic.

## Scopes and Timing Buckets

The main scope is `same_isin`.

| Scope | Definition | Role |
|---|---|---|
| `same_isin` | Same member, same ISIN, same date, active temporal overlap | Main local specification. |
| `all_isin` | Same member, same date, active temporal overlap, any ISIN | Robustness check. |

The `all_isin` scope is useful for checking whether the signal survives when
all simultaneous same-member client activity is pooled. Its interpretation is
weaker because it mixes client flow across instruments.

Within each scope, the active client environment is decomposed into timing
buckets.

| Bucket | Definition | Role |
|---|---|---|
| `all_active` | All eligible client metaorders active during the proprietary target | Baseline active-overlap measure. |
| `preexisting_at_prop_start` | Eligible client metaorders already active when the proprietary target starts | Most informative timing bucket for alignment with already active client flow. |
| `starts_during_prop` | Eligible client metaorders that start during the proprietary target interval | Descriptive co-movement bucket. |

The timing buckets do not identify intent or information transmission. They are
interval-level diagnostics based on reconstructed metaorders.

## Minimum Observation Filter

Per-member statistics are reported with the filter `MIN_OBS_PER_MEMBER = 30`.
For member $m$, scope $s$, and timing bucket $b$, define:

$$
n_{\mathrm{valid}}(m,s,b)
=
\#\left\{
i:
\operatorname{Member}(i)=m,\,
\operatorname{scope}(i)=s,\,
\operatorname{bucket}(i)=b,\,
B_i \text{ is finite}
\right\}.
$$

A proprietary target contributes to $n_{\mathrm{valid}}$ only if there is
strictly positive active same-member client volume in the selected environment.
The threshold therefore counts valid proprietary target observations, not raw
client metaorders.

The filter means:

$$
n_{\mathrm{valid}}(m,s,b) \ge 30.
$$

It does not require 30 client metaorders, nor 30 distinct overlapping client
metaorders. It requires 30 proprietary targets with a well-defined active client
imbalance. Members below the threshold remain in the output tables but are
excluded from the per-member plots and from the filtered per-member summaries.

## Run and Outputs

The current run used:

```bash
conda activate main
python scripts/run_analysis.py crowding overlap
python scripts/run_analysis.py crowding member-overlap
```

The member-overlap command expects the active-overlap target table written by
`scripts/run_analysis.py crowding overlap` unless `INPUT_PATH` is set
explicitly.

Input:

- `out_files/ftsemib/crowding_overlap_analysis/overlap_features.parquet`

Main outputs:

- `out_files/ftsemib/member_active_overlap_crowding/active_member_overlap_targets.parquet`
- `out_files/ftsemib/member_active_overlap_crowding/global_correlations.csv`
- `out_files/ftsemib/member_active_overlap_crowding/per_member_correlations.csv`
- `out_files/ftsemib/member_active_overlap_crowding/member_window_correlations.csv`
- `out_files/ftsemib/member_active_overlap_crowding/member_comovement_series.csv`
- `out_files/ftsemib/member_active_overlap_crowding/run_manifest.json`

Figures:

- `images/ftsemib/member_active_overlap_crowding/png/global_lead_lag_correlations.png`
- `images/ftsemib/member_active_overlap_crowding/png/per_member_correlations_same_isin_all_active.png`
- `images/ftsemib/member_active_overlap_crowding/png/member_comovement_same_isin_all_active.png`
- `images/ftsemib/member_active_overlap_crowding/png/member_window_heatmap_same_isin_all_active.png`

The run manifest records commit `3713ecde63f7edf72134edf62ace9564750364ab`,
`BOOTSTRAP_RUNS = 1000`, `RANDOM_STATE = 0`, and `N_JOBS = 4`.

## Sample Size

The same-ISIN active-overlap restriction is severe. This is expected: the
analysis only retains proprietary targets that have active same-member client
flow in the selected environment.

| Scope | Bucket | Proprietary targets | Valid active-overlap targets | Members with valid targets |
|---|---|---:|---:|---:|
| `same_isin` | `all_active` | 588,334 | 2,320 | 12 |
| `same_isin` | `preexisting_at_prop_start` | 588,334 | 1,446 | 11 |
| `same_isin` | `starts_during_prop` | 588,334 | 1,225 | 12 |
| `all_isin` | `all_active` | 588,334 | 20,636 | 15 |
| `all_isin` | `preexisting_at_prop_start` | 588,334 | 17,274 | 14 |
| `all_isin` | `starts_during_prop` | 588,334 | 7,234 | 15 |

The main `same_isin / all_active` specification contains 2,320 valid
proprietary targets. This sample is small relative to the full proprietary
metaorder panel, but it is the relevant sample for local prop-client overlap.

## Pooled Benchmark

Although the analysis is designed to identify member-level heterogeneity, pooled
statistics provide a useful benchmark for the average sign and scale of the
effect.

| Scope | Bucket | Corr | 95% CI | Valid targets | Dates | Members |
|---|---|---:|---:|---:|---:|---:|
| `same_isin` | `all_active` | 0.036 | [-0.032, 0.104] | 2,320 | 226 | 12 |
| `same_isin` | `preexisting_at_prop_start` | 0.051 | [-0.034, 0.137] | 1,446 | 202 | 11 |
| `same_isin` | `starts_during_prop` | 0.037 | [-0.038, 0.105] | 1,225 | 200 | 12 |
| `all_isin` | `all_active` | 0.011 | [-0.055, 0.077] | 20,636 | 251 | 15 |
| `all_isin` | `preexisting_at_prop_start` | 0.011 | [-0.057, 0.079] | 17,274 | 250 | 14 |
| `all_isin` | `starts_during_prop` | 0.032 | [-0.045, 0.107] | 7,234 | 249 | 15 |

The pooled correlations are small and their Date-cluster bootstrap confidence
intervals include zero. The average pooled effect is therefore weak. This does
not rule out member-specific behavior, because positive and negative members can
offset each other in the pooled sample.

The corresponding target-level signed alignment summaries are:

| Scope | Bucket | Mean $a_i = \varepsilon_i^{\mathrm{prop}} B_i$ | Median | Share positive |
|---|---|---:|---:|---:|
| `same_isin` | `all_active` | 0.034 | 0.571 | 0.516 |
| `same_isin` | `preexisting_at_prop_start` | 0.050 | 1.000 | 0.525 |
| `same_isin` | `starts_during_prop` | 0.033 | 0.330 | 0.516 |
| `all_isin` | `all_active` | 0.010 | 0.004 | 0.500 |

The average signed alignment is positive in the same-ISIN specifications, but
the share of positive observations is only slightly above one half. The high
medians should therefore not be read as evidence that the typical signal is
large in an unconditional sense. They arise because the local active client
environment is often one-sided. In the main `same_isin / all_active`
specification, 68.1% of valid targets have only one active client metaorder and
90.0% have $|B_i|=1$. The signed alignment distribution is consequently close
to a two-point distribution: 46.6% of observations have $a_i=+1$ and 43.4% have
$a_i=-1$. Since the positive mass is only slightly larger than the negative
mass, the mean remains close to zero, while the median falls on the positive
side of the distribution.

The effect is even more extreme in `same_isin / preexisting_at_prop_start`: all
valid observations have exactly one active client metaorder, so $a_i \in
\{-1,+1\}$. A positive share of 52.5% is enough to make the median equal to
$+1$, while the mean is only 0.050 because the negative observations nearly
offset the positive ones. The median is therefore informative about the discrete
structure of the local sample, but the mean, share positive, and per-member
correlations are more useful for assessing the economic strength of alignment.

## Member-Level Results

The central object is the distribution of per-member correlations. In the main
`same_isin / all_active` specification, five members satisfy
$n_{\mathrm{valid}} \ge 30$.

| Member | Corr | 95% CI | Valid targets | Dates |
|---:|---:|---:|---:|---:|
| 91138 | -0.019 | [-0.339, 0.265] | 50 | 35 |
| 91149 | 0.177 | [-0.009, 0.388] | 195 | 104 |
| 91157 | -0.006 | [-0.117, 0.084] | 946 | 68 |
| 96862 | 0.431 | [0.249, 0.614] | 89 | 61 |
| 116286 | 0.033 | [-0.075, 0.156] | 981 | 162 |

Most members with sufficient local overlap are close to zero. Member `96862`
stands out with a large positive correlation and a confidence interval that
excludes zero. Member `91149` is also positive, although its confidence interval
barely includes zero in the baseline bucket.

The timing decomposition highlights the member-bucket pairs with the strongest
positive signals outside the baseline `all_active` summary. The table is not
restricted to one row per bucket: `starts_during_prop` appears twice because two
different members display sizeable positive correlations in that timing bucket.

| Bucket | Member | Corr | 95% CI | Valid targets | Dates |
|---|---:|---:|---:|---:|---:|
| `preexisting_at_prop_start` | 96862 | 0.382 | [0.059, 0.685] | 34 | 24 |
| `starts_during_prop` | 91149 | 0.269 | [0.079, 0.451] | 99 | 67 |
| `starts_during_prop` | 96862 | 0.501 | [0.273, 0.719] | 62 | 54 |

The `preexisting_at_prop_start` signal for member `96862` is particularly
important because the client flow is already active when the proprietary
metaorder starts. The estimate is based on 34 valid observations, so it should
be interpreted as a statistically visible local signal that warrants further
inspection, not as conclusive evidence about intent.

The `starts_during_prop` signals for members `91149` and `96862` indicate strong
local co-movement, but they do not establish that proprietary trading followed
client flow, because the client metaorders in that bucket start after the
proprietary target has already begun.

Mean correlations among members passing the observation threshold are:

| Scope | Bucket | Members passing | Mean corr |
|---|---|---:|---:|
| `same_isin` | `all_active` | 5 | 0.123 |
| `same_isin` | `preexisting_at_prop_start` | 5 | 0.132 |
| `same_isin` | `starts_during_prop` | 4 | 0.203 |
| `all_isin` | `all_active` | 8 | -0.060 |
| `all_isin` | `preexisting_at_prop_start` | 8 | 0.026 |
| `all_isin` | `starts_during_prop` | 8 | -0.061 |

The same-ISIN means are positive, while the all-ISIN means are close to zero or
negative. This reinforces the interpretation that any positive alignment signal
is local to same-stock overlapping activity rather than a broad same-member
cross-instrument phenomenon.

The paper-facing diagnostic now complements the per-member bar chart with
window-level co-movement curves for the two members with the strongest positive
global same-ISIN all-active correlations. The series are non-overlapping
5-trading-day averages of proprietary target direction and active client
imbalance, computed on valid target observations without imposing a per-window
minimum-count filter. This makes the high correlations easier to inspect without
the sparsity and label problems of the member-window heatmap. Because some
5-day windows contain few targets, the curves are intended as a descriptive
co-movement diagnostic; inference is based on the target-level correlations and
their bootstrap confidence intervals.

## Interpretation

The evidence does not indicate a uniform behavior across members. Instead, the
active-overlap analysis identifies a small set of member-level deviations from
neutral prop-client co-movement.

The main empirical conclusions are:

- in the pooled sample, prop-client active-overlap correlations are small;
- in the main same-ISIN specification, most members with enough observations are
  close to zero;
- member `96862` exhibits a robust positive same-ISIN active-overlap
  correlation, including when client flow is already active at the proprietary
  start time;
- member `91149` exhibits positive co-movement in the bucket where client
  metaorders start during the proprietary interval;
- the all-ISIN robustness does not show a stable positive pattern, suggesting
  that the informative signal is local rather than cross-instrument.

A concise scientific summary is:

> Member-level active-overlap correlations reveal limited evidence of systematic
> prop-client alignment in the pooled sample, but they identify specific members
> with sizeable and statistically visible local co-movement. The strongest
> signal is observed for member `96862` in the same-ISIN environment, including
> the preexisting-client-flow bucket. These results are best interpreted as
> evidence of member-specific episodes of directional alignment, not as proof of
> front running or as a market-wide regularity.

Positive alignment is compatible with several mechanisms, including common
information, inventory management, client facilitation, execution
synchronization, or misuse of client-flow information. The present analysis
therefore serves as a screening tool for identifying members and windows that
deserve finer transaction-level inspection.
