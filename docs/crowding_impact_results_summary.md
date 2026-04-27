# Crowding Impact Analysis: Short Summary

This note is a compact companion to [`crowding_impact_analysis.md`](crowding_impact_analysis.md).
It explains, in plain language, what the `scripts/crowding_impact_analysis.py`
workflow does and what the current `ftsemib` run found on `2026-04-22`.

## Question

The analysis asks whether end-of-execution price impact is larger when a
metaorder trades in the same direction as the stock-day order-flow imbalance,
and whether this relationship differs between:

- client metaorders
- proprietary metaorders

The reported run uses:

- `CROWDING_SCOPE: all`
- dataset `ftsemib`
- `200` date-cluster bootstrap replications
- benchmark relative sizes `phi = 10^-3` and `phi = 10^-2`

The exact run configuration is recorded in:

- `out_files/ftsemib/crowding_impact/run_manifest.json`

## Data and Sample

The workflow starts from the filtered per-metaorder tables:

- `out_files/ftsemib/metaorders_info_sameday_filtered_member_non_proprietary.parquet`
- `out_files/ftsemib/metaorders_info_sameday_filtered_member_proprietary.parquet`

The final working sample in the documented run is:

| Group | Metaorders | Dates | ISINs |
|---|---:|---:|---:|
| Client | 255,499 | 251 | 41 |
| Proprietary | 588,172 | 251 | 41 |

For each metaorder `i`, crowding is measured as:

```text
c_i = epsilon_i * imb_i
```

where `epsilon_i` is the trade direction of metaorder `i`, and `imb_i` is the
leave-one-out signed imbalance from all other metaorders on the same stock-day.

High `c_i` means the metaorder is aligned with the surrounding stock-day flow.
Low `c_i` means it is trading against that flow.

## Analysis Carried Out

The workflow has three main blocks.

### 1. Crowding-conditioned impact curves

For client and proprietary flow separately, the script:

1. splits crowding into three group-specific buckets: low, mid, and high
2. fits impact curves inside each bucket

```text
E[I | phi, q] = Y_q * phi^gamma_q
```

3. evaluates fitted impact at `phi = 10^-3` and `phi = 10^-2`
4. compares the high-crowding fit to the low-crowding fit

Uncertainty is measured with a date-cluster bootstrap, so the confidence
intervals account for day-level dependence.

### 2. Participation-rate robustness

The same crowding comparison is repeated after splitting the sample into:

- low `eta`
- high `eta`

This checks whether the crowding effect is stable once participation rate is
coarsely controlled for.

### 3. Joint-bin regression

The additional robustness block controls jointly for:

- `eta = Participation Rate`
- `V_t / V`
- `|imbalance|`

The script first builds a `6 x 6 x 4` binning grid over these variables, keeps
cells with at least `20` observations, and regresses the log of the cell mean
impact on the bin centers:

```text
log(E[I | cell]) = log(Y) + beta_eta log(eta) + beta_f log(V_t/V) + beta_c |imb|
```

This regression is run separately for client and proprietary flow, and the
coefficients are again summarized with bootstrap percentile intervals.

## Main Results

### Unconditional crowding effect

The fitted high-minus-low crowding premium is positive for both groups:

| Group | `phi = 10^-3` | 95% CI | `phi = 10^-2` | 95% CI |
|---|---:|---|---:|---|
| Client | 0.004958 | [0.004484, 0.005794] | 0.011348 | [0.009676, 0.012846] |
| Proprietary | 0.001860 | [0.001552, 0.002132] | 0.005920 | [0.005022, 0.006845] |

This means that impact rises with crowding for both client and proprietary
metaorders.

### Client vs proprietary comparison

The crowding premium is larger for clients than for proprietary flow:

| Benchmark size | Proprietary minus client | 95% CI |
|---|---:|---|
| `phi = 10^-3` | -0.003098 | [-0.003770, -0.002715] |
| `phi = 10^-2` | -0.005428 | [-0.006443, -0.004236] |

So this specification does not support the claim that proprietary metaorders
are more crowding-sensitive in the pooled sample. It supports the opposite
statement: the unconditional crowding premium is larger for clients.

### Participation-rate robustness

The `eta`-split results are mixed.

- Proprietary flow remains strongly positive in the low-`eta` subsample.
- At high `eta`, both groups are negative at `phi = 10^-3` and positive at
  `phi = 10^-2`.
- The client low-`eta` bootstrap is unstable, so those subsample estimates
  should be treated cautiously.

### Joint-bin regression

The additional joint-bin regression gives the following `|imb|` slopes:

| Group | `beta_c` on `|imb|` | 95% CI | Interpretation |
|---|---:|---|---|
| Client | 0.0845 | [-0.0484, 0.2710] | Positive estimate, not statistically significant |
| Proprietary | 0.0768 | [0.0249, 0.1341] | Positive and statistically significant |

The between-group difference in the `|imb|` slope is:

| Difference | Estimate | 95% CI |
|---|---:|---|
| `beta_c^(prop) - beta_c^(client)` | -0.0078 | [-0.1920, 0.1249] |

So the correct interpretation is:

- the proprietary `|imb|` slope is positive and statistically significant
- the client `|imb|` slope is not statistically significant
- the proprietary slope is not significantly larger than the client slope

This distinction matters. A coefficient being significant in one group and not
significant in another does not imply that the two group coefficients are
significantly different from each other.

## Bottom Line

The current crowding-impact workflow supports four main conclusions.

1. Impact increases with crowding for both client and proprietary metaorders.
2. In the unconditional pooled-stock-day analysis, the crowding premium is
   larger for clients than for proprietary flow.
3. The participation-controlled evidence is more mixed and less stable.
4. In the joint-bin regression, the residual `|imb|` slope is significant for
   proprietary flow, not significant for client flow, and not significantly
   different across groups.

## Key Output Files

The main tables written by the workflow are:

- `out_files/ftsemib/crowding_impact/predicted_impacts_main.csv`
- `out_files/ftsemib/crowding_impact/group_difference_main.csv`
- `out_files/ftsemib/crowding_impact/monotonic_contrasts_main.csv`
- `out_files/ftsemib/crowding_impact/joint_regression_fit_summary.csv`
- `out_files/ftsemib/crowding_impact/joint_regression_bootstrap_coefficients.csv`
- `out_files/ftsemib/crowding_impact/joint_regression_group_difference.csv`

The main figures are:

- `images/ftsemib/crowding_impact/html/main_crowding_impact_curves.html`
- `images/ftsemib/crowding_impact/html/predicted_impact_by_crowding_quantile.html`
- `images/ftsemib/crowding_impact/html/crowding_gap_differences.html`
- `images/ftsemib/crowding_impact/html/joint_bin_regression_coefficients.html`
