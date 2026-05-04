# Codex Task: Create a Python script to analyze impact conditioned on crowding

## Goal

Create a Python analysis script that tests whether market impact increases with crowding, separately for **client** and **proprietary** metaorders, and whether that increase is **stronger for proprietary** metaorders.

Do **not** assume any repository structure. Discover and reuse the repo's existing data-loading, utilities, plotting, and analysis code where possible.

## Context

The paper already defines:

- instantaneous impact in volatility units: \(I_i = \varepsilon_i \Delta p_i / \sigma_{d(i)}\)
- relative size: \(\phi_i = Q_i / V_{d(i)}\)
- participation rate: \(\eta_i = Q_i / V_i^{window}\)
- within-group leave-one-out imbalance on the same stock-day
- day-cluster bootstrap for crowding analyses

The new script should mirror that framework, but condition impact on **crowding quantiles** instead of participation quantiles.

## Main analysis design

### 1. Construct the working metaorder table

Build or load a metaorder-level dataframe containing at least:

- unique metaorder id
- date
- instrument / ISIN
- capacity group: proprietary vs client
- sign \(\varepsilon_i\)
- total size \(Q_i\)
- relative size \(\phi_i\)
- instantaneous impact \(I_i\)
- participation rate \(\eta_i\)
- any fields required to compute imbalance if not already stored

If imbalance variables are already available in repo-native outputs, reuse them. Otherwise recompute them exactly from the metaorder table.

### 2. Apply the same filters as the impact section

As closely as possible, mirror the existing impact-analysis filters:

- finite \(I_i\), \(\phi_i\), \(\eta_i\)
- minimum size filter: \(\phi_i > 10^{-5}\)
- exclude \(\eta_i \ge 1\)
- exclude observations where crowding cannot be computed because there are no other same-group metaorders on that stock-day

Print sample sizes before and after filtering.

## Do not use raw imbalance as the main crowding variable

For the **main result**, define **aligned within-group crowding** as:

\[
c_i = \varepsilon_i \cdot imb_i
\]

where \(imb_i\) is the **within-group leave-one-out volume-weighted imbalance** on the same stock-day.

Interpretation:

- high \(c_i\): metaorder trades in the same direction as surrounding same-group flow
- low \(c_i\): metaorder trades against surrounding same-group flow

This is the economically relevant crowding measure.

Also compute these only as robustness checks if convenient:

- \(|imb_i|\)
- cross-group aligned imbalance
- all-vs-all aligned imbalance

But the primary analysis must use **within-group aligned crowding**.

## Quantile split

For **client** and **proprietary** separately:

1. compute \(c_i\)
2. split observations into **3 quantiles** of \(c_i\): low, mid, high crowding
3. store quantile cutpoints and sample sizes

Use **group-specific quantiles** for the main specification.

Optional robustness:

- pooled quantile edges across groups
- 4 quantiles instead of 3

## Estimate impact curves within each crowding quantile

Within each `(group, crowding_quantile)` subset, estimate the same power-law impact curve used in the paper:

\[
E[I_i \mid \phi_i, q] = Y_q \phi_i^{\gamma_q}
\]

using the same log-binned weighted least squares procedure already used elsewhere in the repo.

### Required fitting procedure

1. log-bin \(\phi_i\)
2. for each bin compute:
   - bin center \(\phi_k\)
   - mean impact \(\bar I_k\)
   - SEM
   - count
3. discard bins with too few observations or non-positive / non-finite mean impact
4. fit in log-space with weights proportional to:

\[
w_k \approx \left(\frac{\bar I_k}{SEM_k}\right)^2
\]

For each fitted curve, report:

- \(Y_q\)
- \(\gamma_q\)
- standard errors
- number of metaorders
- number of retained bins
- goodness-of-fit diagnostics if easy to obtain

## Key comparison: do not rely only on \(Y\) or \(\gamma\)

Because conditional curves may cross, do **not** compare groups using only fitted coefficients.

Also compute predicted impact at common benchmark sizes, for example:

- \(\phi^*_1 = 10^{-3}\)
- \(\phi^*_2 = 10^{-2}\)

For each group \(G\), compute:

\[
\Delta_G(\phi^*) = \hat I_G(\phi^*, high) - \hat I_G(\phi^*, low)
\]

This is the main summary of the crowding effect.

Then evaluate:

1. whether \(\Delta_G(\phi^*) > 0\) for both groups
2. whether \(\Delta_{prop}(\phi^*) > \Delta_{client}(\phi^*)\)

## Inference

Use a **day-cluster bootstrap** throughout, consistent with the paper's methodology.

Bootstrap by resampling trading days with replacement and recomputing the full analysis.

Make the number of bootstrap iterations configurable from CLI.

### Bootstrap outputs

For each group and crowding quantile, compute bootstrap confidence intervals for:

- \(Y_q\)
- \(\gamma_q\)
- predicted impact at each benchmark \(\phi^*\)
- contrasts such as high-minus-low and mid-minus-low

Also bootstrap the between-group difference:

\[
\Delta_{prop}(\phi^*) - \Delta_{client}(\phi^*)
\]

Use percentile intervals.

## Very important robustness: control for participation

The paper already shows crowding rises with participation, especially for clients. Therefore a simple crowding split may partly reflect aggressiveness rather than a pure crowding effect.

So include at least one robustness analysis controlling for \(\eta\).

### Preferred robustness

Repeat the crowding-quantile analysis **within broad participation bins**, for example:

- low \(\eta\) vs high \(\eta\)
- or terciles of \(\eta\)

That is, estimate curves within cells like:

- `(group, eta_bin, crowding_bin)`

If data get sparse, reduce the number of bins.

### Minimal fallback

If the cell approach is too sparse, run a secondary model that controls jointly for \(\phi\), crowding, and \(\eta\), but keep the quantile-based analysis as the primary reported result.

## Outputs to generate

Save publication-ready outputs, not just console text.

### Tables

Create CSV files for at least:

1. sample sizes by group and crowding quantile
2. crowding cutpoints
3. fitted \(Y_q\), \(\gamma_q\), standard errors
4. predicted impacts at benchmark sizes
5. bootstrap confidence intervals
6. monotonic contrasts across quantiles
7. proprietary-minus-client difference in crowding effect

### Figures

Create at least these figures:

1. **Main figure:** impact curves overlaid by crowding quantile, one panel for client and one for proprietary
2. **Summary figure:** predicted impact at benchmark sizes vs crowding quantile, separate panels by group
3. **Difference figure:** high-minus-low crowding impact gap for client vs proprietary, with bootstrap confidence intervals
4. **Robustness figure:** same analysis within low/high participation bins

Reuse existing plotting conventions from the repo if available.

## Acceptance criteria

The script is complete only if it clearly answers all of the following:

1. Does impact increase with aligned crowding within client flow?
2. Does impact increase with aligned crowding within proprietary flow?
3. Is the increase monotonic across low / mid / high crowding?
4. Is the increase larger for proprietary than for client?
5. Does the conclusion survive conditioning on participation?

If the evidence is weak or mixed, report that honestly rather than forcing a positive result.

## Guardrails

Avoid these mistakes:

- do **not** use raw imbalance instead of aligned crowding for the main result
- do **not** include the metaorder itself in the imbalance calculation
- do **not** pool proprietary and client flow for the main crowding quantiles
- do **not** compare only exponents; also compare predicted impact at common benchmark sizes
- do **not** ignore participation as a confounder
- do **not** silently drop too many observations
- do **not** hardcode repository paths or assume project layout

## Optional extension

If the repo already has dynamic impact path machinery, add an optional extension:

- split metaorders by aligned crowding quantiles
- compute mean normalized impact paths by crowding bucket
- compare both peak impact and persistence, e.g. post-trade retention ratios

This is a secondary extension. The primary deliverable is the end-of-execution impact analysis.

## Deliverable requirements

Produce:

- one standalone analysis script
- helper functions only if needed
- clean CLI arguments for input / output / bootstrap settings
- reproducible outputs with fixed random seed
- concise console logging
- robust behavior even if repo internals differ from expectations

Prefer reuse of existing repo loaders and analysis helpers over reimplementing project-specific logic from scratch.
