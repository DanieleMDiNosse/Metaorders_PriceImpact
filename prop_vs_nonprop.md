# Proprietary vs Non‑Proprietary Crowding in `prop_vs_nonprop.py`

This document explains the mathematics and implementation of `prop_vs_nonprop.py`, and summarizes the empirical results obtained on the current dataset (`out_files/metaorders_info_sameday_filtered_member_proprietary.parquet` and `out_files/metaorders_info_sameday_filtered_member_non_proprietary.parquet`). The focus is on:

- how **local** and **environmental** imbalances are defined,
- how correlations and confidence intervals are computed,
- how crowding is decomposed into **within‑group**, **cross‑group**, and **all‑vs‑all** effects,
- and how daily metaorder **count imbalances** are constructed.

Throughout, metaorders are already precomputed objects coming from the metaorder pipeline (see `metaorder_computation.py` and `metaorder_distributions.md`). Here we only use their summary columns.

---

## 1. Inputs and Notation

The function `load_metaorders` (`prop_vs_nonprop.py:752`) loads a metaorder file (`.parquet`, `.csv` or `.pkl`) and ensures the presence of:

- `ISIN`: stock identifier;
- `Date`: trading date (derived from `Period` if needed);
- `Q`: metaorder volume (shares);
- `Direction`: metaorder sign \(D_i \in \{+1,-1\}\), buy vs sell.

Additional columns (if present) are used for descriptive statistics:

- `Q/V`: relative size of the metaorder as a fraction of daily traded volume;
- `Participation Rate`: volume share during the metaorder execution window.

Two datasets are loaded:

- **Proprietary** flow: `metaorders_proprietary` (flag `Group = "prop"`),
- **Non‑proprietary (client)** flow: `metaorders_non_proprietary` (flag `Group = "client"`),

in `main` (`prop_vs_nonprop.py:784–791`).

---

## 2. Local Daily Imbalance (Within‑Group Crowding)

### 2.1 Definition

The core object is the **local daily imbalance** built from *other* metaorders in the same ISIN and date. It is implemented in `add_daily_imbalance` (`prop_vs_nonprop.py:101`).

For each metaorder \(i\) with:

- ISIN \(k\),
- date \(d\),
- volume \(Q_i > 0\),
- direction \(D_i \in \{+1,-1\}\),

consider the set \(\mathcal{G}_{k,d}\) of all metaorders on the same \((k,d)\). Let the signed volume of metaorder \(j\) be
\[
Q_j^{\text{sign}} = Q_j D_j.
\]

The **local imbalance** for metaorder \(i\) is defined as
\[
\text{imbalance}^{\text{local}}_i
  = \frac{\displaystyle \sum_{j \in \mathcal{G}_{k,d}\setminus\{i\}} Q_j D_j}
         {\displaystyle \sum_{j \in \mathcal{G}_{k,d}\setminus\{i\}} Q_j}.
\]

Interpretation:

- \(\text{imbalance}^{\text{local}}_i \approx +1\): other metaorders on that stock/day are mostly **buys**;
- \(\text{imbalance}^{\text{local}}_i \approx -1\): other metaorders are mostly **sells**;
- \(\text{imbalance}^{\text{local}}_i \approx 0\): buy and sell volumes roughly balance.

If there is only one metaorder in \(\mathcal{G}_{k,d}\), the denominator is zero and the imbalance is set to `NaN`. This corresponds to days with no “other” flow and is handled explicitly in `analyze_flow` (`prop_vs_nonprop.py:161–165`).

### 2.2 Implementation sketch

`add_daily_imbalance` groups metaorders by `(ISIN, Date)` and uses vectorized operations:

- computes a helper column `__QD__ = Q * Direction`,
- aggregates per group: `total_Q = sum(Q)`, `total_QD = sum(__QD__)`,
- for each row, subtracts its own contribution to obtain “others only”:
  \[
  \text{numer}_i = \text{total\_QD}_{k,d} - Q_i D_i, \qquad
  \text{denom}_i = \text{total\_Q}_{k,d} - Q_i.
  \]
- sets
  \[
  \text{imbalance}^{\text{local}}_i =
    \begin{cases}
      \text{numer}_i / \text{denom}_i, & \text{if } \text{denom}_i > 0,\\[0.2em]
      \text{NaN}, & \text{otherwise},
    \end{cases}
  \]
  as in `prop_vs_nonprop.py:121–128`.

This is applied separately to proprietary and client metaorders in `main` (`prop_vs_nonprop.py:793–795`).

---

## 3. Correlation and Confidence Intervals

### 3.1 Pearson correlation

The function `corr_with_ci` (`prop_vs_nonprop.py:37`) computes the **Pearson correlation** between two vectors \(x\) and \(y\):
\[
r = \text{Corr}(x, y)
  = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}
         {\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2}
          \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}.
\]

In this script, the main application is:

- \(x_i = D_i\) (direction),
- \(y_i = \text{imbalance}^{\text{local}}_i\) or some variant of imbalance.

### 3.2 Fisher’s z‑transform and CI

For a sample correlation \(r\) computed from \(n\) observations, `corr_with_ci` uses Fisher’s z‑transform to build a \((1-\alpha)\) confidence interval:

1. Clip extreme correlations to avoid numerical issues:
   \[
   r_{\text{clip}} = \min(\max(r, -0.999999), 0.999999).
   \]
2. Compute
   \[
   z = \tfrac12 \log\left(\frac{1 + r_{\text{clip}}}{1 - r_{\text{clip}}}\right).
   \]
3. The standard error of \(z\) is
   \[
   \text{SE}_z = \frac{1}{\sqrt{n - 3}}.
   \]
4. For a two‑sided interval with significance level \(\alpha\), define
   \[
   z_{\alpha/2} = \Phi^{-1}\bigl(1-\tfrac{\alpha}{2}\bigr),
   \]
   where \(\Phi^{-1}\) is the quantile function of the standard normal (`scipy.stats.norm.ppf` in `prop_vs_nonprop.py:79–80`).
5. Then the CI in z‑space is
   \[
   [z_{\text{lo}}, z_{\text{hi}}]
     = \bigl[z - z_{\alpha/2} \,\text{SE}_z,\;
             z + z_{\alpha/2} \,\text{SE}_z\bigr].
   \]
6. Transform back to correlations using the hyperbolic tangent:
   \[
   r_{\text{lo}} = \tanh(z_{\text{lo}}), \qquad
   r_{\text{hi}} = \tanh(z_{\text{hi}}).
   \]

The function returns \((r, r_{\text{lo}}, r_{\text{hi}}, n)\).

---

## 4. Global Imbalance Sanity Check

In `analyze_flow` (`prop_vs_nonprop.py:208–225`), the script also computes a **global imbalance** correlation as a sanity check.

Let:

- \(Q_i\) and \(D_i\) be the volume and direction of metaorder \(i\),
- \(\sum_j Q_j = Q_{\text{tot}}\),
- \(\sum_j Q_j D_j = QD_{\text{tot}}\).

For each metaorder \(i\), define the imbalance of *all other* metaorders in the full sample (not restricted to same day or ISIN) as
\[
\text{imbalance}^{\text{global}}_i
  = \frac{QD_{\text{tot}} - Q_i D_i}{Q_{\text{tot}} - Q_i}.
\]

The script computes the correlation
\[
r^{\text{global}}
  = \text{Corr}\bigl(D_i,\ \text{imbalance}^{\text{global}}_i\bigr),
\]
again with Fisher‑based CIs. By construction, even if directions \(D_i\) were random, this correlation tends to be **negative** because each metaorder is subtracted from the total when forming its own “others” imbalance (see the explanatory printout in `prop_vs_nonprop.py:224–226`).

---

## 5. Daily Crowding Time Series (Within Group)

The function `daily_crowding_ts` (`prop_vs_nonprop.py:233`) computes **daily correlations** between direction and an imbalance column.

For each calendar date \(d\), it groups metaorders and evaluates
\[
r_d = \text{Corr}\bigl(D_i,\ \text{imbalance}_i\bigr),
\]
using only rows with that `Date`. For each date it also stores \((r_d, r_{d,\text{lo}}, r_{d,\text{hi}}, n_d)\).

In `run_daily_crowding_analysis` (`prop_vs_nonprop.py:406`), this is applied separately to:

- proprietary metaorders (`metaorders_proprietary`),
- client metaorders (`metaorders_non_proprietary`),

using local imbalance as the imbalance column. Days with fewer than `min_n` metaorders are dropped. When plotting is enabled, `plot_daily_crowding` (`prop_vs_nonprop.py:297`) generates:

**Figures (within‑group crowding):**

- Daily correlations with confidence bands  
  ![Daily crowding: Corr(Direction, daily imbalance_local)](images/prop_vs_nonprop/daily_crowding_daily_corr.png)
- Smoothed daily correlations (5‑day rolling mean)  
  ![Smoothed daily crowding (5‑day rolling mean)](images/prop_vs_nonprop/daily_crowding_rolling_5d.png)

These figures show how crowding changes over time for proprietary vs client flow.

---

## 6. Cross‑Group Crowding: Prop vs Client Environments

### 6.1 Environment imbalance

To study **cross‑group crowding**, the script first computes, for each group, an **environment imbalance** built from the *other* group only.

`compute_environment_imbalance` (`prop_vs_nonprop.py:255`) takes a source DataFrame (e.g. client metaorders) and for each \((\text{ISIN}, \text{Date})\) computes:
\[
\text{imbalance}^{\text{env}}_{k,d}
  = \frac{\sum_{j \in \mathcal{G}^{\text{src}}_{k,d}} Q_j D_j}
         {\sum_{j \in \mathcal{G}^{\text{src}}_{k,d}} Q_j},
\]
where \(\mathcal{G}^{\text{src}}_{k,d}\) is the set of source‑group metaorders on stock \(k\) and date \(d\). This produces a per‑day, per‑ISIN imbalance for the environment group.

`attach_environment_imbalance` (`prop_vs_nonprop.py:278`) then merges this quantity into a **target** DataFrame so that each target metaorder \(i\) inherits the environment imbalance on its own \((k,d)\).

Concretely, in `run_cross_group_crowding_analysis` (`prop_vs_nonprop.py:461`):

- proprietary metaorders obtain `imbalance_client_env` built from client flow;
- client metaorders obtain `imbalance_prop_env` built from proprietary flow.

### 6.2 Cross‑group daily correlations

The script then computes daily correlations using `daily_crowding_ts` on these environment columns:

- for prop: \(r_d^{\text{prop|client}} = \text{Corr}(D_i^{\text{prop}}, \text{imbalance}^{\text{client env}}_i)\),
- for client: \(r_d^{\text{client|prop}} = \text{Corr}(D_i^{\text{client}}, \text{imbalance}^{\text{prop env}}_i)\).

Filtered daily series (with `n_d \ge \text{min_n}`) are summarized and, when plotting is enabled, `plot_daily_crowding` produces:

**Figures (cross‑group crowding):**

- Daily correlations prop vs client and client vs prop  
  ![Cross‑group crowding: Corr(Direction, other‑group imbalance)](images/prop_vs_nonprop/cross_crowding_daily_corr.png)
- Smoothed cross‑group crowding (5‑day rolling mean)  
  ![Smoothed cross‑group crowding (5‑day rolling mean)](images/prop_vs_nonprop/cross_crowding_rolling_5d.png)

These figures quantify whether one group tends to trade **with** or **against** the imbalance created by the other group.

---

## 7. Crowding vs All Others (Combined Prop + Client)

In `run_all_vs_all_crowding_analysis` (`prop_vs_nonprop.py:532`), the two groups are concatenated into a single DataFrame:
\[
\text{combined} = \text{proprietary} \cup \text{client}.
\]

`add_daily_imbalance` is then called with a new column name `imbalance_all_others` to compute, for each metaorder \(i\),
\[
\text{imbalance}^{\text{all}}_i
  = \frac{\displaystyle \sum_{j \in \mathcal{G}_{k,d}\setminus\{i\}} Q_j D_j}
         {\displaystyle \sum_{j \in \mathcal{G}_{k,d}\setminus\{i\}} Q_j},
\]
where “others” now include **both** proprietary and client metaorders.

The combined DataFrame is then split back into:

- `prop_all`: rows with `Group == "prop"`,
- `client_all`: rows with `Group == "client"`,

and daily correlations are computed separately using `daily_crowding_ts` with `imbalance_all_others`. After filtering by `min_n`, summary statistics and (optionally) plots are produced:

**Figures (crowding vs all others):**

- Daily correlations vs all other metaorders  
  ![Crowding versus all other metaorders](images/prop_vs_nonprop/all_vs_all_crowding_daily_corr.png)
- Smoothed crowding vs all others (5‑day rolling mean)  
  ![Smoothed crowding vs all others (5‑day rolling mean)](images/prop_vs_nonprop/all_vs_all_crowding_rolling_5d.png)

These series measure how each group’s trade directions align with the **overall** daily metaorder imbalance generated by both groups.

---

## 8. Daily Metaorder Count Imbalance

The script also compares how many metaorders each group executes per day, independently of size and sign.

`compute_daily_metaorder_counts` (`prop_vs_nonprop.py:602`) builds a DataFrame indexed by `Date` with:

- `Proprietary(d)`: number of proprietary metaorders on day \(d\),
- `Client(d)`: number of client metaorders on day \(d\).

For each day, it defines the **count imbalance**
\[
\text{imbalance\_counts}(d)
  = \frac{\text{Proprietary}(d) - \text{Client}(d)}
         {\text{Proprietary}(d) + \text{Client}(d)}.
\]

By construction:

- values near \(+1\) mean almost all metaorders are proprietary,
- values near \(-1\) mean almost all are client,
- values near \(0\) mean similar counts in both groups.

`run_daily_count_imbalance_analysis` (`prop_vs_nonprop.py:602–647`) prints summary statistics and, when plots are enabled, `plot_daily_count_imbalance` saves:

**Figures (daily count imbalance):**

- Time series of daily count imbalance  
  ![Daily metaorder count imbalance timeseries](images/prop_vs_nonprop/daily_counts_imbalance_timeseries.png)
- Distribution of daily count imbalance  
  ![Histogram of daily metaorder count imbalance](images/prop_vs_nonprop/daily_counts_imbalance_histogram.png)

---

## 9. Empirical Results on the Current Dataset

Running

```bash
python prop_vs_nonprop.py --no-plots
```

on the current dataset yields the following key findings.

### 9.1 Sample sizes and basic stats

- **Proprietary metaorders**
  - Total metaorders: \(1{,}507{,}398\)  
  - Unique ISINs: 41; trading days: 251  
  - Relative size range \(Q/V\): \(2.16\times 10^{-8}\) to \(3.19\times 10^{-1}\)  
  - Participation rate: mean \(0.270\), standard deviation \(0.247\)

- **Client metaorders**
  - Total metaorders: \(678{,}106\)  
  - Unique ISINs: 41; trading days: 251  
  - Relative size range \(Q/V\): \(1.00\times 10^{-5}\) to \(5.34\times 10^{-1}\)  
  - Participation rate: mean \(0.193\), standard deviation \(0.229\)

The share of `NaN` local imbalance is effectively zero for both groups (days with a single metaorder in an ISIN are extremely rare).

### 9.2 Local crowding: Corr(Direction, local imbalance)

- **Proprietary flow**
  - \(r = \text{Corr}(D, \text{imbalance}^{\text{local}}) \approx 0.045\)  
    95% CI: \([0.043, 0.046]\); \(n = 1{,}507{,}398\).
  - Mean local imbalance conditional on direction:
    - buys: \(\mathbb{E}[\text{imbalance}^{\text{local}} \mid D=+1] \approx -0.027\),
    - sells: \(\mathbb{E}[\text{imbalance}^{\text{local}} \mid D=-1] \approx -0.046\).
  - Interpretation in the script: **roughly uncorrelated** with daily imbalance (idiosyncratic flow).

- **Client flow**
  - \(r \approx 0.082\)  
    95% CI: \([0.080, 0.084]\); \(n = 678{,}105\).
  - Mean local imbalance conditional on direction:
    - buys: \(\approx +0.024\),
    - sells: \(\approx -0.037\).
  - Interpreted as: trades tend to go **with** the daily metaorder imbalance (**crowding / herding**).

For comparison, the **global imbalance** correlations (mechanically negative) are:

- proprietary: \(r^{\text{global}} \approx -0.185\) (95% CI \([-0.186, -0.183]\)),
- client: \(r^{\text{global}} \approx -0.141\) (95% CI \([-0.144, -0.139]\)).

### 9.3 Daily correlations (within group)

Using `run_daily_crowding_analysis` with `min_n = 100`:

- **Proprietary**
  - All 251 days satisfy \(n_d \ge 100\).
  - Mean daily correlation (unfiltered and filtered): \(\bar{r}_d^{\text{prop}} \approx 0.032\).

- **Client**
  - Again, 251 days with \(n_d \ge 100\).
  - Mean daily correlation: \(\bar{r}_d^{\text{client}} \approx 0.073\).

Thus, on average, client flow shows stronger within‑group crowding than proprietary flow.

### 9.4 Cross‑group crowding

From `run_cross_group_crowding_analysis`:

- **Prop vs client environment**
  - Days with \(n_d \ge 100\): 251 (out of 251).
  - Mean daily correlation (unfiltered and filtered):  
    \(\bar{r}_d^{\text{prop|client}} \approx -0.015\).

- **Client vs prop environment**
  - Same number of days: 251.
  - Mean daily correlation:  
    \(\bar{r}_d^{\text{client|prop}} \approx -0.011\).

Both numbers are slightly **negative**, indicating a mild tendency for each group to trade **against** the imbalance produced by the other group.

### 9.5 Crowding vs all others

From `run_all_vs_all_crowding_analysis`:

- **Proprietary vs all others**
  - All days pass the \(n_d \ge 100\) filter.
  - Mean daily correlation: \(\bar{r}_d^{\text{prop|all}} \approx 0.011\).

- **Client vs all others**
  - Mean daily correlation: \(\bar{r}_d^{\text{client|all}} \approx 0.041\).

Client flow again shows stronger alignment with the aggregate daily imbalance than proprietary flow.

### 9.6 Daily metaorder count imbalance

`run_daily_count_imbalance_analysis` reports the first rows of the daily counts, e.g.

| Date       | Proprietary | Client | imbalance\_counts |
|-----------:|------------:|-------:|------------------:|
| 2024‑06‑03 | 5537        | 2946   | 0.305             |
| 2024‑06‑04 | 6495        | 2850   | 0.390             |
| 2024‑06‑05 | 5207        | 2593   | 0.335             |
| 2024‑06‑06 | 5482        | 2812   | 0.322             |
| 2024‑06‑07 | 5540        | 2182   | 0.435             |

Over the full sample:

- Mean daily count imbalance: \(\mathbb{E}[\text{imbalance\_counts}] \approx +0.385\),
- Standard deviation: \(\approx 0.050\).

This indicates that, on a typical day, proprietary metaorders are substantially more numerous than client metaorders.

---

## 10. Summary

- `prop_vs_nonprop.py` constructs **local** and **environmental** volume‑weighted imbalances of metaorders and studies how trade directions correlate with these imbalances at both the global and daily level.
- Mathematically, local imbalance is a self‑excluded signed‑volume average over other metaorders on the same ISIN and day; cross‑group and all‑vs‑all analyses reuse the same structure with different environments.
- **Empirically, client flow exhibits stronger positive crowding (trading with the daily imbalance) than proprietary flow, while both groups slightly lean against the imbalance created by the other group.**
- **Daily count imbalances reveal that proprietary metaorders are both more numerous and, on average, less strongly crowded than client metaorders.**

The accompanying figures under `images/prop_vs_nonprop` visualize these effects over time and across groups.
