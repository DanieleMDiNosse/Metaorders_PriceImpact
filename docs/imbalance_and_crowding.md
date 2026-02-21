# Proprietary vs Non-Proprietary Crowding in `scripts/crowding_analysis.py`

`scripts/crowding_analysis.py` compares proprietary and client flow after metaorders have been built by `scripts/metaorder_computation.py`. It attaches imbalance metrics, optional daily returns, and produces crowding diagnostics and plots.

The imbalance study is designed to answer several questions:

- Do metaorders tend to go **with** (herding) or **against** (contrarian/liquidity-providing) the prevailing signed volume imbalance on the same stock and day?
- How does this behavior differ between **proprietary** flow and **client** flow?
- How crowded is each group:
  - relative to its **own** daily flow (within-group crowding),
  - relative to the **other** group’s flow (cross-group crowding),
  - relative to the union of **all other** metaorders (prop+client combined),
  - and, for members, relative to the flow of their **own clients** (member-level crowding)?

All these questions are expressed as correlations between **metaorder direction** and suitably defined **signed volume imbalance** measures.

## Latest run results (ftsemib, `pipeline_20260220_190521`)

This section reports the outputs of the most recent `run_all_pipelines.sh` bundle:

- log: `out_files/ftsemib/logs/pipeline_20260220_190521/crowding_analysis.log`
- figures (PNG + Plotly HTML): `images/ftsemib/prop_vs_nonprop/png/` and `images/ftsemib/prop_vs_nonprop/html/`
- crowding vs participation rate $\eta$: tables in `out_files/ftsemib/crowding_vs_part_rate/` and curves in `images/ftsemib/crowding_vs_part_rate/png/` (HTML in `.../html/`, provenance in `out_files/ftsemib/crowding_vs_part_rate/run_manifest.json`)

**Window:** 2024-06-03 to 2025-05-30 (251 trading days).

### Numerical summary

**Within-group (local, ISIN–Date).**

| Group | Metaorders | $Q/V$ range | $\eta$ mean ± std | $\\mathrm{Corr}(\\epsilon, \\mathrm{imb}_{\\mathrm{local}})$ (95% CI) |
|---|---:|---:|---:|---:|
| Proprietary | 588,334 | $10^{-5}$ – 0.272 | 0.136 ± 0.131 | 0.108 [0.102, 0.113] |
| Client (non-proprietary) | 255,535 | $10^{-5}$ – 0.353 | 0.085 ± 0.099 | 0.187 [0.180, 0.196] |

**Daily mean correlation (n ≥ 100 each day; 251/251 days for both groups).**

| Metric | Proprietary | Client (non-proprietary) |
|---|---:|---:|
| Mean daily $\\mathrm{Corr}(\\epsilon, \\mathrm{imb}_{\\mathrm{local}})$ (95% CI) | 0.081 [0.077, 0.086] | 0.169 [0.162, 0.177] |

**Cross-group (ISIN–Date environment).**

| Metric | Value (95% CI) |
|---|---:|
| Global $\\mathrm{Corr}(\\epsilon_{\\mathrm{prop}}, \\mathrm{imb}_{\\mathrm{client\\ env}})$ | -0.023 [-0.028, -0.018] |
| Global $\\mathrm{Corr}(\\epsilon_{\\mathrm{client}}, \\mathrm{imb}_{\\mathrm{prop\\ env}})$ | -0.016 [-0.025, -0.007] |
| Mean daily $\\mathrm{Corr}(\\epsilon_{\\mathrm{prop}}, \\mathrm{imb}_{\\mathrm{client\\ env}})$ (95% CI) | -0.018 [-0.023, -0.014] |
| Mean daily $\\mathrm{Corr}(\\epsilon_{\\mathrm{client}}, \\mathrm{imb}_{\\mathrm{prop\\ env}})$ (95% CI) | -0.003 [-0.010, 0.004] |

**All-vs-all (each group vs all other metaorders).**

| Metric (daily, n ≥ 100) | Proprietary | Client (non-proprietary) |
|---|---:|---:|
| Mean daily $\\mathrm{Corr}(\\epsilon, \\mathrm{imb}_{\\mathrm{all\\ others}})$ | 0.041 | 0.113 |

**Member-level (prop direction vs member-day client imbalance).**

- Global: $r = 0.011$ (p = 0.014, 95% CI [0.001, 0.022], n = 37,278).
- Per-member (threshold `MIN_METAORDERS_PER_MEMBER=30` on both prop and client totals): 10/45 members pass; mean per-member $r = 0.003$.
- Member-window heatmap (`MEMBER_WINDOW_DAYS=3`, threshold `N_MIN_PER_MEMBER_CLIENT=5` on both prop and client counts per member-window; `BOOTSTRAP_HEATMAP=false` so no significance filtering):
  - windows: 84
  - member-window groups in prop data: 2,124
  - count-qualified cells: 205
  - finite-correlation cells: 188 (7 members have ≥1 finite cell)
  - mean correlation across finite cells: -0.028 (range [-1, +1])
  - `n_used` (finite cells): p10=5, median=11, p90≈544

**Daily count imbalance (prop vs client counts).**

- Imbalance definition: $(N_{\\mathrm{prop}} - N_{\\mathrm{client}}) / (N_{\\mathrm{prop}} + N_{\\mathrm{client}})$.
- Mean ± std: +0.403 ± 0.060.

### Plots (PNG)

**Within-group crowding.** (HTML: `../images/ftsemib/prop_vs_nonprop/html/daily_crowding_daily_corr.html`, `../images/ftsemib/prop_vs_nonprop/html/daily_crowding_rolling_3d.html`)

| Daily correlation | Rolling (3-day mean) |
|---|---|
| ![Within-group crowding, daily](../images/ftsemib/prop_vs_nonprop/png/daily_crowding_daily_corr.png) | ![Within-group crowding, rolling](../images/ftsemib/prop_vs_nonprop/png/daily_crowding_rolling_3d.png) |

**Cross-group crowding.** (HTML: `../images/ftsemib/prop_vs_nonprop/html/cross_crowding_daily_corr.html`, `../images/ftsemib/prop_vs_nonprop/html/cross_crowding_rolling_3d.html`)

| Daily correlation | Rolling (3-day mean) |
|---|---|
| ![Cross-group crowding, daily](../images/ftsemib/prop_vs_nonprop/png/cross_crowding_daily_corr.png) | ![Cross-group crowding, rolling](../images/ftsemib/prop_vs_nonprop/png/cross_crowding_rolling_3d.png) |

**All-vs-all crowding.** (HTML: `../images/ftsemib/prop_vs_nonprop/html/all_vs_all_crowding_daily_corr.html`, `../images/ftsemib/prop_vs_nonprop/html/all_vs_all_crowding_rolling_3d.html`)

| Daily correlation | Rolling (3-day mean) |
|---|---|
| ![All-vs-all crowding, daily](../images/ftsemib/prop_vs_nonprop/png/all_vs_all_crowding_daily_corr.png) | ![All-vs-all crowding, rolling](../images/ftsemib/prop_vs_nonprop/png/all_vs_all_crowding_rolling_3d.png) |

**Imbalance diagnostics.** (HTML: `../images/ftsemib/prop_vs_nonprop/html/imbalance_distribution.html`, `../images/ftsemib/prop_vs_nonprop/html/imbalance_vs_daily_returns.html`)

| Imbalance distribution | Imbalance vs daily returns |
|---|---|
| ![Imbalance distribution](../images/ftsemib/prop_vs_nonprop/png/imbalance_distribution.png) | ![Imbalance vs daily returns](../images/ftsemib/prop_vs_nonprop/png/imbalance_vs_daily_returns.png) |

**Daily count imbalance.** (HTML: `../images/ftsemib/prop_vs_nonprop/html/daily_counts_imbalance_timeseries.html`, `../images/ftsemib/prop_vs_nonprop/html/daily_counts_imbalance_histogram.html`)

| Time series | Histogram |
|---|---|
| ![Daily count imbalance, time series](../images/ftsemib/prop_vs_nonprop/png/daily_counts_imbalance_timeseries.png) | ![Daily count imbalance, histogram](../images/ftsemib/prop_vs_nonprop/png/daily_counts_imbalance_histogram.png) |

**Member-level crowding.** (HTML: `../images/ftsemib/prop_vs_nonprop/html/member_prop_client_crowding_hist.html`, `../images/ftsemib/prop_vs_nonprop/html/member_prop_client_crowding_heatmap_3d.html`)

| Per-member correlations | Member × window heatmap |
|---|---|
| ![Member prop-client crowding per member](../images/ftsemib/prop_vs_nonprop/png/member_prop_client_crowding_hist.png) | ![Member prop-client crowding heatmap](../images/ftsemib/prop_vs_nonprop/png/member_prop_client_crowding_heatmap_3d.png) |

## Inputs and preprocessing
- Expects per-metaorder parquet files (defaults in the script):
  - `out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_member_proprietary.parquet`
  - `out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_member_non_proprietary.parquet`
- Adds `Group` (`"prop"` or `"client"`) and ensures a `Date` column (derived from `Period` if missing).
- If `RUN_METAORDER_DICT_STATS=True`, run `scripts/metaorder_statistics.py` separately to rebuild distribution plots (durations, inter-arrivals, volumes, Q/V, participation, metaorders per member) from `metaorders_dict_all_{LEVEL}_{PROPRIETARY_TAG}.pkl` and parquet tapes (default `data/parquet/`). Outputs land in `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/` and `.../html/`.
- Within-group imbalance is added via `add_daily_imbalance` unless already present.
- Cross-group environment imbalances are added via `attach_environment_imbalance` unless already present (`imbalance_client_env` for prop, `imbalance_prop_env` for client).
- Member-level client environment imbalance is added via `attach_member_client_imbalance` unless already present (`imbalance_client_member_env` on proprietary metaorders).
- If `ATTACH_DAILY_RETURNS=True`, close-to-close daily log returns are computed from per-ISIN parquet trades (using `RETURNS_DATA_DIR` and `RETURNS_TRADING_HOURS`, with `RETURNS_DATA_DIR` typically set to `data/parquet/`) and attached as `DAILY_RETURN_COL`. Parquets are rewritten when new columns are added.

## Core objects and notation

After `scripts/metaorder_computation.py` has run, each row in the proprietary/client parquet files corresponds to a metaorder with fields including:

- `ISIN`: instrument identifier.
- `Date`: trading day.
- `Member`: member/broker executing the metaorder.
- `Client`: client identifier (for client flow).
- `Q`: metaorder volume.
- `Direction ∈ {+1, −1}`: buy (+1) or sell (−1).

The imbalance quantities are always based on **signed volume**:

$$
Q_i \epsilon_i,
$$

aggregated over appropriate subsets of metaorders and normalized by total volume.

For a set of indices $\mathcal{S}$,

$$
\text{imbalance}(\mathcal{S})
  = \frac{\sum_{j \in \mathcal{S}} Q_j \epsilon_j}{\sum_{j \in \mathcal{S}} Q_j},
$$

with the convention that the imbalance is `NaN` if the denominator is zero.

In code, this pattern is implemented via:

- `compute_environment_imbalance(source_df, group_cols, side_col="Direction", vol_col="Q", new_col="imbalance_env")`:
  - Builds a grouped table on `group_cols` (e.g. `("ISIN", "Date")` or `("Member", "Date")`) with:
    - `total_Q = sum(Q_j)`
    - `total_QD = sum(Q_j * Direction_j)`
    - `new_col = total_QD / total_Q` when `total_Q > 0`, else `NaN`.
- `attach_environment_imbalance(target_df, environment_df, new_col, group_cols=("ISIN","Date"), ...)`:
  - Calls `compute_environment_imbalance` on `environment_df` and merges the resulting imbalance column onto `target_df`.

The within-group local imbalance uses a closely related construction but excludes the current metaorder, as detailed next.

## Local daily imbalance (within-group)

For a metaorder $i$ on ISIN $k$ and date $d$, with volume $Q_i$ and direction $\epsilon_i \in \{+1,-1\}$, define
$$
\text{imbalance}^{\text{local}}_i
	  = \frac{\sum_{j \in \mathcal{G}_{k,d}\setminus\{i\}} Q_j \epsilon_j}
	         {\sum_{j \in \mathcal{G}_{k,d}\setminus\{i\}} Q_j},
	$$
where $\mathcal{G}_{k,d}$ are the other metaorders on the same $(k,d)$. Days with a single metaorder yield `NaN` for this field. This is computed separately for proprietary and client flow.

In code, this is handled by:

- `add_daily_imbalance(df, group_cols=("ISIN", "Date"), side_col="Direction", vol_col="Q", new_col="imbalance_local")`:
  - For each `(ISIN, Date)` group:
    - Adds `__QD__ = Q * Direction`.
    - Computes `total_Q` and `total_QD` via `groupby.transform("sum")`.
    - For each row:
      - `denom = total_Q - Q_i`, `numer = total_QD - __QD__i`.
      - `imbalance_local = numer / denom` where `denom > 0`, else `NaN`.

This exactly implements the self-exclusion $\mathcal{G}_{k,d}\setminus\{i\}$.

## Cross-group and all-others imbalance

- **Cross-group environment (ISIN–Date level):**
  - For proprietary metaorders, `imbalance_client_env(ISIN, Date)` is defined as:
    $$
    \frac{\sum_{\text{client metaorders on }(k,d)} Q_j \epsilon_j}
           {\sum_{\text{client metaorders on }(k,d)} Q_j}.
    $$
    Implemented by `attach_environment_imbalance(metaorders_proprietary, metaorders_non_proprietary, new_col="imbalance_client_env")`.
  - For client metaorders, `imbalance_prop_env(ISIN, Date)` is defined analogously using proprietary flow and attached to the client dataframe.

- **All-others imbalance (prop + client combined, self-excluded):**
  - In `run_all_vs_all_crowding_analysis`, proprietary and client metaorders are concatenated, and `add_daily_imbalance` is applied with `group_cols=("ISIN", "Date")` and `new_col="imbalance_all_others"`.
  - For each metaorder $i$, `imbalance_all_others` is the signed volume imbalance of all other metaorders (prop+client) on the same `(ISIN, Date)`, excluding $i$ itself, mirroring the definition of `imbalance_local` but on the combined set.

## Member-level client imbalance (Member–Date level)

The member-level analysis asks whether a member’s proprietary flow tends to align with the aggregate flow of its **own clients**.

For a given member $m$ and date $d$, define $\mathcal{C}_{m,d}$ as the set of **client** metaorders with:
- `Member = m`,
- `Date = d`,
- any ISIN.

The member-level client imbalance is:

$$
\text{imbalance\_client\_member\_env}(m,d)
  = \frac{\sum_{j \in \mathcal{C}_{m,d}} Q_j \epsilon_j}
         {\sum_{j \in \mathcal{C}_{m,d}} Q_j},
$$

with `NaN` if the denominator is zero (no client volume for that member/day).

This is computed by:

- `attach_member_client_imbalance(metaorders_proprietary, metaorders_non_proprietary, new_col="imbalance_client_member_env")`:
  - Calls `compute_environment_imbalance` on the **client** dataframe with `group_cols=("Member", "Date")`.
  - Merges the resulting `(Member, Date) → imbalance_client_member_env` table onto the **proprietary** dataframe keyed by `(Member, Date)`.
  - Each proprietary metaorder $i$ with `(Member_i, Date_i)` receives the same member-level client imbalance value for that member and day.

The member-level crowding analysis is then performed by:

- `run_member_level_prop_client_crowding_analysis(metaorders_proprietary, metaorders_non_proprietary, env_col="imbalance_client_member_env", ...)`:
  - Computes a **global** correlation across all proprietary metaorders:
    $$
    r_{\text{global}} = \text{Corr}\big(\epsilon_i,\ \text{imbalance\_client\_member\_env}(\text{Member}_i, \text{Date}_i)\big)
    $$
    with a **bootstrap percentile CI** (`corr_with_ci`) and a **permutation p-value** (`corr_with_bootstrap_p`).
  - Computes **per-member** correlations:
    $$
    r_m = \text{Corr}\big(\epsilon_i,\ \text{imbalance\_client\_member\_env}(m, \text{Date}_i)\big)
    \quad \text{for metaorders with Member } m,
    $$
    including a **Date-cluster bootstrap percentile CI** (`lo`, `hi`), a permutation p-value `p`, and `n` (sample size).
  - Filters to members with at least `MIN_METAORDERS_PER_MEMBER` **proprietary** metaorders and at least `MIN_METAORDERS_PER_MEMBER` **client** metaorders; exports:
    - `member_prop_client_crowding_per_member.parquet` with one row per member (`Member`, `r`, `lo`, `hi`, `p`, `n`, `n_client_total`).
    - `member_prop_client_crowding_hist.png`, a per-member **bar chart** (x = Member, y = correlation), sorted by correlation.
  - Builds a **`MEMBER_WINDOW_DAYS`-day, non-overlapping window heatmap** (saved to `member_prop_client_crowding_heatmap_{MEMBER_WINDOW_DAYS}d.png`):
    - Dates are binned into consecutive `MEMBER_WINDOW_DAYS`-day blocks (no overlap).
    - For each `(Member, Window)`, compute `Corr(Direction_prop, imbalance_client_member_env)` using proprietary metaorders in that window **only if** there are at least `N_MIN_PER_MEMBER_CLIENT` proprietary metaorders **and** `N_MIN_PER_MEMBER_CLIENT` client metaorders for that member in the same window; otherwise the cell is `NaN`.
    - If `BOOTSTRAP_HEATMAP=true`, the script also computes a permutation p-value per cell and filters non-significant cells (`p > P_VALUE_THRESHOLD`) by setting them to `NaN`. With the current default (`BOOTSTRAP_HEATMAP=false`), **no significance filtering is applied**.
    - Heatmap axes: x = members, y = `MEMBER_WINDOW_DAYS`-day windows; colormap clipped to [-1, +1].

## Inference: correlations, bootstrap confidence intervals, and permutation p-values

### Pearson correlation

Given a sample of pairs $(x_i, y_i)_{i=1}^n$, the script measures linear crowding via the **Pearson correlation**:
$$
r = \frac{\sum_{i=1}^n (x_i - \bar x)(y_i - \bar y)}
         {\sqrt{\sum_{i=1}^n (x_i - \bar x)^2}\,\sqrt{\sum_{i=1}^n (y_i - \bar y)^2}},
$$
where $\bar x$ and $\bar y$ are sample means. In our context:

- $x_i$ is typically `Direction` (±1) of a metaorder.
- $y_i$ is some imbalance measure (`imbalance_local`, `imbalance_client_env`, `imbalance_client_member_env`, etc.).

This $r$ is the point estimate of how strongly metaorder signs co-move with the imbalance proxy.

### Bootstrap confidence intervals (non-parametric)

Two bootstrap schemes appear in `scripts/crowding_analysis.py`:

- **Row bootstrap over pairs $(x_i, y_i)$** (implemented in `corr_with_ci`): used for some global summaries and for the **pointwise per-day CIs** in `daily_crowding_ts`.
- **Date-cluster bootstrap** (implemented via `corr_with_cluster_bootstrap_ci_and_permutation_p` and used by wrappers such as `_date_cluster_corr_ci`): used for the **global** and **mean-daily** crowding summaries printed in `crowding_analysis.log`, resampling trading dates with replacement to account for within-day dependence.

**Row bootstrap (pair bootstrap).**

For a given sample:

1. Compute the original correlation:
   $$
   r_{\text{obs}} = r\big((x_i,y_i)_{i=1}^n\big).
   $$
2. For `BOOTSTRAP_RUNS` times:
   - Draw indices $i_1,\dots,i_n$ independently with replacement from $\{1,\dots,n\}$.
   - Form a bootstrap sample $\{(x_{i_k}, y_{i_k})\}_{k=1}^n$.
   - Compute the bootstrap correlation
     $$
     r^*_b = r\big((x_{i_k}, y_{i_k})_{k=1}^n\big).
     $$
3. Collect the bootstrap correlations $\{r^*_b\}$; let $\alpha$ be the nominal significance level (`ALPHA`, default 0.05).
4. The **bootstrap percentile confidence interval** is
   $$
   \big[\,\hat r_{\text{lo}}, \hat r_{\text{hi}}\,\big]
     = \left[ \text{quantile}_{\alpha/2}\big(\{r^*_b\}\big),\ 
              \text{quantile}_{1-\alpha/2}\big(\{r^*_b\}\big)\right].
   $$

This interval makes no Gaussian or Fisher‑z assumptions; it approximates the sampling distribution of $r$ directly from the data. In `scripts/crowding_analysis.py` this logic is implemented in `corr_with_ci`, which returns $(r, \text{lo}, \text{hi}, n)$ using the bootstrap percentile method for the CI. For very small samples ($n \le 3$), the script treats both $r$ and its CI as undefined (NaN) to avoid unstable behavior.

**Date-cluster bootstrap (high level).**

When `Date` is available as a clustering column, the script can form a dependence-aware CI by:

1. Grouping observations by trading day.
2. Resampling trading days with replacement for `BOOTSTRAP_RUNS` replications.
3. Recomputing the correlation on the concatenated resampled days.
4. Taking a percentile interval across replications.

### Permutation / bootstrap p-values (two-sided)

To test whether an observed correlation is compatible with **no relationship** between signs and imbalance, the script uses a permutation-style p-value:

1. Compute $r_{\text{obs}}$ on the original pairs $(x_i, y_i)$.
2. For `BOOTSTRAP_RUNS` times:
   - Keep $x_i$ fixed.
   - Randomly permute $y_i$ to break any alignment between signs and imbalances:
     $\{y_{\pi(i)}\}$ where $\pi$ is a random permutation of $\{1,\dots,n\}$.
   - Compute the permuted correlation
     $$
     r^{\text{perm}}_b = r\big((x_i, y_{\pi_b(i)})_{i=1}^n\big).
     $$
3. The empirical null distribution $\{r^{\text{perm}}_b\}$ approximates what correlations we would see if `Direction` and imbalance were **independent** but with the same marginals.
4. The two-sided p-value is
   $$
   p = 2 \min\left(
      \mathbb{P}(r^{\text{perm}}_b \ge r_{\text{obs}}),
      \mathbb{P}(r^{\text{perm}}_b \le r_{\text{obs}})
   \right),
   $$
   estimated by the corresponding empirical frequencies.

This is implemented in `corr_with_bootstrap_p`, which returns $(r_{\text{obs}}, p, n)$. The behavior is controlled by:

- `BOOTSTRAP_RUNS`: number of permutations (or “bootstrap runs”) to approximate the null.
- `P_VALUE_THRESHOLD`: the p‑value cutoff used to decide which correlations are considered statistically significant (only used for heatmap filtering when `BOOTSTRAP_HEATMAP=true`).

### Where these inference tools are used

- **Global within-group crowding (prop / client):**
  - `analyze_flow` reports:
    - $r = \text{Corr}(\text{Direction}, \text{imbalance\_local})$,
    - a **Date-cluster bootstrap** CI (printed as “Date-cluster bootstrap CI” in logs).
  - Intuition: how strongly a metaorder’s sign tends to go with/against the daily imbalance of other metaorders in the same group.

- **Daily crowding time series:**
  - `daily_crowding_ts` computes, per day:
    - $r_d = \text{Corr}(\text{Direction}, \text{imbalance})$,
    - a within-day **row bootstrap** CI,
    - and a within-day permutation p‑value (stored in the daily table; not used in the log-level mean summaries).
  - These are then summarized and plotted over time for proprietary vs client flow.

- **Cross-group and all-others crowding:**
  - The same correlation logic is used with:
    - cross-group environments (`imbalance_client_env`, `imbalance_prop_env`),
    - all-others imbalance (`imbalance_all_others`),
  and global and mean-daily summaries are reported with **Date-cluster bootstrap** CIs (p-values are not printed for these blocks in the current script).

- **Member-level analysis (per member, full sample):**
  - For each member $m$, the script computes:
    $$
    r_m = \text{Corr}\big(\epsilon_i,\ \text{imbalance\_client\_member\_env}(m, \text{Date}_i)\big),
    $$
    with:
    - a bootstrap CI on $r_m$,
    - a permutation p‑value $p_m$.
  - Members are filtered by **two symmetric count thresholds**:
    - number of proprietary metaorders for that member, $n_{\text{prop}, m} \ge \text{MIN\_METAORDERS\_PER\_MEMBER}$,
    - number of client metaorders for that member, $n_{\text{client}, m} \ge \text{MIN\_METAORDERS\_PER\_MEMBER}$.
  - Intuition: we only interpret member‑level correlations when both the proprietary side (directions) and the client environment (imbalance estimates) are supported by sufficient data.

- **Member–window heatmap (`MEMBER_WINDOW_DAYS`-day blocks):**
  - Dates are grouped into non-overlapping `MEMBER_WINDOW_DAYS`-day windows.
  - For each `(Member, Window)`:
    - The script counts:
      - proprietary metaorders in that window: $n_{\text{prop}, m, w}$,
      - client metaorders in that window: $n_{\text{client}, m, w}$.
    - If $n_{\text{prop}, m, w} < \text{N\_MIN\_PER\_MEMBER\_CLIENT}$ or $n_{\text{client}, m, w} < \text{N\_MIN\_PER\_MEMBER\_CLIENT}$, the cell is treated as missing (`NaN`).
    - Otherwise, it computes:
      $$
      r_{m,w} = \text{Corr}\big(\epsilon_i,\ \text{imbalance\_client\_member\_env}(m, \text{Date}_i)\big)
      $$
      using only proprietary metaorders of member $m$ in that `MEMBER_WINDOW_DAYS`-day window.
    - If `BOOTSTRAP_HEATMAP=true`, the script also computes a permutation p‑value $p_{m,w}$ and filters cells with $p_{m,w} > \text{P\_VALUE\_THRESHOLD}$ by setting them to `NaN`. With `BOOTSTRAP_HEATMAP=false`, no p‑values are computed for the heatmap and no significance filtering is applied.
  - Intuition: the heatmap highlights windows where a member’s proprietary flow is significantly aligned or opposed to the aggregate flow of its own clients, based on sufficiently many observations on both sides.

## Daily crowding analyses and outputs

All plots are saved under `images/{DATASET_NAME}/prop_vs_nonprop/png/` (and interactive counterparts in `.../html/`) with the smoothing window controlled by `SMOOTHING_DAYS` (default 3) and daily sample-size filter `MIN_N` (default 100).

- **Within-group crowding:** `run_daily_crowding_analysis` computes daily $r(D, \text{imbalance}^{\text{local}})$ for each group and plots
  - `daily_crowding_daily_corr.png`
  - `daily_crowding_rolling_{SMOOTHING_DAYS}d.png`
- **Cross-group crowding:** `run_cross_group_crowding_analysis` uses environment imbalances and plots
  - `cross_crowding_daily_corr.png`
  - `cross_crowding_rolling_{SMOOTHING_DAYS}d.png`
- **Crowding vs all others:** `run_all_vs_all_crowding_analysis` builds `imbalance_all_others` on prop+client combined and plots
  - `all_vs_all_crowding_daily_corr.png`
  - `all_vs_all_crowding_rolling_{SMOOTHING_DAYS}d.png`

## Selected results (latest run)

- For full plots and headline numbers, see **Latest run results (ftsemib, `pipeline_20260220_190521`)** at the top of this document.
- Global member-level correlation (prop direction vs member-level client imbalance):
  - $r = 0.011$ (p = 0.014, 95% CI [0.001, 0.022], $n = 37{,}278$).
  - Interpretation: very weak positive alignment on average.
- Per-member bar chart (`member_prop_client_crowding_hist.png`):
  - Members are sorted by their correlation; most sit near zero, with a few modestly positive/negative outliers.
- Member-window heatmap (`member_prop_client_crowding_heatmap_{MEMBER_WINDOW_DAYS}d.png`):
  - Windowing: non-overlapping 3-day trading-date blocks (`MEMBER_WINDOW_DAYS=3`).
  - Count threshold: $n_\\text{prop} \\ge 5$ and $n_\\text{client} \\ge 5$ per `(Member, Window)` (`N_MIN_PER_MEMBER_CLIENT=5`); no significance filtering in the default config (`BOOTSTRAP_HEATMAP=false`).
  - Summary: 84 windows; 2,124 `(Member, Window)` groups in proprietary data; 205 count-qualified cells; 188 finite-correlation cells (7 members have ≥1 finite cell).
  - Mean correlation across finite cells: about -0.028 (range [-1, +1]); typical `n_used`: p10=5, median=11, p90≈544.
  - Takeaway: most member–window blocks hover near zero, indicating limited systematic alignment; a handful of windows show stronger alignment or opposition (often in small-sample windows).

### Crowding vs participation rate ($\eta$)

The optional routine `run_crowding_vs_part_rate_analysis` studies how crowding varies with the metaorder participation rate
$$
\eta_i \equiv \frac{Q_i}{V_i^{\mathrm{during}}},
$$
by binning metaorders into participation-rate quantiles (typically deciles) and recomputing crowding metrics within each bin.

Besides the correlation $r=\mathrm{Corr}(\epsilon_i,\mathrm{imb}_i)$, the primary bounded effect size reported by this analysis is the **alignment**
$$
a_i \equiv \epsilon_i \cdot \mathrm{imb}_i \in [-1,1],
$$
which is positive when metaorders trade *with* the environment and negative when they trade *against* it.

In this context, $\mathrm{imb}_i$ is `imbalance_local` for the within-group (local) plots, while for the cross-group plots it is the other group’s environment imbalance (`imbalance_client_env` for proprietary metaorders, `imbalance_prop_env` for client metaorders).

Latest-run outputs for this analysis are written to:

- tables + provenance: `out_files/ftsemib/crowding_vs_part_rate/` (see `run_manifest.json`)
- curves: `images/ftsemib/crowding_vs_part_rate/png/` (interactive Plotly HTML in `.../html/`)

Run configuration (from `out_files/ftsemib/crowding_vs_part_rate/run_manifest.json`): 251 trading days (2024-06-03 to 2025-05-30), pooled $\eta$ deciles (`eta_bins=10`, `eta_binning=pooled_quantiles`, `eta_max=1.0`), Date-cluster bootstrap (`bootstrap_runs=1000`, `seed=0`, `alpha=0.05`), and `permutation_runs=0` (so no permutation p-values are produced for this run).

The tables below summarize the key effect sizes from that report:
- **Top-bottom $\Delta$:** top-$\eta$ decile minus bottom-$\eta$ decile.
- **Spearman:** trend estimate across $\eta$ bins (with bootstrap CI).

**Within-group (local) crowding.** Both proprietary and client flow exhibit monotone increases of alignment and correlation with $\eta$, with a much stronger effect for clients. Mean $|\mathrm{imb}|$ also increases with $\eta$, i.e. high-participation executions occur in more one-sided within-group environments.

| metric (local) | prop: top-bottom $\Delta$ | prop: Spearman | client: top-bottom $\Delta$ | client: Spearman |
|---|---|---|---|---|
| mean alignment $\mathbb{E}[\epsilon\cdot\mathrm{imb}]$ | 0.0296 [0.0237, 0.0359] | 0.980 [0.939, 1.000] | 0.1890 [0.1680, 0.2087] | 0.960 [0.927, 0.964] |
| mean $|\mathrm{imb}|$ | 0.0280 [0.0203, 0.0360] | 0.988 [0.952, 1.000] | 0.0253 [0.0164, 0.0346] | 0.798 [0.636, 0.915] |
| $\mathrm{Corr}(\epsilon,\mathrm{imb})$ | 0.0795 [0.0603, 0.0988] | 0.962 [0.891, 1.000] | 0.3860 [0.3498, 0.4184] | 0.955 [0.915, 0.964] |

**Cross-group crowding.** Proprietary metaorders are mildly contrarian to the contemporaneous client imbalance across $\eta$ (weak slope), while client metaorders become increasingly contrarian to the contemporaneous proprietary imbalance as $\eta$ rises (sign flip from slightly positive at low $\eta$ to strongly negative at high $\eta$ in correlation units). Mean $|\mathrm{imb}|$ of the other group’s environment increases with $\eta$ for both directions.

| metric (cross) | prop vs client env: top-bottom $\Delta$ | prop vs client env: Spearman | client vs prop env: top-bottom $\Delta$ | client vs prop env: Spearman |
|---|---|---|---|---|
| mean alignment $\mathbb{E}[\epsilon\cdot\mathrm{imb}]$ | -0.0057 [-0.0136, 0.0028] | -0.246 [-0.758, 0.406] | -0.0447 [-0.0551, -0.0358] | -0.878 [-0.927, -0.830] |
| mean $|\mathrm{imb}|$ | 0.0453 [0.0360, 0.0547] | 0.994 [0.976, 1.000] | 0.0244 [0.0183, 0.0305] | 0.921 [0.794, 0.988] |
| $\mathrm{Corr}(\epsilon,\mathrm{imb})$ | -0.0072 [-0.0242, 0.0108] | -0.063 [-0.636, 0.564] | -0.1739 [-0.2048, -0.1448] | -0.900 [-0.927, -0.842] |

**Curves by $\eta$ decile (PNG).** (Interactive HTML: see `../images/ftsemib/crowding_vs_part_rate/html/`.)

| Metric | Local (within-group) | Cross-group |
|---|---|---|
| Mean alignment $\\mathbb{E}[\\epsilon\\cdot\\mathrm{imb}]$ | ![Mean alignment vs eta (local)](../images/ftsemib/crowding_vs_part_rate/png/curve_mean_align_vs_eta_local.png) | ![Mean alignment vs eta (cross)](../images/ftsemib/crowding_vs_part_rate/png/curve_mean_align_vs_eta_cross.png) |
| Mean $|\\mathrm{imb}|$ | ![Mean abs imbalance vs eta (local)](../images/ftsemib/crowding_vs_part_rate/png/curve_mean_abs_imb_vs_eta_local.png) | ![Mean abs imbalance vs eta (cross)](../images/ftsemib/crowding_vs_part_rate/png/curve_mean_abs_imb_vs_eta_cross.png) |
| $\\mathrm{Corr}(\\epsilon,\\mathrm{imb})$ | ![Corr vs eta (local)](../images/ftsemib/crowding_vs_part_rate/png/curve_corr_dir_imb_vs_eta_local.png) | ![Corr vs eta (cross)](../images/ftsemib/crowding_vs_part_rate/png/curve_corr_dir_imb_vs_eta_cross.png) |

**Interpretation (high level).**
- Across both local and cross environments, mean $|\mathrm{imb}|$ increases with $\eta$, suggesting participation behaves like an execution-state proxy that co-moves with “crowded” regimes.
- Within-group: higher $\eta$ is associated with stronger herding, especially for clients (consistent with clustered directional demand).
- Cross-group: client flow becomes more contrarian to proprietary imbalance at high $\eta$, consistent with an intermediation/risk-transfer ecology rather than “everyone on the same side”.

**Limitations.**
- This is non-parametric conditioning (binning). Since $\eta$ correlates with size/speed proxies (e.g., `Q/V`, `Vt/V`), causal interpretation requires conditioning on these covariates.
- Date clustering is used for CIs; clustering by (ISIN, Date) yields similar inference in this run.

## Additional diagnostics

- **Daily count imbalance:** `run_daily_count_imbalance_analysis` plots `daily_counts_imbalance_timeseries.png` and `daily_counts_imbalance_histogram.png`.
- **Crowding vs participation rate ($\eta$):** `run_crowding_vs_part_rate_analysis` delegates to `scripts/crowding_vs_part_rate.py` and writes tables/plots under `out_files/{DATASET_NAME}/crowding_vs_part_rate/` and `images/{DATASET_NAME}/crowding_vs_part_rate/` (see `run_manifest.json` in the output folder for provenance).
- **Imbalance vs daily returns:** when `ATTACH_DAILY_RETURNS` and `PLOT_IMBALANCE_VS_RETURNS` are true, `imbalance_vs_daily_returns.png` is produced.
- **Imbalance distributions:** if `DISTRIBUTIONS_IMBALANCE=True`, PDFs of within- and cross-group imbalances are written to `imbalance_distribution.png`.
- **Autocorrelation of metaorder signs:** if `ACF_IMBALANCE=True`, per-ISIN ACF plots with bootstrap noise bands are written under `images/{DATASET_NAME}/prop_vs_nonprop/png/acf/`.
- **Metaorder dictionary stats:** when enabled via `RUN_METAORDER_DICT_STATS`, duration/inter-arrival/volume/Q/V/participation and metaorders-per-member plots are refreshed under `images/{DATASET_NAME}/{METAORDER_STATS_LEVEL}_{METAORDER_STATS_PROPRIETARY_TAG}/png/` and `.../html/`.

## Running the script

1. Run `scripts/metaorder_computation.py` for both `PROPRIETARY=True` and `PROPRIETARY=False` to generate the filtered metaorder info parquets.
2. Configure `config_ymls/crowding_analysis.yml` (paths, ATTACH_DAILY_RETURNS, ACF, etc.).
3. Execute `python scripts/crowding_analysis.py`. Updated parquet files are written in place when new columns are added, and plots appear under `images/{DATASET_NAME}/prop_vs_nonprop/png/` and `.../html/` (plus per-level folders when metaorder-dictionary stats are enabled).
