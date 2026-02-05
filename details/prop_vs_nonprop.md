# Proprietary vs Non-Proprietary Crowding in `metaorder_statistics.py`

`metaorder_statistics.py` compares proprietary and client flow after metaorders have been built by `metaorder_computation.py`. It attaches imbalance metrics, optional daily returns, and produces crowding diagnostics and plots.

The imbalance study is designed to answer several questions:

- Do metaorders tend to go **with** (herding) or **against** (contrarian/liquidity-providing) the prevailing signed volume imbalance on the same stock and day?
- How does this behavior differ between **proprietary** flow and **client** flow?
- How crowded is each group:
  - relative to its **own** daily flow (within-group crowding),
  - relative to the **other** group’s flow (cross-group crowding),
  - relative to the union of **all other** metaorders (prop+client combined),
  - and, for members, relative to the flow of their **own clients** (member-level crowding)?

All these questions are expressed as correlations between **metaorder direction** and suitably defined **signed volume imbalance** measures.

## Inputs and preprocessing
- Expects per-metaorder parquet files (defaults in the script):
  - `out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_member_proprietary.parquet`
  - `out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_member_non_proprietary.parquet`
- Adds `Group` (`"prop"` or `"client"`) and ensures a `Date` column (derived from `Period` if missing).
- If `RUN_METAORDER_DICT_STATS=True`, it first calls `run_metaorder_dict_statistics` to rebuild distribution plots (durations, inter-arrivals, volumes, Q/V, participation, metaorders per member) from `metaorders_dict_all_{LEVEL}_{PROPRIETARY_TAG}.pkl` and the parquet tapes in `data/{DATASET_NAME}/`. Outputs land in `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/`.
- Within-group imbalance is added via `add_daily_imbalance` unless already present.
- Cross-group environment imbalances are added via `attach_environment_imbalance` unless already present (`imbalance_client_env` for prop, `imbalance_prop_env` for client).
- Member-level client environment imbalance is added via `attach_member_client_imbalance` unless already present (`imbalance_client_member_env` on proprietary metaorders).
- If `ATTACH_DAILY_RETURNS=True`, close-to-close daily log returns are computed from the raw per-ISIN parquet trades (using `RETURNS_DATA_DIR` and `RETURNS_TRADING_HOURS`, with `RETURNS_DATA_DIR` typically set to `data/{DATASET_NAME}`) and attached as `DAILY_RETURN_COL`. Parquets are rewritten when new columns are added.

## Core objects and notation

After `metaorder_computation.py` has run, each row in the proprietary/client parquet files corresponds to a metaorder with fields including:

- `ISIN`: instrument identifier.
- `Date`: trading day.
- `Member`: member/broker executing the metaorder.
- `Client`: client identifier (for client flow).
- `Q`: metaorder volume.
- `Direction ∈ {+1, −1}`: buy (+1) or sell (−1).

The imbalance quantities are always based on **signed volume**:

\[
Q_i \epsilon_i,
\]

aggregated over appropriate subsets of metaorders and normalized by total volume.

For a set of indices \(\mathcal{S}\),

\[
\text{imbalance}(\mathcal{S})
  = \frac{\sum_{j \in \mathcal{S}} Q_j \epsilon_j}{\sum_{j \in \mathcal{S}} Q_j},
\]

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

For a metaorder \(i\) on ISIN \(k\) and date \(d\), with volume \(Q_i\) and direction \(\epsilon_i \in \{+1,-1\}\), define
\[
\text{imbalance}^{\text{local}}_i
	  = \frac{\sum_{j \in \mathcal{G}_{k,d}\setminus\{i\}} Q_j \epsilon_j}
	         {\sum_{j \in \mathcal{G}_{k,d}\setminus\{i\}} Q_j},
	\]
	where \(\mathcal{G}_{k,d}\) are the other metaorders on the same \((k,d)\). Days with a single metaorder yield `NaN` for this field. This is computed separately for proprietary and client flow.

In code, this is handled by:

- `add_daily_imbalance(df, group_cols=("ISIN", "Date"), side_col="Direction", vol_col="Q", new_col="imbalance_local")`:
  - For each `(ISIN, Date)` group:
    - Adds `__QD__ = Q * Direction`.
    - Computes `total_Q` and `total_QD` via `groupby.transform("sum")`.
    - For each row:
      - `denom = total_Q - Q_i`, `numer = total_QD - __QD__i`.
      - `imbalance_local = numer / denom` where `denom > 0`, else `NaN`.

This exactly implements the self-exclusion \(\mathcal{G}_{k,d}\setminus\{i\}\).

## Cross-group and all-others imbalance

- **Cross-group environment (ISIN–Date level):**
  - For proprietary metaorders, `imbalance_client_env(ISIN, Date)` is defined as:
    \[
    \frac{\sum_{\text{client metaorders on }(k,d)} Q_j \epsilon_j}
           {\sum_{\text{client metaorders on }(k,d)} Q_j}.
    \]
    Implemented by `attach_environment_imbalance(metaorders_proprietary, metaorders_non_proprietary, new_col="imbalance_client_env")`.
  - For client metaorders, `imbalance_prop_env(ISIN, Date)` is defined analogously using proprietary flow and attached to the client dataframe.

- **All-others imbalance (prop + client combined, self-excluded):**
  - In `run_all_vs_all_crowding_analysis`, proprietary and client metaorders are concatenated, and `add_daily_imbalance` is applied with `group_cols=("ISIN", "Date")` and `new_col="imbalance_all_others"`.
  - For each metaorder \(i\), `imbalance_all_others` is the signed volume imbalance of all other metaorders (prop+client) on the same `(ISIN, Date)`, excluding \(i\) itself, mirroring the definition of `imbalance_local` but on the combined set.

## Member-level client imbalance (Member–Date level)

The member-level analysis asks whether a member’s proprietary flow tends to align with the aggregate flow of its **own clients**.

For a given member \(m\) and date \(d\), define \(\mathcal{C}_{m,d}\) as the set of **client** metaorders with:
- `Member = m`,
- `Date = d`,
- any ISIN.

The member-level client imbalance is:

\[
\text{imbalance\_client\_member\_env}(m,d)
  = \frac{\sum_{j \in \mathcal{C}_{m,d}} Q_j \epsilon_j}
         {\sum_{j \in \mathcal{C}_{m,d}} Q_j},
\]

with `NaN` if the denominator is zero (no client volume for that member/day).

This is computed by:

- `attach_member_client_imbalance(metaorders_proprietary, metaorders_non_proprietary, new_col="imbalance_client_member_env")`:
  - Calls `compute_environment_imbalance` on the **client** dataframe with `group_cols=("Member", "Date")`.
  - Merges the resulting `(Member, Date) → imbalance_client_member_env` table onto the **proprietary** dataframe keyed by `(Member, Date)`.
  - Each proprietary metaorder \(i\) with `(Member_i, Date_i)` receives the same member-level client imbalance value for that member and day.

The member-level crowding analysis is then performed by:

- `run_member_level_prop_client_crowding_analysis(metaorders_proprietary, metaorders_non_proprietary, env_col="imbalance_client_member_env", ...)`:
  - Computes a **global** correlation across all proprietary metaorders:
    \[
    r_{\text{global}} = \text{Corr}\big(\epsilon_i,\ \text{imbalance\_client\_member\_env}(\text{Member}_i, \text{Date}_i)\big)
    \]
    with a Fisher-z confidence interval.
  - Computes **per-member** correlations:
    \[
    r_m = \text{Corr}\big(\epsilon_i,\ \text{imbalance\_client\_member\_env}(m, \text{Date}_i)\big)
    \quad \text{for metaorders with Member } m,
    \]
    including a **bootstrap percentile CI** (`lo`, `hi`) and `n` (sample size).
  - Filters to members with at least `MIN_METAORDERS_PER_MEMBER` **proprietary** metaorders and at least `MIN_METAORDERS_PER_MEMBER` **client** metaorders; exports:
    - `member_prop_client_crowding_per_member.parquet` with one row per member (`Member`, `r`, `lo`, `hi`, `p`, `n`, `n_client_total`).
    - `member_prop_client_crowding_hist.png`, a per-member **bar chart** (x = Member, y = correlation), sorted by correlation.
  - All correlations in this routine also carry a **bootstrap p-value** (controlled by `BOOTSTRAP_RUNS` and reported in logs/Parquet); the p-value threshold used for filtering is `P_VALUE_THRESHOLD`.
  - Builds a **5-day, non-overlapping window heatmap** (saved to `member_prop_client_crowding_heatmap.png`):
    - Dates are binned into consecutive 5-day blocks (no overlap).
    - For each `(Member, Window)`, compute `Corr(Direction_prop, imbalance_client_member_env)` using proprietary metaorders in that window **only if** there are at least 10 proprietary metaorders **and** 10 client metaorders for that member in the same window; otherwise the cell is `NaN`.
    - If the bootstrap p-value for that `(Member, Window)` exceeds `P_VALUE_THRESHOLD`, the plotted value is also set to `NaN` (i.e., only statistically significant correlations are shown).
    - Heatmap axes: x = members, y = 5-day windows; colormap clipped to [-1, +1].

## Inference: correlations, bootstrap confidence intervals, and permutation p-values

### Pearson correlation

Given a sample of pairs \((x_i, y_i)_{i=1}^n\), the script measures linear crowding via the **Pearson correlation**:
\[
r = \frac{\sum_{i=1}^n (x_i - \bar x)(y_i - \bar y)}
         {\sqrt{\sum_{i=1}^n (x_i - \bar x)^2}\,\sqrt{\sum_{i=1}^n (y_i - \bar y)^2}},
\]
where \(\bar x\) and \(\bar y\) are sample means. In our context:

- \(x_i\) is typically `Direction` (±1) of a metaorder.
- \(y_i\) is some imbalance measure (`imbalance_local`, `imbalance_client_env`, `imbalance_client_member_env`, etc.).

This \(r\) is the point estimate of how strongly metaorder signs co-move with the imbalance proxy.

### Bootstrap confidence intervals (non-parametric)

Instead of using a parametric Fisher‑z interval, the script estimates uncertainty around \(r\) via a **non‑parametric bootstrap** over pairs \((x_i, y_i)\).

For a given sample:

1. Compute the original correlation:
   \[
   r_{\text{obs}} = r\big((x_i,y_i)_{i=1}^n\big).
   \]
2. For `BOOTSTRAP_RUNS` times:
   - Draw indices \(i_1,\dots,i_n\) independently with replacement from \(\{1,\dots,n\}\).
   - Form a bootstrap sample \(\{(x_{i_k}, y_{i_k})\}_{k=1}^n\).
   - Compute the bootstrap correlation
     \[
     r^*_b = r\big((x_{i_k}, y_{i_k})_{k=1}^n\big).
     \]
3. Collect the bootstrap correlations \(\{r^*_b\}\); let \(\alpha\) be the nominal significance level (`ALPHA`, default 0.05).
4. The **bootstrap percentile confidence interval** is
   \[
   \big[\,\hat r_{\text{lo}}, \hat r_{\text{hi}}\,\big]
     = \left[ \text{quantile}_{\alpha/2}\big(\{r^*_b\}\big),\ 
              \text{quantile}_{1-\alpha/2}\big(\{r^*_b\}\big)\right].
   \]

This interval makes no Gaussian or Fisher‑z assumptions; it approximates the sampling distribution of \(r\) directly from the data. In `metaorder_statistics.py` this logic is implemented in `corr_with_ci`, which returns \((r, \text{lo}, \text{hi}, n)\) using the bootstrap percentile method for the CI. For very small samples (\(n \le 3\)), the script treats both \(r\) and its CI as undefined (NaN) to avoid unstable behavior.

### Permutation / bootstrap p-values (two-sided)

To test whether an observed correlation is compatible with **no relationship** between signs and imbalance, the script uses a permutation-style p-value:

1. Compute \(r_{\text{obs}}\) on the original pairs \((x_i, y_i)\).
2. For `BOOTSTRAP_RUNS` times:
   - Keep \(x_i\) fixed.
   - Randomly permute \(y_i\) to break any alignment between signs and imbalances:
     \(\{y_{\pi(i)}\}\) where \(\pi\) is a random permutation of \(\{1,\dots,n\}\).
   - Compute the permuted correlation
     \[
     r^{\text{perm}}_b = r\big((x_i, y_{\pi_b(i)})_{i=1}^n\big).
     \]
3. The empirical null distribution \(\{r^{\text{perm}}_b\}\) approximates what correlations we would see if `Direction` and imbalance were **independent** but with the same marginals.
4. The two-sided p-value is
   \[
   p = 2 \min\left(
      \mathbb{P}(r^{\text{perm}}_b \ge r_{\text{obs}}),
      \mathbb{P}(r^{\text{perm}}_b \le r_{\text{obs}})
   \right),
   \]
   estimated by the corresponding empirical frequencies.

This is implemented in `corr_with_bootstrap_p`, which returns \((r_{\text{obs}}, p, n)\). The behavior is controlled by:

- `BOOTSTRAP_RUNS`: number of permutations (or “bootstrap runs”) to approximate the null.
- `P_VALUE_THRESHOLD`: the p‑value cutoff used to decide which correlations are considered statistically significant (e.g. when filtering the 5‑day heatmap).

### Where these inference tools are used

- **Global within-group crowding (prop / client):**
  - `analyze_flow` reports:
    - \(r = \text{Corr}(\text{Direction}, \text{imbalance\_local})\),
    - a bootstrap CI,
    - a permutation p‑value.
  - Intuition: how strongly a metaorder’s sign tends to go with/against the daily imbalance of other metaorders in the same group.

- **Daily crowding time series:**
  - `daily_crowding_ts` computes, per day:
    - \(r_d = \text{Corr}(\text{Direction}, \text{imbalance})\),
    - a bootstrap CI,
    - a permutation p‑value.
  - These are then summarized and plotted over time for proprietary vs client flow.

- **Cross-group and all-others crowding:**
  - The same correlation logic is used with:
    - cross-group environments (`imbalance_client_env`, `imbalance_prop_env`),
    - all-others imbalance (`imbalance_all_others`),
  so that each crowding measure has a bootstrap CI and p‑value.

- **Member-level analysis (per member, full sample):**
  - For each member \(m\), the script computes:
    \[
    r_m = \text{Corr}\big(\epsilon_i,\ \text{imbalance\_client\_member\_env}(m, \text{Date}_i)\big),
    \]
    with:
    - a bootstrap CI on \(r_m\),
    - a permutation p‑value \(p_m\).
  - Members are filtered by **two symmetric count thresholds**:
    - number of proprietary metaorders for that member, \(n_{\text{prop}, m} \ge \text{MIN\_METAORDERS\_PER\_MEMBER}\),
    - number of client metaorders for that member, \(n_{\text{client}, m} \ge \text{MIN\_METAORDERS\_PER\_MEMBER}\).
  - Intuition: we only interpret member‑level correlations when both the proprietary side (directions) and the client environment (imbalance estimates) are supported by sufficient data.

- **Member–window heatmap (5-day blocks):**
  - Dates are grouped into non-overlapping 5‑day windows.
  - For each `(Member, Window)`:
    - The script counts:
      - proprietary metaorders in that window: \(n_{\text{prop}, m, w}\),
      - client metaorders in that window: \(n_{\text{client}, m, w}\).
    - If \(n_{\text{prop}, m, w} < 10\) or \(n_{\text{client}, m, w} < 10\), the cell is treated as missing (`NaN`).
    - Otherwise, it computes:
      \[
      r_{m,w} = \text{Corr}\big(\epsilon_i,\ \text{imbalance\_client\_member\_env}(m, \text{Date}_i)\big)
      \]
      using only proprietary metaorders of member \(m\) in that 5‑day window, and a permutation p‑value \(p_{m,w}\).
    - If \(p_{m,w} > \text{P\_VALUE\_THRESHOLD}\), the corresponding cell is set to `NaN` (only significant correlations are shown).
  - Intuition: the heatmap highlights windows where a member’s proprietary flow is significantly aligned or opposed to the aggregate flow of its own clients, based on sufficiently many observations on both sides.

## Daily crowding analyses and outputs

All plots are saved under `images/{DATASET_NAME}/prop_vs_nonprop/` with the smoothing window controlled by `SMOOTHING_DAYS` (default 5) and daily sample-size filter `MIN_N` (default 100).

- **Within-group crowding:** `run_daily_crowding_analysis` computes daily \(r(D, \text{imbalance}^{\text{local}})\) for each group and plots
  - `daily_crowding_daily_corr.png`
  - `daily_crowding_rolling_{SMOOTHING_DAYS}d.png`
- **Cross-group crowding:** `run_cross_group_crowding_analysis` uses environment imbalances and plots
  - `cross_crowding_daily_corr.png`
  - `cross_crowding_rolling_{SMOOTHING_DAYS}d.png`
- **Crowding vs all others:** `run_all_vs_all_crowding_analysis` builds `imbalance_all_others` on prop+client combined and plots
  - `all_vs_all_crowding_daily_corr.png`
  - `all_vs_all_crowding_rolling_{SMOOTHING_DAYS}d.png`

## Selected results (latest run)

- Global member-level correlation (prop direction vs member-level client imbalance):
  - \(r \approx 0.014\) (95% CI ≈ [0.003, 0.024]), \(n \approx 35{,}695\) metaorders.
  - Interpretation: very weak positive alignment on average.
- Per-member bar chart (`member_prop_client_crowding_hist.png`):
  - Members are sorted by their correlation; most sit near zero, with a few modestly positive/negative outliers.
- 5-day window heatmap (`member_prop_client_crowding_heatmap.png`):
  - Valid cells (meeting \(n_\text{prop} \ge 10\) and \(n_\text{client} \ge 10\)): 91 windows across members.
  - Mean correlation across valid cells: about -0.02 (essentially flat).
  - Range observed in the sample: roughly [-0.96, +1.00]; the extremes come from relatively concentrated windows.
  - Takeaway: most member–window blocks hover near zero, indicating limited systematic alignment; a handful of windows show stronger alignment or opposition.

## Additional diagnostics

- **Daily count imbalance:** `run_daily_count_imbalance_analysis` plots `daily_counts_imbalance_timeseries.png` and `daily_counts_imbalance_histogram.png`.
- **Participation vs |imbalance|:** `plot_participation_vs_abs_imbalance` saves `participation_vs_abs_imbalance.png`.
- **Imbalance vs daily returns:** when `ATTACH_DAILY_RETURNS` and `PLOT_IMBALANCE_VS_RETURNS` are true, `imbalance_vs_daily_returns.png` is produced.
- **Imbalance distributions:** if `DISTRIBUTIONS_IMBALANCE=True`, PDFs of within- and cross-group imbalances are written to `imbalance_distribution.png`.
- **Autocorrelation of metaorder signs:** if `ACF_IMBALANCE=True`, per-ISIN ACF plots with bootstrap noise bands are written under `images/{DATASET_NAME}/prop_vs_nonprop/acf/`.
- **Metaorder dictionary stats:** when enabled via `RUN_METAORDER_DICT_STATS`, duration/inter-arrival/volume/Q/V/participation and metaorders-per-member plots are refreshed under `images/{DATASET_NAME}/{METAORDER_STATS_LEVEL}_{METAORDER_STATS_PROPRIETARY_TAG}/`.

## Running the script

1. Run `metaorder_computation.py` for both `PROPRIETARY=True` and `PROPRIETARY=False` to generate the filtered metaorder info parquets.
2. Configure flags at the top of `metaorder_statistics.py` (paths, ATTACH_DAILY_RETURNS, ACF, etc.).
3. Execute `python metaorder_statistics.py`. Updated parquet files are written in place when new columns are added, and plots appear under `images/{DATASET_NAME}/prop_vs_nonprop/` (plus per-level folders when metaorder-dictionary stats are enabled).
