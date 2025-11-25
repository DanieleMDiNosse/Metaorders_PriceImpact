# Power-Law Impact Fits: Model, Filters, and WLS Procedure

This document describes in detail the procedure used in `metaorder_computation.py` to estimate power-law market impact functions from trade-by-trade data. The emphasis is on:

- the model specification,
- the filters applied to construct the estimation dataset,
- and the weighted least-squares (WLS) procedure in log space.

Throughout, we use notation aligned with the implementation.

---

## 1. From Trades to Metaorders

### 1.1 Trade-level preprocessing

Raw CSV files are mapped and transformed into a canonical trade dataset via `map_trade_codes` and `build_trades_view` (see `utils.py`). Conceptually, for each trade \(j\) we have:

- a **direction**
  \[
  \varepsilon_j \in \{+1,-1\},
  \]
  where \(\varepsilon_j = +1\) for aggressive buy trades and \(-1\) for aggressive sell trades;
- an **aggressive trading capacity** (proprietary vs non-proprietary), a client ID, a member ID, and a price \(P_j\);
- buy and sell quantities, from which the traded volume for the event is
  \[
  q_j = q^{\text{buy}}_j + q^{\text{sell}}_j \qquad \text{(one of the two is zero)}.
  \]

The script `metaorder_computation.py` applies an initial filter via `load_trades_filtered`:

1. **Proprietary vs. non-proprietary selection.**  
   Depending on the flag `PROPRIETARY` and the level `LEVEL` (member/client), the dataset is restricted to:
   - proprietary metaorders (`Trade Type Aggressive == "Dealing_on_own_account"`) or
   - non-proprietary metaorders (`Trade Type Aggressive != "Dealing_on_own_account"`)
   - non-proprietary metaorders channeled through possible multiple brokers.
2. **Trading hours filter.**  
   Only trades with timestamps in the continuous trading window
   \[
   t_j \in [09{:}30{:}00,\,17{:}30{:}00]
   \]
   are retained.

3. **Sorting and indexing.**  
   Trades are sorted by `Trade Time` (and a row index) to obtain a time-ordered event stream for each ISIN.

These preprocessed trades form the input for the metaorder detection.

### 1.2 Metaorder detection

Metaorders are defined as contiguous sequences of trades for a given *agent* (member or client) within an ISIN, with the **same sign** of aggressive activity. If `LEVEL=member`, all the trades of the metaorders must be channeled through the same member. The procedure is:

1. For each agent \(a\), construct sparse activity vectors via `agents_activity_sparse`:
   - indices \(\mathcal{I}_a = \{j : \text{agent } a \text{ traded}\}\),
   - associated signs \(\varepsilon_j \in \{+1,-1\}\) for \(j \in \mathcal{I}_a\).

2. Build a dense sign process over all trades for that ISIN; then apply `find_metaorders` to detect **runs of constant sign**:
   - a run is a maximal interval of indices
     \[
     \mathcal{M} = \{j_s, j_s+1, \dots, j_e\}
     \]
     such that \(\varepsilon_{j} = \varepsilon_{j_s}\) for all \(j \in \mathcal{M}\),
   - only runs with length
     \[
     |\mathcal{M}| \geq \texttt{min\_child} = 2
     \]
     are kept.

3. Additional **metaorder-level filters** in `compute_metaorders_per_isin`:
   - **same day:** the start and end timestamps must belong to the same calendar day,
   - **single client:** all trades in the run must belong to a single client ID.

Each surviving run \(\mathcal{M}\) is treated as a metaorder. For metaorder \(i\), we denote its set of trade indices by \(\mathcal{M}_i\) and its sign by
\[
\varepsilon_i \in \{+1,-1\},
\]
defined as the sign of the last trade in the run (coinciding with the run sign).

### 1.3 Splitting by inactivity and trading session

Even after initial detection, very long or overnight sequences are undesirable. The script further refines the metaorders using `split_metaorders_by_gap` and a time-of-day filter:

- Let \(t_j\) be the timestamp of trade \(j\). Define the inter-trade gaps within a metaorder
  \[
  \Delta t_k = t_{j_{k+1}} - t_{j_k}.
  \]
- If \(\Delta t_k > \texttt{MAX\_GAP} = 1 \text{ hour}\), the metaorder is split at that point into separate child metaorders.
- Child metaorders with fewer than `MIN_TRADES = 2` trades are discarded.
- Only trades within a stricter trading session
  \[
  t_j \in [09{:}30,\,17{:}30]
  \]
  are used when applying this splitting, ensuring metaorders do not bridge illiquid pre-/post-market periods.

The result is a collection of same-day, same-client, same-sign metaorders per ISIN.

---

## 2. Daily Volatility and Volume Normalization

Impact is measured in *volatility units*, and size is expressed as a fraction of daily traded volume. Both quantities are computed per ISIN and per day using `build_daily_cache` in `utils.py`.

### 2.1 Daily traded volume

For a given day \(d\), denote the set of trades for a given ISIN by \(\mathcal{J}_d\). The daily total volume is
\[
V_d = \sum_{j \in \mathcal{J}_d} q_j,
\]
where \(q_j\) is the traded quantity of trade \(j\) (buy plus sell quantities).

### 2.2 Daily volatility (realized kernel)

Let \(P_t\) be the last traded price process during day \(d\). The construction is:

1. Restrict to trades in \([d, d+1)\) and build a price series \(P_{t_\ell}\) with **unique timestamps**, keeping the last trade per timestamp.
2. Resample this series every \(\Delta = 120\) seconds using last-tick interpolation and forward-fill within the day:
   \[
   P_{t_0}, P_{t_1}, \dots, P_{t_n}, \qquad t_{\ell+1} - t_\ell = \Delta.
   \]
3. Compute log-returns
   \[
   r_\ell = \log P_{t_{\ell+1}} - \log P_{t_\ell}, \quad \ell = 0,\dots, n-1.
   \]
4. Apply a realized kernel estimator \( \widehat{RK}_d \) (Parzen kernel with bandwidth \(H \sim n^{2/3}\)) to the return sequence \((r_\ell)\). The code computes
   \[
   \widehat{\sigma}_d = \sqrt{\widehat{RK}_d},
   \]
   and stores it as the **daily volatility**.

If the resampled series is too short or degenerate, \(\widehat{\sigma}_d\) may be set to NaN; such days are implicitly filtered out at the impact- and WLS-filtering stage because they induce non-finite values.

### 2.3 Per-metaorder quantities

For a metaorder \(i\) occurring on day \(d(i)\), with trade index set \(\mathcal{M}_i = \{j_s,\dots,j_e\}\), the following quantities are computed in `compute_metaorders_info`:

- **Metaorder volume**
  \[
  Q_i = \sum_{j \in \mathcal{M}_i} q_j.
  \]

- **Volume traded during the metaorder**
  \[
  V_i^{\text{during}} = \sum_{j=j_s}^{j_e} q_j,
  \]
  i.e. the total market volume in the instrument between the first and last trade of the metaorder (not just the agent's volume).

- **Daily total volume**
  \[
  V_{d(i)} = V_d \quad\text{as in Section 2.1.}
  \]

- **Size proxy (participation in daily volume)**
  \[
  \phi_i \equiv \frac{Q_i}{V_{d(i)}},
  \]
  stored as column `Q/V`. This is the primary regressor for the impact model.

- **Participation rate during the metaorder**
  \[
  \eta_i \equiv \frac{Q_i}{V_i^{\text{during}}},
  \]
  stored as `Participation Rate`. This is used for conditioning (e.g. stratifying fits by participation quantiles) but not as a regressor in the baseline model.

---

## 3. Impact Definition and Model Specification

### 3.1 Metaorder impact in volatility units

Let \(P^{\text{start}}_i\) be the price at the first trade of metaorder \(i\), and \(P^{\text{end}}_i\) the price at the last trade. The signed log-return over the execution interval is
\[
\Delta p_i = \log P^{\text{end}}_i - \log P^{\text{start}}_i.
\]

Let \(\varepsilon_i \in \{+1,-1\}\) be the **direction** of the metaorder (buy \(+1\), sell \(-1\)). The daily volatility on the same day is \(\widehat{\sigma}_{d(i)}\). The (instantaneous) impact of metaorder \(i\) in volatility units is defined as
\[
I_i \equiv \frac{\varepsilon_i \, \Delta p_i}{\widehat{\sigma}_{d(i)}}.
\]

In the code, this is implemented as
\[
I_i = \frac{\texttt{Price Change}_i \times \texttt{Direction}_i}{\texttt{Daily Vol}_i},
\]
and stored in column `Impact`.

### 3.2 Power-law impact model

The core model is a **power-law relation** between the (expected) volatility-normalized impact and the metaorder size in units of daily volume:
\[
\mathbb{E}[I_i \mid \phi_i] = Y \, \phi_i^{\gamma},
\]
where

- \(Y > 0\) is a scale parameter (impact for unit participation),
- \(\gamma\) is the **impact exponent**, typically in \((0,1)\) for concave impact (e.g., square-root impact if \(\gamma \approx 1/2\)).

In log space, for \(\phi_i > 0\) and \(I_i > 0\),
\[
\log \mathbb{E}[I_i \mid \phi_i]
  = \log Y + \gamma \log \phi_i.
\]

Because individual \(I_i\) are noisy and can be negative, the estimation is not performed directly on individual observations. Instead, the dataset is **log-binned in \(\phi\)**, and the regression is performed on **bin-level mean impacts**. This substantially reduces heteroskedasticity and noise, at the cost of aggregating data.

### 3.3 Execution and aftermath impact paths

Beyond the scalar end-of-execution impact \(I_i\), the script also tracks the **full impact path** of each metaorder, both during execution and in an aftermath window:

- For each metaorder \(i\), the **partial impact path** records the normalized impact after each child trade in \(\mathcal{M}_i\) (same normalization as \(I_i\)).
- The **aftermath impact path** samples the normalized impact at a fixed number of evenly spaced timestamps after the end of execution, up to a multiple of the metaorder duration (controlled by `AFTERMATH_DURATION_MULTIPLIER` and `AFTERMATH_NUM_SAMPLES`).
- These two vectors are stored in the metaorders info table as columns `partial_impact` and `aftermath_impact`, one pair per metaorder.

For large datasets, these paths are stored compactly as packed float32 byte blobs (one contiguous vector per metaorder) to keep memory usage under control. Downstream functions such as `plot_normalized_impact_path` transparently unpack these blobs, interpolate each path onto a common normalized time grid \(t \in [0, 1 + \text{duration\_multiplier}]\), and average across metaorders to produce the **normalized impact path** plot. This provides a time-resolved view of impact build-up during execution and its relaxation in the aftermath, complementary to the scalar power-law fits.

---

## 4. Filtering Stage Before Fitting

After metaorders are computed and their summary dataframe is built, all filters are applied via `apply_metaorder_filters` in `metaorder_computation.py`. The resulting filtered dataset is saved to `out_files/metaorders_info_sameday_filtered_*.parquet` and reused directly by the WLS step (no re-filtering there).

### 4.1 Structural filters (enforced during construction)

- Trade-level filters: intraday trades only (09:30-17:30) and the chosen proprietary/non-proprietary subset.
- Metaorder construction: same-sign runs, at least two trades, same day, single client, no inactivity gaps above 1 hour (splits runs; short fragments are dropped).

### 4.2 Unified post-computation filter (`apply_metaorder_filters`)

Let \(\phi_i = Q_i / V_{d(i)}\) and \(I_i = \varepsilon_i \Delta p_i / \widehat{\sigma}_{d(i)}\). The unified filter performs:

1. **Duration floor.** Keep metaorders with execution time
   \[
   T_i \geq \texttt{SECONDS\_FILTER} = 60\ \text{s}.
   \]
2. **Minimum relative size.** Require
   \[
   \phi_i > \texttt{MIN\_QV} = 10^{-5}.
   \]
3. **Impact computation and cleanup.** Compute \(I_i\); replace \(\pm\infty\) in numeric columns with NaN; drop rows with non-finite `Q/V` or `Impact`.

This yields a clean table with well-defined \((\phi_i, I_i)\) pairs and auxiliary fields such as participation rate, ready for binning.

### 4.3 Bin-level guards inside the WLS routine

`fit_power_law_logbins_wls_new` applies only bin-level checks:

- log-binning of \(\phi\) into `n_logbins` bins;
- keep bins with count \(\ge \texttt{min\_count}\), positive/finite mean impact, and positive/finite SEM;
- require at least \(2+p\) bins (intercept, exponent, and \(p\) optional controls) to avoid underdetermined regressions.

---

## 5. Weighted Least-Squares in Log Space

### 5.1 Regression model on binned data

On the retained bins, define \(X_k = \log \phi_k^{\text{center}}\) and \(Z_k = \log \bar{I}_k\). The model is
\[
Z_k = a + \gamma X_k + \sum_{\ell=1}^p \beta_\ell C_{k,\ell} + \epsilon_k,
\]
with \(a = \log Y\), exponent \(\gamma\), optional controls \(C_{k,\ell}\), and residuals \(\epsilon_k\). The prefactor estimate is \(\widehat{Y} = e^{\widehat{a}}\).

### 5.2 Weights from bin uncertainty

Using a delta-method approximation,
\[
\operatorname{Var}(Z_k) \approx \left(\frac{\text{SEM}_k}{\bar{I}_k}\right)^2,
\qquad
w_k = \frac{1}{\operatorname{Var}(Z_k)} = \frac{1}{(\text{SEM}_k / \bar{I}_k)^2}.
\]
Bins with more observations or lower dispersion carry higher weight.

### 5.3 Estimator, uncertainty, and diagnostics

The WLS estimate solves
\[
\widehat{\theta} = (A^\top W A)^{-1} A^\top W \mathbf{Z},
\]
with \(A\) the design matrix \([1, X, C_1, \dots, C_p]\) and \(W = \operatorname{diag}(w_k)\). Residual variance is
\[
\widehat{s}^2 = \frac{\sum_k w_k (Z_k - \widehat{Z}_k)^2}{K - (2+p)},
\]
yielding \(\operatorname{Cov}(\widehat{\theta}) \approx \widehat{s}^2 (A^\top W A)^{-1}\). Standard errors follow from the diagonal; \(\operatorname{SE}(\widehat{Y}) \approx \widehat{Y}\,\operatorname{SE}(\widehat{a})\).

Goodness-of-fit metrics:
- **Log-space \(R^2_{\log}\):** weighted \(R^2\) on \(Z_k\).
- **Linear-space \(R^2_{\text{lin}}\):** \(R^2\) on \(\bar{I}_k\) versus \(\widehat{Y}\phi_k^{\widehat{\gamma}}\) (ignoring controls), for interpretability in the original scale.

---

## 6. Summary

- Metaorders: same-sign, same-day, single-client sequences with at least two trades; split on inactivity \(>1\) hour; trade-level intraday and proprietary/non-proprietary filters.
- Normalization: size proxy \(\phi_i = Q_i / V_{d(i)}\); impact \(I_i = \varepsilon_i \Delta p_i / \widehat{\sigma}_{d(i)}\).
- Single post-computation filter (`apply_metaorder_filters`): duration \(\ge 60\)s, \(\phi_i > 10^{-5}\), finite `Impact`/`Q/V`.
- Estimation: log-binned WLS of \(I\) on \(\phi\) (plus optional controls), with bin-level guards and delta-method weights; report \(\widehat{Y}\), \(\widehat{\gamma}\), standard errors, and \(R^2\) diagnostics.

This pipeline concentrates all filtering before fitting, minimizing repeated conditioning while preserving the econometric rigor of the power-law impact estimation.
