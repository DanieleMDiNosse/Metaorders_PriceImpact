# Metaorders, Price Impact, and Crowding (CONSOB trades)

This repository is a toolkit for reconstructing **metaorders** (large orders executed via many child trades), estimating **volatility-normalized market impact**, and studying **crowding / co-impact** by comparing **proprietary** vs **client** aggressive flow in CONSOB trade data. Metaorders are identified as same-sign sequences of aggressive trades for an agent (member or client) within an instrument and day; impact is measured in daily-volatility units to make sizes comparable across names and days; and the proprietary/client segmentation enables crowding analyses that ask whether each group tends to trade *with* or *against* the prevailing signed volume imbalance.

---

## Repository at a glance

The core logic is in a small set of scripts:

- [`metaorder_computation.py`](../metaorder_computation.py): preprocessing, metaorder detection, per-metaorder features, impact estimation (log-binned WLS), optional impact paths/signature plots/surfaces; controlled by [`config_ymls/metaorder_computation.yml`](../config_ymls/metaorder_computation.yml).
- [`metaorder_statistics.py`](../metaorder_statistics.py): crowding/imbalance metrics, correlations with bootstrap CIs and permutation p-values, daily/rolling plots, member-level analyses; controlled by [`config_ymls/metaorder_statistics.yml`](../config_ymls/metaorder_statistics.yml).
- [`plot_prop_nonprop_fits.py`](../plot_prop_nonprop_fits.py): reloads filtered outputs and overlays proprietary vs client impact fits in a single figure.
- [`member_statistics.py`](../member_statistics.py): descriptive member/ISIN statistics and Plotly figures (requires trade-level per-ISIN Parquets).
- [`utils.py`](../utils.py): trade-schema mapping, realized-volatility estimators, sparse activity construction, and metaorder detection helpers.

Deep-dive documentation (rewritten under `docs/`):

- Power-law impact fits: [`docs/POWER_LAW_IMPACT_FITS.md`](POWER_LAW_IMPACT_FITS.md)
- Proprietary vs non-proprietary crowding: [`docs/prop_vs_nonprop.md`](prop_vs_nonprop.md)
- Metaorder distributions: [`docs/metaorder_distributions.md`](metaorder_distributions.md)

---

## Scientific goal / research questions

This project focuses on four related empirical questions:

1. **Concavity of impact vs size:** does average impact scale as a power law in size (square-root-like), or is a logarithmic dependence competitive?
2. **Heterogeneity by trading capacity:** how do impact curves differ between **proprietary** and **client** aggressive flow?
3. **Crowding within and across groups:** do metaorders trade with the prevailing signed imbalance within their group (herding), and how does this relate across groups (prop vs client, client vs prop)?
4. **Member-level prop–client crowding:** do proprietary metaorders at a given member co-move with that member’s client-side flow imbalance?

---

## Data requirements (not included in repo)

The raw trade data is proprietary and **not shipped** in this repository. The code expects to be able to build a canonical trade-level view with the schema required by `utils.build_trades_view`.

### Expected trade-level input schema

`utils.build_trades_view` requires the following columns (exact names):

- `MIC`
- `TRADING_DAY`
- `TRADETIME`
- `TRADED_QUANTITY`
- `TRADED_PRICE`
- `TRADED_AMOUNT`
- `COD_BUY`
- `CLIENT_IDENTIFIC_SHORT_CODE_BUY`
- `PASSIVE_ORDER_INDICATOR_BUY`
- `COD_SELL`
- `CLIENT_IDENTIFIC_SHORT_CODE_SELL`
- `TRADING_CAPACITY_BUY`
- `TRADING_CAPACITY_SELL`

In typical usage, numeric trade codes are first mapped to readable labels via `utils.map_trade_codes` (e.g., capacity codes mapped to strings such as `Dealing_on_own_account`).

### Trading-hours filter

All analyses use the continuous trading session filter:

- **09:30:00–17:30:00**

### Proprietary vs client classification

The core split is:

- **Proprietary flow:** `Trade Type Aggressive == "Dealing_on_own_account"`
- **Client (non-proprietary) flow:** all other aggressive capacities

Note: “client” here denotes non-proprietary aggressive flow in the dataset; it should not be interpreted as retail flow.

---

## Metaorder reconstruction (member/client level)

Metaorders are reconstructed from a time-ordered stream of aggressive trades, following the operational definition documented in:

- [`docs/metaorder_distributions.md`](metaorder_distributions.md)
- [`docs/POWER_LAW_IMPACT_FITS.md`](POWER_LAW_IMPACT_FITS.md)

### Definition (high level)

At a given `LEVEL` (member or client), for a fixed ISIN and day, a **metaorder** is:

- a **contiguous run** of trades by a single agent (member or client),
- with **constant aggressive sign** (buy or sell),
- constrained to a **single calendar day** and a **single client ID** for the run,
- **split** if inactivity gaps exceed `MAX_GAP = 1h`,
- filtered to keep runs with at least `MIN_TRADES = 5` and duration at least `SECONDS_FILTER = 120` seconds.

### Pipeline pseudocode

```text
for each ISIN:
  load and filter trades to 09:30–17:30
  select group (proprietary or client) and agent level (member/client)
  for each agent:
    build signed activity process (sparse → dense on the ISIN timeline)
    detect same-sign runs (candidate metaorders)
    enforce same-day and single-client constraints
    split each run when inter-trade gap > MAX_GAP
    filter by MIN_TRADES and SECONDS_FILTER
  save:
    metaorders_dict_all_{LEVEL}_{PROPRIETARY_TAG}.pkl
    metaorders_info_sameday_{...}.parquet (per-metaorder features)
```

---

## Per-metaorder variables and normalization

For each metaorder $i$, the scripts compute a standard set of quantities (notation aligned with `docs/POWER_LAW_IMPACT_FITS.md`):

- **Metaorder volume (shares):**
  $$
  Q_i = \sum_{j \in \mathcal{M}_i} q_j.
  $$
- **Daily traded volume (shares):**
  $$
  V_{d(i)} = \sum_{j \in \mathcal{J}_{d(i)}} q_j.
  $$
- **Relative size (primary regressor):**
  $$
  \phi_i \equiv \frac{Q_i}{V_{d(i)}} \quad \text{(stored as `Q/V`)}.
  $$
- **Participation rate over execution window:**
  $$
  \eta_i \equiv \frac{Q_i}{V_i^{\text{during}}} \quad \text{(stored as `Participation Rate`)}.
  $$
- **Signed log-price change over execution:**
  $$
  \Delta p_i = \log P_i^{\text{end}} - \log P_i^{\text{start}}, \qquad \varepsilon_i \in \{+1,-1\}.
  $$
- **Volatility-normalized impact:**
  $$
  I_i \equiv \frac{\varepsilon_i \Delta p_i}{\sigma_{d(i)}}.
  $$

### Configurable normalization (volume and volatility modes)

Two key normalizations are configurable in `config_ymls/metaorder_computation.yml`:

- `Q_V_DENOMINATOR_MODE ∈ {same_day, prev_day, avg_5d}` selects the denominator used for `Q/V`.
- `DAILY_VOL_MODE ∈ {same_day, prev_day, avg_5d}` selects the daily volatility used for impact normalization.

The daily volatility estimator is described in detail in [`docs/POWER_LAW_IMPACT_FITS.md`](POWER_LAW_IMPACT_FITS.md) (realized-kernel-based daily volatility built from resampled trade prices).

---

## Impact models and estimation

This repository implements a standard “metaorder impact curve” estimation workflow:

### Power-law model (log-binned WLS)

The baseline model is:
$$
\mathbb{E}[I \mid \phi] = Y\,\phi^{\gamma}.
$$

Estimation (implemented in `metaorder_computation.py`, described in [`docs/POWER_LAW_IMPACT_FITS.md`](POWER_LAW_IMPACT_FITS.md)):

1. Filter to valid observations (finite `Q/V`, finite daily vol, optional participation-rate cutoff).
2. Log-bin $\phi = Q/V$ into a fixed number of bins.
3. Compute per-bin mean impact $\bar I_b$ and its uncertainty (SEM).
4. Fit a **weighted least squares** regression in log space, using weights $w_b = 1/\mathrm{SEM}_b^2$.

### Logarithmic overlay model (brief)

For comparison, the code can also fit a logarithmic form on the same binned data (see [`docs/POWER_LAW_IMPACT_FITS.md`](POWER_LAW_IMPACT_FITS.md) for the specification and filters).

### Bivariate impact surface (brief)

Beyond the one-dimensional impact curve, the code exports a bivariate “impact surface” in $(Q/V, \eta)$, including:

- a 2D heatmap (PNG), and
- an interactive 3D surface (HTML, Plotly).

Read more: [`docs/POWER_LAW_IMPACT_FITS.md`](POWER_LAW_IMPACT_FITS.md).

---

## Crowding / imbalance framework

Crowding is studied using correlations between metaorder **direction** and **signed volume imbalance** measures computed over appropriate environments (ISIN–day, member–day, within-group, cross-group). The full methodological description lives in [`docs/prop_vs_nonprop.md`](prop_vs_nonprop.md).

### Imbalance definitions (core idea)

For a set of metaorders $\mathcal{S}$, define a signed-volume imbalance:
$$
\mathrm{imbalance}(\mathcal{S}) =
\frac{\sum_{j \in \mathcal{S}} Q_j \varepsilon_j}{\sum_{j \in \mathcal{S}} Q_j},
$$
with `NaN` if the denominator is zero.

The “local” within-group imbalance is computed **excluding the metaorder itself**, which avoids a trivial self-contribution but introduces a mechanical bias that must be interpreted carefully (see the log notes in `metaorder_statistics.log` and the discussion in [`docs/prop_vs_nonprop.md`](prop_vs_nonprop.md)).

### Analyses performed

`metaorder_statistics.py` runs:

- **within-group crowding:** prop vs prop, client vs client,
- **cross-group crowding:** prop direction vs client environment imbalance (and vice versa),
- **all-vs-all crowding:** each group vs the combined environment,
- **member-level prop–client crowding:** correlation of proprietary directions with the member-day client imbalance.

### Inference

Uncertainty and significance are computed using:

- non-parametric **bootstrap percentile** confidence intervals for correlations, and
- permutation-style **p-values** under sign–imbalance independence.

Note on determinism: the bootstrap/permutation draws use NumPy’s default RNG without a fixed seed, so repeated runs are not bitwise deterministic unless you modify the code to set a seed.

---

## Results artifacts included in this repository (ftsemib)

This repo includes a set of **already-produced** artifacts under `out_files/ftsemib/`, `images/ftsemib/`, and in the logs `metaorder_computation.log` and `metaorder_statistics.log`. The report below focuses on `ftsemib` because `mot` has no shipped figures in `images/mot/`.

### Latest snapshot (from logs)

The numbers below are copied from the committed log files to avoid fabricating results:

| Component | Group | Logged timestamp | Logged result |
|---|---|---:|---|
| Impact WLS (after `PR < 1.0`) | Proprietary | 2025-12-19 | $N$: 588,336 → 588,174; $\gamma = 0.366208 \pm 0.00828$ |
| Impact WLS (after `PR < 1.0`) | Client (non-proprietary) | 2025-12-19 | $N$: 255,140 → 255,104; $\gamma = 0.508227 \pm 0.0226$ |
| Daily crowding (mean corr, `n >= 100`) | Proprietary | 2025-12-17 | mean corr: 0.081 |
| Daily crowding (mean corr, `n >= 100`) | Client (non-proprietary) | 2025-12-17 | mean corr: 0.168 |
| Cross crowding (mean corr, `n >= 100`) | Prop vs client env | 2025-12-17 | mean corr: -0.018 |
| Cross crowding (mean corr, `n >= 100`) | Client vs prop env | 2025-12-17 | mean corr: -0.003 |
| All-vs-all crowding (mean corr, `n >= 100`) | Prop vs all | 2025-12-17 | mean corr: 0.041 |
| All-vs-all crowding (mean corr, `n >= 100`) | Client vs all | 2025-12-17 | mean corr: 0.112 |
| Member-level prop–client crowding | Per-member mean (threshold `n >= 30`) | 2025-12-17 | mean per-member corr: -0.003 |

---

## Figures

Figures are embedded by referencing the existing repository paths (relative to this `docs/` directory).

### Member structure

**Unique members per ISIN**

![Unique members per ISIN](../images/ftsemib/member_statistics/png/members_per_isin.png)

**Proprietary vs client aggressive trades per ISIN**

![Proprietary vs client trades per ISIN](../images/ftsemib/member_statistics/png/proprietary_vs_client_trades_per_isin.png)

**Member ISIN coverage**

![Member ISIN coverage per member](../images/ftsemib/member_statistics/png/member_isin_coverage_per_member.png)

**Member activity heatmap (trades by day)**

![Member activity heatmap](../images/ftsemib/member_statistics/png/member_activity_heatmap.png)

### Distributions + paths

**Metaorder duration distribution**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop duration distribution](../images/ftsemib/member_proprietary/metaorder_duration_all.png) | ![Client duration distribution](../images/ftsemib/member_non_proprietary/metaorder_duration_all.png) |

**Relative size $Q/V$**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop Q/V distribution](../images/ftsemib/member_proprietary/q_over_v_all.png) | ![Client Q/V distribution](../images/ftsemib/member_non_proprietary/q_over_v_all.png) |

**Participation rate $\eta = Q/V_{\text{window}}$**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop participation distribution](../images/ftsemib/member_proprietary/participation_rate_all.png) | ![Client participation distribution](../images/ftsemib/member_non_proprietary/participation_rate_all.png) |

**Normalized impact path (during execution + aftermath)**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop normalized impact path](../images/ftsemib/member_proprietary/normalized_impact_path_member_proprietary.png) | ![Client normalized impact path](../images/ftsemib/member_non_proprietary/normalized_impact_path_member_non_proprietary.png) |

### Impact fits

**Overall impact fits (log-binned WLS)**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop power-law fit overall](../images/ftsemib/member_proprietary/power_law_fit_overall_member.png) | ![Client power-law fit overall](../images/ftsemib/member_non_proprietary/power_law_fit_overall_member.png) |

**Overlay: proprietary vs client**

![Prop vs client impact overlay](../images/ftsemib/prop_vs_nonprop/power_law_prop_vs_nonprop.png)

### Impact surface (heatmaps + interactive HTML)

**Heatmaps**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop impact surface heatmap](../images/ftsemib/member_proprietary/impact_surface_qv_participation_heatmap_member_proprietary.png) | ![Client impact surface heatmap](../images/ftsemib/member_non_proprietary/impact_surface_qv_participation_heatmap_member_non_proprietary.png) |

**Interactive 3D surfaces (HTML links)**

These are standalone HTML files; open them locally in a browser for interactivity:

- Proprietary:
  - Surface: `images/ftsemib/member_proprietary/impact_surface_qv_participation_3d_surface_member_proprietary.html` (see `../images/ftsemib/member_proprietary/impact_surface_qv_participation_3d_surface_member_proprietary.html`)
  - Bivariate fits: `images/ftsemib/member_proprietary/impact_surface_qv_participation_3d_surface_bivariate_fits_member_proprietary.html` (see `../images/ftsemib/member_proprietary/impact_surface_qv_participation_3d_surface_bivariate_fits_member_proprietary.html`)
- Client (non-proprietary):
  - Surface: `images/ftsemib/member_non_proprietary/impact_surface_qv_participation_3d_surface_member_non_proprietary.html` (see `../images/ftsemib/member_non_proprietary/impact_surface_qv_participation_3d_surface_member_non_proprietary.html`)
  - Bivariate fits: `images/ftsemib/member_non_proprietary/impact_surface_qv_participation_3d_surface_bivariate_fits_member_non_proprietary.html` (see `../images/ftsemib/member_non_proprietary/impact_surface_qv_participation_3d_surface_bivariate_fits_member_non_proprietary.html`)
  - Bivariate WLS fit: `images/ftsemib/member_non_proprietary/impact_surface_qv_participation_3d_surface_bivariate_wls_fit_member_non_proprietary.html` (see `../images/ftsemib/member_non_proprietary/impact_surface_qv_participation_3d_surface_bivariate_wls_fit_member_non_proprietary.html`)

### Crowding

**Rolling correlations (5-day smoothing in the shipped artifact set)**

| Within-group | Cross-group | All-vs-all |
|---|---|---|
| ![Within-group rolling correlation](../images/ftsemib/prop_vs_nonprop/daily_crowding_rolling_5d.png) | ![Cross-group rolling correlation](../images/ftsemib/prop_vs_nonprop/cross_crowding_rolling_5d.png) | ![All-vs-all rolling correlation](../images/ftsemib/prop_vs_nonprop/all_vs_all_crowding_rolling_5d.png) |

**Diagnostics and member-level summaries**

- Imbalance distributions:
  ![Imbalance distribution](../images/ftsemib/prop_vs_nonprop/imbalance_distribution.png)
- Member-level prop–client crowding:

| Per-member correlations | Member × window heatmap |
|---|---|
| ![Member prop-client crowding per member](../images/ftsemib/prop_vs_nonprop/member_prop_client_crowding_hist.png) | ![Member prop-client crowding heatmap](../images/ftsemib/prop_vs_nonprop/member_prop_client_crowding_heatmap.png) |

Additional per-ISIN ACF diagnostics are available under:
`images/ftsemib/prop_vs_nonprop/acf/` (see `../images/ftsemib/prop_vs_nonprop/acf/`; one PNG per ISIN).

---

## How to run (reproducibility)

This repository is configuration-first: parameters and toggles are defined in YAML and then the scripts are run from the repo root.

### Main configuration files

- Impact + metaorder reconstruction: `config_ymls/metaorder_computation.yml`
- Crowding/imbalance analysis: `config_ymls/metaorder_statistics.yml`

### Typical workflow (conceptual)

1. Set `DATASET_NAME` and data roots in the YAML files (and ensure the expected input data is available under the configured `DATA_ROOT`).
2. Run `metaorder_computation.py` twice, toggling `PROPRIETARY`:
   - `PROPRIETARY: true` (prop)
   - `PROPRIETARY: false` (client / non-proprietary)
3. Run `metaorder_statistics.py` to generate crowding outputs and plots.
4. Optionally run `plot_prop_nonprop_fits.py` to regenerate the overlay figure.

Convenience wrapper:

- `metaorders_pipeline.sh` runs computation + statistics, but assumes a conda environment named `defi` (`conda activate defi`).

### Dependencies (inferred from imports)

Core scientific stack:

- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `tqdm`, `pyyaml`
- Parquet I/O: `pyarrow` (or another pandas-compatible Parquet engine)

For `member_statistics.py`:

- `plotly`
- an image export backend for `fig.write_image` (e.g., `kaleido`)

---

## Outputs map

The scripts write outputs under `out_files/` and `images/`, typically organized by `DATASET_NAME` and group tags.

| Artifact type | Typical path pattern | Produced by |
|---|---|---|
| Metaorder index dictionaries | `out_files/{DATASET_NAME}/metaorders_dict_all_{LEVEL}_{PROPRIETARY_TAG}.pkl` | `metaorder_computation.py` |
| Per-metaorder tables (unfiltered) | `out_files/{DATASET_NAME}/metaorders_info_sameday_{LEVEL}_{PROPRIETARY_TAG}.parquet` | `metaorder_computation.py` |
| Per-metaorder tables (filtered for fits) | `out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_{LEVEL}_{PROPRIETARY_TAG}.parquet` | `metaorder_computation.py` |
| Impact/distribution plots | `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/...` | `metaorder_computation.py`, `metaorder_statistics.py` |
| Crowding plots | `images/{DATASET_NAME}/prop_vs_nonprop/...` | `metaorder_statistics.py` |



## References

The following references are adapted from the bibliography embedded in `paper/main.tex`:

1. J.-P. Bouchaud, J. D. Farmer, and F. Lillo. *How markets slowly digest changes in supply and demand.* In T. Hens and K. R. Schenk-Hoppé (eds.), *Handbook of Financial Markets: Dynamics and Evolution*. Academic Press, 2008.
2. E. Moro et al. *Market impact and trading profile of hidden orders in stock markets.* *Physical Review E* **80**(6), 066102, 2009.
3. G. Vaglica, F. Lillo, and R. N. Mantegna. *Statistical identification with hidden Markov models of large order splitting strategies in an equity market.* *New Journal of Physics* **12**(7), 075031, 2010.
4. C. Gomes and H. Waelbroeck. *Is market impact a measure of the information value of trades? Market response to liquidity vs. informed metaorders.* *Quantitative Finance* **15**(5), 773–793, 2015.
5. E. Zarinelli, M. Treccani, J. D. Farmer, and F. Lillo. *Beyond the square root: Evidence for logarithmic dependence of market impact on size and participation rate.* *Market Microstructure and Liquidity* **1**(2):1550004, 2015.
6. F. Bucci, M. Benzaquen, F. Lillo, and J.-P. Bouchaud. *Slow decay of impact in equity markets: insights from the ANcerno database.* *Market Microstructure and Liquidity* **4**(3–4), 1950006, 2018.
7. M. Briere, C.-A. Lehalle, T. Nefedova, and A. Raboun. *Modelling transaction costs when trades may be crowded: a Bayesian network using partially observable orders imbalance.* Book chapter, 2019.
8. F. Bucci et al. *Co-impact: crowding effects in institutional trading activity.* *Quantitative Finance* **20**(2), 193–205, 2020.
9. F. Bucci, F. Lillo, J.-P. Bouchaud, and M. Benzaquen. *Are trading invariants really invariant? Trading costs matter.* *Quantitative Finance* **20**(7), 1059–1068, 2020.
10. Y. Sato and K. Kanazawa. *Does the square-root price impact law belong to the strict universal scalings? Quantitative support by a complete survey of the Tokyo Stock Exchange market.* Preprint, 2024.
11. G. Maitrier, G. Loeper, and J.-P. Bouchaud. *Generating realistic metaorders from public data.* arXiv:2503.18199, 2025.
12. M. Naviglio et al. *Why is the estimation of metaorder impact with public market data so challenging?* arXiv:2501.17096, 2025.
13. G. Guillaume. *The hidden complexity of price formation.* PhD thesis, 2025.
14. D. M. Di Nosse. *Metaorders_PriceImpact: Study of metaorder price impact using proprietary CONSOB data.* Repository, 2025. Origin remote: `git@github.com:DanieleMDiNosse/Metaorders_PriceImpact.git`.
