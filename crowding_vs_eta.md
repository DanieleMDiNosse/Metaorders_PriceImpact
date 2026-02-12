# Crowding vs Participation Rate (η)

## Introduction
Crowding (or *herding / co-impact*) is a central stylized fact in market microstructure: metaorders do not arrive in isolation, and the signed flow of other agents on the same instrument and day can amplify, dampen, or even reverse effective impact and execution conditions. A natural execution-state proxy is the **participation rate**,
$$
\eta_i \equiv \frac{Q_i}{V^{\mathrm{during}}_i},
$$
which measures how aggressively metaorder $i$ participates in the contemporaneous traded volume during its execution window.

This report studies how crowding varies with $\eta$ by conditioning crowding measures on **participation-rate quantile bins** (deciles). The guiding question is:

> Do high-$\eta$ metaorders exhibit stronger crowding (higher alignment with the prevailing imbalance) than low-$\eta$ metaorders?

The analysis is performed for **proprietary** and **client (non-proprietary)** metaorders, and for two environment definitions:

1. **Within-group (local) environment**: imbalance of *other metaorders of the same group* on the same $(\mathrm{ISIN}, \mathrm{Date})$, computed leave-one-out.
2. **Cross-group environment**: imbalance of the *other group* on the same $(\mathrm{ISIN}, \mathrm{Date})$.

The results shown here come from the `ftsemib` dataset over **251 trading days** (2024-06-03 to 2025-05-30), using:
- $\eta$ deciles (`eta_bins=10`) with **pooled quantile edges** (same bin boundaries for prop and client),
- `bootstrap_runs=1000`, `seed=0`,
- cluster bootstrap CIs using **Date** clustering (single panel per figure).

Reproducibility: see `out_files/ftsemib/crowding_vs_part_rate/run_manifest.json`.

## Definitions and Metrics
For a given environment imbalance value $\mathrm{imb}_i \in [-1,1]$ and metaorder direction $\epsilon_i \in \{+1,-1\}$:

- **Crowding intensity** (environment one-sidedness):
  $$
  |\mathrm{imb}_i|
  $$
- **Crowding alignment** (primary, bounded effect size):
  $$
  a_i \equiv \epsilon_i \cdot \mathrm{imb}_i \in [-1,1]
  $$
  Positive $a_i$ means metaorders trade *with* the environment; negative means *against* it.
- **Correlation crowding metric** (headline-style):
  $$
  \mathrm{Corr}(\epsilon_i,\mathrm{imb}_i)
  $$

Each figure plots these metrics **by $\eta$-bin**, separately for proprietary and client metaorders, with Date-cluster bootstrap CI bands (resampling trading days).

## Results: Within-Group (Local) Crowding vs η
In the local setting, $\mathrm{imb}_i$ is the leave-one-out signed-volume imbalance of **other metaorders of the same group** on the same $(\mathrm{ISIN}, \mathrm{Date})$. This is the canonical “co-impact” style definition: the metaorder is removed from its own environment to avoid trivial self-contamination.

### Plot 1: Mean Alignment vs η (Local)
![Local: mean alignment vs participation](images/ftsemib/crowding_vs_part_rate/curve_mean_align_vs_eta_local.png)

Interactive: [HTML](images/ftsemib/crowding_vs_part_rate/curve_mean_align_vs_eta_local.html)

**Commentary (microstructure interpretation).**
- **Proprietary flow:** alignment increases smoothly with $\eta$, consistent with stronger within-group herding for high-participation executions.
- **Client flow:** the effect is much stronger. Low-$\eta$ client metaorders are close to uncorrelated with the same-day client imbalance, while high-$\eta$ client metaorders are strongly aligned with it.

Quantitatively (top-decile minus bottom-decile $\Delta$ with bootstrap CI; Spearman trend with CI):

| group | cluster | top-bottom Δ | Spearman |
|---|---|---|---|
| prop | date | 0.0296 [0.0237, 0.0359] | 0.980 [0.939, 1.000] |
| client | date | 0.1890 [0.1680, 0.2087] | 0.960 [0.927, 0.964] |

### Plot 2: Mean |Imbalance| vs η (Local)
![Local: mean absolute imbalance vs participation](images/ftsemib/crowding_vs_part_rate/curve_mean_abs_imb_vs_eta_local.png)

Interactive: [HTML](images/ftsemib/crowding_vs_part_rate/curve_mean_abs_imb_vs_eta_local.html)

**Commentary.**
- For both groups, the environment becomes **more one-sided** as $\eta$ increases: high-participation metaorders tend to occur on $(\mathrm{ISIN},\mathrm{Date})$ where the rest of the group’s flow is more imbalanced.
- The client curve is higher in level (local client imbalance tends to be more extreme), but the incremental $\eta$-trend is present in both.

| group | cluster | top-bottom Δ | Spearman |
|---|---|---|---|
| prop | date | 0.0280 [0.0203, 0.0360] | 0.988 [0.952, 1.000] |
| client | date | 0.0253 [0.0164, 0.0346] | 0.798 [0.636, 0.915] |

### Plot 3: Corr(Direction, Imbalance) vs η (Local)
![Local: corr(Direction, imbalance) vs participation](images/ftsemib/crowding_vs_part_rate/curve_corr_dir_imb_vs_eta_local.png)

Interactive: [HTML](images/ftsemib/crowding_vs_part_rate/curve_corr_dir_imb_vs_eta_local.html)

**Commentary.**
- This plot mirrors Plot 1 but in correlation units. The monotone increase is clear for both groups.
- The client-side increase is large: the highest-$\eta$ decile exhibits strong positive correlation with the within-client imbalance, consistent with clustered directional demand (classic crowding/co-impact signature).

| group | cluster | top-bottom Δ | Spearman |
|---|---|---|---|
| prop | date | 0.0795 [0.0603, 0.0988] | 0.962 [0.891, 1.000] |
| client | date | 0.3860 [0.3498, 0.4184] | 0.955 [0.915, 0.964] |

## Results: Cross-Group Crowding vs η
In the cross setting, $\mathrm{imb}_i$ is computed from the *other group’s* signed imbalance on the same $(\mathrm{ISIN}, \mathrm{Date})$:
- for proprietary metaorders: $\mathrm{imb}_i = \mathrm{imbalance\_client\_env}$,
- for client metaorders: $\mathrm{imb}_i = \mathrm{imbalance\_prop\_env}$.

This is closer to an “exogenous environment” view: the metaorder is not mechanically part of the other group’s imbalance.

### Plot 4: Mean Alignment vs η (Cross-Group)
![Cross: mean alignment vs participation](images/ftsemib/crowding_vs_part_rate/curve_mean_align_vs_eta_cross.png)

Interactive: [HTML](images/ftsemib/crowding_vs_part_rate/curve_mean_align_vs_eta_cross.html)

**Commentary.**
- **Proprietary vs client environment:** mean alignment is **negative across $\eta$**, suggesting proprietary metaorders systematically trade *against* the contemporaneous client signed imbalance on the same $(\mathrm{ISIN},\mathrm{Date})$. The $\eta$-dependence (slope) is weak.
- **Client vs prop environment:** alignment becomes **more negative** as $\eta$ increases. High-$\eta$ client metaorders are increasingly *contrarian* to proprietary flow on the same day and instrument.

This pattern is consistent with a market-ecology interpretation in which one side (often proprietary) partly intermediates or offsets the other side’s directional demand, especially in periods where client executions are aggressive (high $\eta$).

| group | cluster | top-bottom Δ | Spearman |
|---|---|---|---|
| prop | date | -0.0057 [-0.0136, 0.0028] | -0.246 [-0.758, 0.406] |
| client | date | -0.0447 [-0.0551, -0.0358] | -0.878 [-0.927, -0.830] |

### Plot 5: Mean |Imbalance| vs η (Cross-Group)
![Cross: mean absolute imbalance vs participation](images/ftsemib/crowding_vs_part_rate/curve_mean_abs_imb_vs_eta_cross.png)

Interactive: [HTML](images/ftsemib/crowding_vs_part_rate/curve_mean_abs_imb_vs_eta_cross.html)

**Commentary.**
- High-$\eta$ metaorders occur when the *other group’s* environment is **more one-sided** (larger $|\mathrm{imb}|$). This is true for both directions (prop conditioned on client environment, and client conditioned on prop environment).
- Interpreting this requires care: the result can reflect “shared state variables” (market news days, volatility, correlated execution schedules) and/or strategic responses (agents scale participation when the other side’s flow is concentrated).

| group | cluster | top-bottom Δ | Spearman |
|---|---|---|---|
| prop | date | 0.0453 [0.0360, 0.0547] | 0.994 [0.976, 1.000] |
| client | date | 0.0244 [0.0183, 0.0305] | 0.921 [0.794, 0.988] |

### Plot 6: Corr(Direction, Imbalance) vs η (Cross-Group)
![Cross: corr(Direction, imbalance) vs participation](images/ftsemib/crowding_vs_part_rate/curve_corr_dir_imb_vs_eta_cross.png)

Interactive: [HTML](images/ftsemib/crowding_vs_part_rate/curve_corr_dir_imb_vs_eta_cross.html)

**Commentary.**
- **Proprietary metaorders vs client environment:** correlation is negative and only weakly dependent on $\eta$ (small, noisy slope).
- **Client metaorders vs proprietary environment:** the sign flips with $\eta$: low-$\eta$ is slightly positive, while high-$\eta$ is strongly negative. This is a sharp “inter-group contrarian” signature for aggressive client executions.

| group | cluster | top-bottom Δ | Spearman |
|---|---|---|---|
| prop | date | -0.0072 [-0.0242, 0.0108] | -0.063 [-0.636, 0.564] |
| client | date | -0.1739 [-0.2048, -0.1448] | -0.900 [-0.927, -0.842] |

## Detailed Discussion (Econophysics / Market Microstructure View)
### 1) Participation as an “execution state” correlates with crowded environments
Across both local and cross setups, **mean $|\mathrm{imb}|$** increases with $\eta$. This supports the idea that high-participation executions are typically carried out in regimes where aggregate flow is already concentrated in one direction (crowded state). In econophysics terms, $\eta$ behaves like a state variable that co-moves with “order-flow polarization”.

### 2) Within-group: higher η is strongly associated with herding (especially on the client side)
The local alignment and correlation curves are strongly increasing in $\eta$, with large effect sizes for clients. This is compatible with:
- clustered institutional demand (many client metaorders with aligned signs on the same day and instrument),
- execution scheduling around common information (macro/news), producing synchronized directional trading,
- endogenous feedback where a crowded local environment leads participants to trade more aggressively (higher $\eta$) to complete execution.

### 3) Cross-group: higher η corresponds to *more contrarian* inter-group interaction (client vs prop)
The cross-group results are qualitatively different:
- the environment becomes more polarized with $\eta$ (larger $|\mathrm{imb}|$),
- but alignment to the other group becomes **negative** at high $\eta$ for client metaorders.

This is consistent with a **risk-transfer / intermediation** narrative: when client executions are aggressive (high $\eta$), proprietary flow is more often on the opposite side (or, equivalently, client flow is opposite to proprietary imbalance). This is a microstructure-relevant signature because it suggests that crowding is not simply “everyone on the same side” across all participants; rather, the ecology can be segmented into groups that **co-move internally** but **offset each other**.

### 4) Robustness and limitations
- Clustering by (ISIN, Date) gives essentially the same inference as Date clustering; for simplicity we report Date clustering only.
- This report is non-parametric (binning). Participation $\eta$ correlates with size/speed proxies (e.g., `Q/V`, `Vt/V`), so a full causal reading requires conditioning on these covariates.
- The script attempted interaction regressions, but `statsmodels` was not available in the current environment (see `out_files/ftsemib/crowding_vs_part_rate/regression_results_*.csv`).

## Output Index
Plots:
- `images/ftsemib/crowding_vs_part_rate/curve_mean_align_vs_eta_local.png` (and `.html`)
- `images/ftsemib/crowding_vs_part_rate/curve_mean_abs_imb_vs_eta_local.png` (and `.html`)
- `images/ftsemib/crowding_vs_part_rate/curve_corr_dir_imb_vs_eta_local.png` (and `.html`)
- `images/ftsemib/crowding_vs_part_rate/curve_mean_align_vs_eta_cross.png` (and `.html`)
- `images/ftsemib/crowding_vs_part_rate/curve_mean_abs_imb_vs_eta_cross.png` (and `.html`)
- `images/ftsemib/crowding_vs_part_rate/curve_corr_dir_imb_vs_eta_cross.png` (and `.html`)

Tables:
- `out_files/ftsemib/crowding_vs_part_rate/bin_summary_prop_local.csv`
- `out_files/ftsemib/crowding_vs_part_rate/bin_summary_client_local.csv`
- `out_files/ftsemib/crowding_vs_part_rate/bin_summary_prop_cross.csv`
- `out_files/ftsemib/crowding_vs_part_rate/bin_summary_client_cross.csv`
- `out_files/ftsemib/crowding_vs_part_rate/daily_bin_panel_prop_local.csv`
- `out_files/ftsemib/crowding_vs_part_rate/daily_bin_panel_client_local.csv`
- `out_files/ftsemib/crowding_vs_part_rate/daily_bin_panel_prop_cross.csv`
- `out_files/ftsemib/crowding_vs_part_rate/daily_bin_panel_client_cross.csv`
- `out_files/ftsemib/crowding_vs_part_rate/effect_sizes_local.csv`
- `out_files/ftsemib/crowding_vs_part_rate/effect_sizes_cross.csv`
- `out_files/ftsemib/crowding_vs_part_rate/regression_results_local.csv` (skipped: missing statsmodels)
- `out_files/ftsemib/crowding_vs_part_rate/regression_results_cross.csv` (skipped: missing statsmodels)

## Appendix: Self-Check Plots (Not for Inference)
The folder `images/ftsemib/crowding_vs_part_rate/self_check_20260212_153424/` contains plots produced by running the pipeline on a very small subsample of dates (`--self-check`). These are **sanity checks** (to validate that the script runs end-to-end) and should not be interpreted economically.

<!-- ![Self-check: local mean alignment vs participation](images/ftsemib/crowding_vs_part_rate/self_check_20260212_153424/curve_mean_align_vs_eta_local.png)

![Self-check: local mean absolute imbalance vs participation](images/ftsemib/crowding_vs_part_rate/self_check_20260212_153424/curve_mean_abs_imb_vs_eta_local.png)

![Self-check: local corr(Direction, imbalance) vs participation](images/ftsemib/crowding_vs_part_rate/self_check_20260212_153424/curve_corr_dir_imb_vs_eta_local.png) -->
