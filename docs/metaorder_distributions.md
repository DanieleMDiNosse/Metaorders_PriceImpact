# Distributions computed from the metaorder dictionary

The plots described here are produced by `run_metaorder_dict_statistics` in `scripts/metaorder_statistics.py` (enable `RUN_METAORDER_DICT_STATS=True`). They consume the metaorder indices saved by `scripts/metaorder_computation.py` in `out_files/{DATASET_NAME}/metaorders_dict_all_{LEVEL}_{PROPRIETARY_TAG}.pkl` together with the per-ISIN parquet tapes from `PARQUET_PATH` (default `data/parquet/`).

Metaorders are detected in `scripts/metaorder_computation.py` as contiguous same-sign runs for a single member or client, within one trading day and a single client ID. Runs are split when inter-trade gaps exceed `MAX_GAP=1h`, then filtered to keep at least `MIN_TRADES=5` trades and `SECONDS_FILTER=120` seconds of duration. Only trades in the continuous session 09:30:00-17:30:00 are used. Figures are written under `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/` and `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/html/` (for example, `images/ftsemib/member_proprietary/png/`).

## Latest run summary (ftsemib, `pipeline_20260220_190521`)

Source logs:

- Proprietary (member): `out_files/ftsemib/logs/pipeline_20260220_190521/metaorder_statistics_prop_true_nat_all.log`
- Client / non-proprietary (member): `out_files/ftsemib/logs/pipeline_20260220_190521/metaorder_statistics_prop_false_nat_all.log`

**Metaorder counts (member-level).**

| Group | Total metaorders | Buy | Sell |
|---|---:|---:|---:|
| Proprietary | 588,543 | 279,992 | 308,551 |
| Client (non-proprietary) | 256,371 | 129,919 | 126,452 |

**Quantiles (median [p10, p90]) on `metaorders_info_sameday_member_*.parquet`.**

| Proprietary slice | n | Q | Q/V | Participation Rate ($\\eta$) | N Child | Vt/V |
|---|---:|---:|---:|---:|---:|---:|
| all | 588,543 | 6,519 [819, 48,358] | 0.001818 [0.000405, 0.007979] | 0.09495 [0.02321, 0.3032] | 7 [5, 16] | 0.01966 [0.004697, 0.09793] |
| foreign | 586,450 | 6,509 [817, 48,254] | 0.002280 [0.000513, 0.009934] | 0.1202 [0.02972, 0.3716] | 7 [5, 16] | 0.01971 [0.004641, 0.09777] |
| IT | 2,093 | 10,950 [1,826, 67,910] | 0.007285 [0.001498, 0.07040] | 0.09199 [0.009234, 0.6997] | 7 [5, 19] | 0.1157 [0.02451, 0.4091] |

| Client (non-proprietary) slice | n | Q | Q/V | Participation Rate ($\\eta$) | N Child | Vt/V |
|---|---:|---:|---:|---:|---:|---:|
| all | 256,371 | 7,172 [744, 60,486] | 0.001814 [0.000312, 0.01013] | 0.04990 [0.006792, 0.2024] | 9 [5, 27] | 0.04360 [0.008337, 0.2036] |
| foreign | 232,260 | 7,060 [709, 60,004] | 0.002387 [0.000408, 0.01300] | 0.06707 [0.01002, 0.2548] | 9 [5, 29] | 0.04218 [0.008016, 0.2022] |
| IT | 24,111 | 8,131 [1,250, 66,122] | 0.005183 [0.001079, 0.03202] | 0.08587 [0.01413, 0.6562] | 6 [5, 15] | 0.06493 [0.01345, 0.2547] |

### Plots (PNG)

All plots below have interactive HTML counterparts in the matching `images/ftsemib/.../html/` folders.

**Member nationality share (known only).**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop nationality share](../images/ftsemib/member_proprietary/png/nationality_share_proprietary.png) | ![Client nationality share](../images/ftsemib/member_non_proprietary/png/nationality_share_non_proprietary.png) |

**Metaorders per member.**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop metaorders per member](../images/ftsemib/member_proprietary/png/metaorders_per_member_all.png) | ![Client metaorders per member](../images/ftsemib/member_non_proprietary/png/metaorders_per_member_all.png) |

**Metaorder duration.**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop duration](../images/ftsemib/member_proprietary/png/metaorder_duration_all.png) | ![Client duration](../images/ftsemib/member_non_proprietary/png/metaorder_duration_all.png) |

**Inter-arrival times within metaorders.**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop inter-arrival](../images/ftsemib/member_proprietary/png/interarrival_all.png) | ![Client inter-arrival](../images/ftsemib/member_non_proprietary/png/interarrival_all.png) |

**Metaorder volume (Q).**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop volumes](../images/ftsemib/member_proprietary/png/volumes_all.png) | ![Client volumes](../images/ftsemib/member_non_proprietary/png/volumes_all.png) |

**Relative size (Q/V).**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop Q/V](../images/ftsemib/member_proprietary/png/q_over_v_all.png) | ![Client Q/V](../images/ftsemib/member_non_proprietary/png/q_over_v_all.png) |

**Participation rate ($\\eta = Q/V_{\\mathrm{window}}$).**

| Proprietary | Client (non-proprietary) |
|---|---|
| ![Prop participation rate](../images/ftsemib/member_proprietary/png/participation_rate_all.png) | ![Client participation rate](../images/ftsemib/member_non_proprietary/png/participation_rate_all.png) |

**Mean daily metaorder-volume share (shared comparison).**

This stacked bar is written once per run under `images/{DATASET_NAME}/prop_vs_nonprop/png/mean_daily_metaorder_volume_share.png` (with the matching HTML in `.../html/`). It shows one bar per ISIN, where each bar is the arithmetic mean across trading days of
$$
\frac{\text{daily proprietary metaorder volume} + \text{daily client metaorder volume}}{\text{total daily market volume for that ISIN}},
$$
with proprietary and client contributions highlighted as separate colored segments.

## Member nationality (IT vs foreign)

When `LEVEL="member"`, each metaorder can be assigned the nationality of the executing broker (member) using `data/members_nationality.parquet`. In the `ftsemib` runs, the nationality split is extremely skewed toward **foreign** brokers, especially for **proprietary** metaorders:

| Metaorder type (`PROPRIETARY_TAG`) | Italian brokers (IT) | Foreign brokers | Total metaorders (known) |
| --- | --- | --- | --- |
| `proprietary` | 2,093 (0.36%) | 586,450 (99.64%) | 588,543 |
| `non_proprietary` | 24,111 (9.40%) | 232,260 (90.60%) | 256,371 |

Implications for interpretation:

- Italian brokers appear to execute **almost no proprietary metaorders** in this dataset (only 0.36% at the member level), suggesting they rarely trade in a metaorder-like manner for their own account.
- As a consequence, **most aggregate distributions, diagnostics, and impact results produced by this repository are effectively driven by foreign broker executions**, both for proprietary and non-proprietary flow.

Figures: `images/ftsemib/member_proprietary/png/nationality_share_proprietary.png` and `images/ftsemib/member_non_proprietary/png/nationality_share_non_proprietary.png`.
Source: `out_files/ftsemib/logs/pipeline_20260220_190521/metaorder_statistics_prop_true_nat_all.log` and `out_files/ftsemib/logs/pipeline_20260220_190521/metaorder_statistics_prop_false_nat_all.log` (run on 2026-02-20).

## Metaorders per member

For each member (broker) the script counts how many metaorders they execute across all ISINs:

- Random variable: for each member $m$, the **metaorder count**
  $$
  N_m = \text{number of metaorders initiated by member } m.
  $$
- Distribution: histogram over all members of $N_m$.

Figure: `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/metaorders_per_member_all.png`.

## Metaorder duration

For each metaorder, defined by its first and last child trade indices $(s, e)$, with corresponding times $t_s$ and $t_e$, the script computes its **duration**
$$
T = (t_e - t_s) \quad \text{(in seconds)}.
$$

The histogram is plotted in minutes $T/60$, with a log-scale on the y-axis:

- Random variable: metaorder duration $T$.
- Distribution: density of $T/60$ across all metaorders and ISINs.

Figure: `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/metaorder_duration_all.png`.

## Inter-arrival times within a metaorder

Within each metaorder, the script also looks at the gaps between consecutive child trades. For a metaorder with trade times $(t_1, t_2, \dots, t_k)$, it computes the **inter-arrival times**
$$
\Delta t_i = t_{i+1} - t_i, \quad i = 1,\dots,k-1,
$$
and aggregates all such gaps over all metaorders and ISINs. The histogram is plotted in minutes $\Delta t/60$ (log-scale on the y-axis).

- Random variable: inter-arrival time between consecutive trades inside a metaorder.
- Distribution: density of $\Delta t/60$ across child-trade gaps.

Figure: `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/interarrival_all.png`.

## Metaorder volumes

For each metaorder, the **metaorder volume** $Q$ is defined as the sum of traded quantities across all its child trades:
$$
Q = \sum_{i \in \text{metaorder}} \left( \text{Total Quantity Buy}_i + \text{Total Quantity Sell}_i \right).
$$

This is a volume-based notion (no notion of notional). The script collects all $Q$ across ISINs and plots their empirical distribution.

- Random variable: metaorder volume $Q$ (in units of shares).
- Distribution: density of $Q$ across all metaorders.

Figure: `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/volumes_all.png`.

## Relative size: $Q/V$

To measure how large a metaorder is relative to the day's activity in a given ISIN, the script computes:

- the **daily traded volume** $V_{\text{day}}$ for each date as
  $$
  V_{\text{day}} = \sum_{\text{that day}} \left( \text{Total Quantity Buy} + \text{Total Quantity Sell} \right);
  $$
- for each metaorder starting on date $d$ with volume $Q$, the **relative size**
  $$
  \frac{Q}{V_{\text{day}}}.
  $$

The histogram is plotted in percent $100 \times Q/V_{\text{day}}$ (log-scale on the y-axis):

- Random variable: relative size $Q/V_{\text{day}}$.
- Distribution: density of $100 \times Q/V_{\text{day}}$ across metaorders.

Figure: `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/q_over_v_all.png`.

## Participation rate within the execution window

For each metaorder, the script compares its volume to the total volume traded in the same time window $[t_s, t_e]$. Let

- $Q$ be the metaorder volume (as above),
- $V_{\text{window}}$ be the total market volume (buy + sell) between $t_s$ and $t_e$.

Then the **participation rate** is
$$
\eta = \frac{Q}{V_{\text{window}}}.
$$

The histogram is plotted for $100 \times \eta$ (in percent), with log-scale on the y-axis:

- Random variable: participation rate $\eta$.
- Distribution: density of $100 \times \eta$ across metaorders.

Figure: `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/participation_rate_all.png`.

## Mean daily metaorder-volume share by ISIN

To summarize how much of an ISIN's daily activity is attributable to detected metaorders, the script also builds a day-level ratio for each instrument separately. For each ISIN $k$ and trading day $d$, define:

- the **total market volume for that ISIN**
  $$
  V_{k,d} = \sum_{j \in \mathcal{J}_{k,d}}
  \left( \text{Total Quantity Buy}_j + \text{Total Quantity Sell}_j \right),
  $$
- the **proprietary metaorder volume for that ISIN**
  $$
  Q_{k,d}^{\text{prop}} = \sum_{i \in \mathcal{M}^{\text{prop}}_{k,d}} Q_i,
  $$
- the **client metaorder volume for that ISIN**
  $$
  Q_{k,d}^{\text{client}} = \sum_{i \in \mathcal{M}^{\text{client}}_{k,d}} Q_i.
  $$

The daily shares are then
$$
r_{k,d}^{\text{prop}} = \frac{Q_{k,d}^{\text{prop}}}{V_{k,d}}, \qquad
r_{k,d}^{\text{client}} = \frac{Q_{k,d}^{\text{client}}}{V_{k,d}}, \qquad
r_{k,d}^{\text{total}} = r_{k,d}^{\text{prop}} + r_{k,d}^{\text{client}}.
$$

For each ISIN, the plotted stacked bar reports the arithmetic means of $r_{k,d}^{\text{prop}}$ and $r_{k,d}^{\text{client}}$ across trading days, so the full bar height equals the mean of $r_{k,d}^{\text{total}}$ for that ISIN.

Figure: `images/{DATASET_NAME}/prop_vs_nonprop/png/mean_daily_metaorder_volume_share.png`.

## Volatility signature plots

`scripts/metaorder_computation.py` can generate **volatility signature plots** per ISIN when `RUN_SIGNATURE_PLOTS=True`:
Optionally, set `N_SIGNATURE_PLOTS` to an integer to limit the run to the first N ISINs.

1. Resample the trade-time price series at interval $\Delta$, compute log-returns for each day, and compute per-day volatility estimators:
   - realized variance (RV),
   - bipower variation (BPV),
   - realized kernel (RK).
2. For each $\Delta$, aggregate daily values across days and compute the mean and the standard error of the mean.

Each figure shows mean $\pm 2\,\text{SE}$ versus $\Delta$ for the three estimators. Outputs live under `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/signature_plots/` (both `.png` and `.html` files are saved in this folder).

## Normalized impact paths

When `COMPUTE_IMPACT_PATHS=True` (default `False` in the script), `scripts/metaorder_computation.py` stores compact partial and aftermath impact paths for each metaorder (packed float32 bytes). `plot_normalized_impact_path`, triggered by `RUN_IMPACT_PATH_PLOT=True`, unpacks and interpolates these paths on a common grid to show:

- how impact accumulates during execution, and
- how it relaxes or overshoots after completion (up to `AFTERMATH_DURATION_MULTIPLIER` times the duration, with `AFTERMATH_NUM_SAMPLES` evenly spaced samples).

Figure: `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/normalized_impact_path_{LEVEL}_{PROPRIETARY_TAG}.png`.

## Impact vs size: power-law and logarithmic fits

The WLS section of `scripts/metaorder_computation.py` studies the relationship between price impact and relative size $Q/V$. Key steps:

1. Build `metaorders_info_sameday_{LEVEL}_{PROPRIETARY_TAG}.parquet`, then filter to `Q/V > MIN_QV` (default `1e-5`), drop non-finite values, and compute `Impact = Price Change * Direction / Daily Vol`. The `Q/V` denominator can be chosen via `Q_V_DENOMINATOR_MODE` (`same_day`, `prev_day`, `avg_5d`, default `avg_5d`), and the daily volatility via `DAILY_VOL_MODE` (`same_day`, `prev_day`, `avg_5d`, default `avg_5d`).
2. Log-bin $Q/V$ into `n_logbins=30` bins, keep bins with at least `min_count=20` metaorders and finite standard errors, and compute mean impacts and SEMs per bin.
3. Fit a weighted least-squares regression in log space to the **power-law** model
   $$
   \frac{I}{\sigma_{\text{day}}} \approx Y \left(\frac{Q}{V_{\text{day}}}\right)^{\gamma},
   $$
   optionally with additional bin-level controls (`fit_power_law_logbins_wls_new`).
4. Using the same bins, optionally fit a **logarithmic** model
   $$
   \frac{I}{\sigma_{\text{day}}} \approx a \,\log_{10}\!\bigl(1 + b\,Q/V_{\text{day}}\bigr).
   $$

Outputs:

- Overall fits: `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/power_law_fit_overall_{LEVEL}.png` (power-law plus logarithmic overlay).
- Fits conditioned on participation quantiles: `images/{DATASET_NAME}/{LEVEL}_{PROPRIETARY_TAG}/png/power_law_fits_by_participation_rate_{LEVEL}.png`.

These figures summarise how impact scales with size across the detected metaorders.
