# Distributions computed in `metaorder_computation.py`

This document explains the main random variables and distributions that are constructed and plotted in `metaorder_computation.py`, and links them to the corresponding figures already generated under `images/`.

Throughout, the script:

- identifies **metaorders** as consecutive same-sign runs of trades by the same agent (member or client), filtered to stay within a single day and single client;
- computes metaorder-level statistics (size, duration, participation, impact, etc.);
- aggregates these statistics across all ISINs and produces histograms and functional fits.

Unless stated otherwise, the examples below refer to `LEVEL="member"` and both `PROPRIETARY=False` (non‑proprietary flow) and `PROPRIETARY=True` (proprietary flow), which correspond to the folders:

- non‑proprietary: `images/member_non_proprietary`
- proprietary: `images/member_proprietary`

## Metaorders per member

For each member (broker) the script counts how many metaorders they execute across all ISINs:

- Random variable: for each member \(m\), the **metaorder count**
  \[
  N_m = \text{number of metaorders initiated by member } m.
  \]
- Distribution: histogram over all members of \(N_m\).

Figures:

- Non‑proprietary: ![Metaorders per member (non‑prop)](images/member_non_proprietary/metaorders_per_member_all.png)
- Proprietary: ![Metaorders per member (prop)](images/member_proprietary/metaorders_per_member_all.png)

## Metaorder duration

For each metaorder, defined by its first and last child trade indices \((s, e)\), with corresponding times \(t_s\) and \(t_e\), the script computes its **duration**
\[
T = (t_e - t_s) \quad \text{(in seconds)}.
\]

The histogram is plotted in minutes \(T/60\), with a log‑scale on the y‑axis:

- Random variable: metaorder duration \(T\).
- Distribution: density of \(T/60\) across all metaorders and ISINs.

Figures:

- Non‑proprietary: ![Metaorder duration (non‑prop)](images/member_non_proprietary/metaorder_duration_all.png)
- Proprietary: ![Metaorder duration (prop)](images/member_proprietary/metaorder_duration_all.png)

## Inter‑arrival times within a metaorder

Within each metaorder, the script also looks at the gaps between consecutive child trades. For a metaorder with trade times \((t_1, t_2, \dots, t_k)\), it computes the **inter‑arrival times**
\[
\Delta t_i = t_{i+1} - t_i, \quad i = 1,\dots,k-1,
\]
and aggregates all such gaps over all metaorders and ISINs. The histogram is plotted in minutes \(\Delta t/60\) (log‑scale on the y‑axis).

- Random variable: inter‑arrival time between consecutive trades inside a metaorder.
- Distribution: density of \(\Delta t/60\) across all child‑trade gaps.

Figures:

- Non‑proprietary: ![Inter‑arrival (non‑prop)](images/member_non_proprietary/interarrival_all.png)
- Proprietary: ![Inter‑arrival (prop)](images/member_proprietary/interarrival_all.png)

## Metaorder volumes

For each metaorder, the **metaorder volume** \(Q\) is defined as the sum of traded quantities across all its child trades:
\[
Q = \sum_{i \in \text{metaorder}} \left( \text{Total Quantity Buy}_i + \text{Total Quantity Sell}_i \right).
\]

This is a purely volume‑based notion (no notion of notional here). The script collects all \(Q\) across ISINs and plots their empirical distribution:

- Random variable: metaorder volume \(Q\) (in units of shares).
- Distribution: density of \(Q\) across all metaorders.

Figures:

- Non‑proprietary: ![Metaorder volumes (non‑prop)](images/member_non_proprietary/volumes_all.png)
- Proprietary: ![Metaorder volumes (prop)](images/member_proprietary/volumes_all.png)

## Relative size: \(Q/V\)

To measure how large a metaorder is relative to the day’s activity in a given ISIN, the script computes:

- the **daily traded volume** \(V_{\text{day}}\) for each date as
  \[
  V_{\text{day}} = \sum_{\text{that day}} \left( \text{Total Quantity Buy} + \text{Total Quantity Sell} \right);
  \]
- for each metaorder starting on date \(d\) with volume \(Q\), the **relative size**
  \[
  \frac{Q}{V_{\text{day}}}.
  \]

The histogram is plotted in percent \(100 \times Q/V_{\text{day}}\) (log‑scale on the y‑axis):

- Random variable: relative size \(Q/V_{\text{day}}\).
- Distribution: density of \(100 \times Q/V_{\text{day}}\) across metaorders.

Figures:

- Non‑proprietary: ![Q/V (non‑prop)](images/member_non_proprietary/q_over_v_all.png)
- Proprietary: ![Q/V (prop)](images/member_proprietary/q_over_v_all.png)

## Participation rate within the execution window

For each metaorder, the script compares its volume to the total volume traded in the same time window \([t_s, t_e]\). Let

- \(Q\) be the metaorder volume (as above),
- \(V_{\text{window}}\) be the total market volume (buy + sell) between \(t_s\) and \(t_e\).

Then the **participation rate** is
\[
\eta = \frac{Q}{V_{\text{window}}}.
\]

The histogram is plotted for \(100 \times \eta\) (in percent), with log‑scale on the y‑axis:

- Random variable: participation rate \(\eta\).
- Distribution: density of \(100 \times \eta\) across metaorders.

Figures:

- Non‑proprietary: ![Participation rate (non‑prop)](images/member_non_proprietary/participation_rate_all.png)
- Proprietary: ![Participation rate (prop)](images/member_proprietary/participation_rate_all.png)

## Volatility signature plots

In the **signature plots** section the script studies how different volatility estimators behave as a function of the sampling interval \(\Delta\) (in seconds), for each ISIN separately.

For a given ISIN and sampling interval \(\Delta\):

1. It resamples the trade‑time price series at interval \(\Delta\), computes log‑returns for each day, and then computes per‑day volatility estimators:
   - realized variance (RV),
   - bipower variation (BPV),
   - realized kernel (RK).
2. For each \(\Delta\), it aggregates the daily values across days and computes:
   - the mean,
   - the standard error of the mean (SE).

The signature plot then displays, for each estimator, the mean and \( \pm 2 \,\text{SE} \) as a function of \(\Delta\):

- Random variable: daily volatility estimates (RV, BPV, RK) at a given sampling interval.
- Distribution: empirical distribution of those estimates across days; the plots summarize this by mean \(\pm 2\text{SE}\) versus \(\Delta\).

Example figures (non‑proprietary; similar ones exist under `member_proprietary` for the same ISINs):

- ![Signature plot (example ISIN, non‑prop)](images/member_non_proprietary/signature_plots/signature_plot_IT0000066123.png)
- ![Signature plot (same ISIN, prop)](images/member_proprietary/signature_plots/signature_plot_IT0000066123.png)

Each figure has three panels:

- left: realized variance vs \(\Delta\),
- middle: bipower variation vs \(\Delta\),
- right: realized kernel vs \(\Delta\).

## Impact vs size: power‑law fits

Finally, in the **WLS fits** section, the script studies the relationship between price impact and relative size \(Q/V\).

For each metaorder, it constructs:

- **Daily volatility** \(\sigma_{\text{day}}\): from `build_daily_cache`, based on a realized‑kernel estimator on 120‑second log‑returns.
- **Signed log‑return over the metaorder**:
  \[
  \Delta p = \log P_{\text{end}} - \log P_{\text{start}},
  \]
  where \(P_{\text{start}}\) and \(P_{\text{end}}\) are the first and last contract prices of the metaorder.
- **Direction** \(s \in \{+1, -1\}\) (buy or sell).
- **Normalized impact**
  \[
  I = \frac{s \,\Delta p}{\sigma_{\text{day}}}.
  \]
- **Relative size** \(Q/V_{\text{day}}\) as above.

The script then:

1. **Filters** metaorders (minimum duration, minimum \(Q/V\), finite values).
2. **Bins** \(\log_{10}(Q/V)\) into logarithmic bins; in each bin it computes the mean impact and its standard error.
3. **Fits** a weighted least‑squares regression in log‑space to the model
   \[
   \frac{I}{\sigma_{\text{day}}} \approx Y \left(\frac{Q}{V_{\text{day}}}\right)^{\gamma},
   \]
   where \(Y\) and \(\gamma\) are estimated together with their standard errors and goodness‑of‑fit measures (\(R^2\) in log‑ and linear space).

The resulting empirical relationship and its binned points are shown in:

- Non‑proprietary: ![Power‑law fit (non‑prop)](images/member_non_proprietary/power_law_fit_overall_member.png)
- Proprietary: ![Power‑law fit (prop)](images/member_proprietary/power_law_fit_overall_member.png)

The script also conditions on **participation rate** by splitting metaorders into quantiles of \(\eta\), repeating the same log‑binning and WLS fit within each quantile. This yields:

- Non‑proprietary: ![Power‑law by participation (non‑prop)](images/member_non_proprietary/power_law_fits_by_participation_rate_member.png)
- Proprietary: ![Power‑law by participation (prop)](images/member_proprietary/power_law_fits_by_participation_rate_member.png)

These figures summarize how the distribution of impact conditional on size \(Q/V\) changes as a function of participation rate.
