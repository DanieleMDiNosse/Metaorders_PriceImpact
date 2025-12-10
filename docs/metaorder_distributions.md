# Distributions computed from the metaorder dictionary

The plots described here are produced by `run_metaorder_dict_statistics` in `metaorder_statistics.py` (enable `RUN_METAORDER_DICT_STATS=True`). They consume the metaorder indices saved by `metaorder_computation.py` in `out_files/metaorders_dict_all_{LEVEL}_{PROPRIETARY_TAG}.pkl` together with the per-ISIN parquet tapes in `data/`.

Metaorders are detected in `metaorder_computation.py` as contiguous same-sign runs for a single member or client, within one trading day and a single client ID. Runs are split when inter-trade gaps exceed `MAX_GAP=1h`, then filtered to keep at least `MIN_TRADES=5` trades and `SECONDS_FILTER=120` seconds of duration. Only trades in the continuous session 09:30:00-17:30:00 are used. Figures are written under `images/{LEVEL}_{PROPRIETARY_TAG}/` (for example, `images/member_proprietary/` when `LEVEL="member"` and `PROPRIETARY=True`).

## Metaorders per member

For each member (broker) the script counts how many metaorders they execute across all ISINs:

- Random variable: for each member \(m\), the **metaorder count**
  \[
  N_m = \text{number of metaorders initiated by member } m.
  \]
- Distribution: histogram over all members of \(N_m\).

Figure: `images/{LEVEL}_{PROPRIETARY_TAG}/metaorders_per_member_all.png`.

## Metaorder duration

For each metaorder, defined by its first and last child trade indices \((s, e)\), with corresponding times \(t_s\) and \(t_e\), the script computes its **duration**
\[
T = (t_e - t_s) \quad \text{(in seconds)}.
\]

The histogram is plotted in minutes \(T/60\), with a log-scale on the y-axis:

- Random variable: metaorder duration \(T\).
- Distribution: density of \(T/60\) across all metaorders and ISINs.

Figure: `images/{LEVEL}_{PROPRIETARY_TAG}/metaorder_duration_all.png`.

## Inter-arrival times within a metaorder

Within each metaorder, the script also looks at the gaps between consecutive child trades. For a metaorder with trade times \((t_1, t_2, \dots, t_k)\), it computes the **inter-arrival times**
\[
\Delta t_i = t_{i+1} - t_i, \quad i = 1,\dots,k-1,
\]
and aggregates all such gaps over all metaorders and ISINs. The histogram is plotted in minutes \(\Delta t/60\) (log-scale on the y-axis).

- Random variable: inter-arrival time between consecutive trades inside a metaorder.
- Distribution: density of \(\Delta t/60\) across child-trade gaps.

Figure: `images/{LEVEL}_{PROPRIETARY_TAG}/interarrival_all.png`.

## Metaorder volumes

For each metaorder, the **metaorder volume** \(Q\) is defined as the sum of traded quantities across all its child trades:
\[
Q = \sum_{i \in \text{metaorder}} \left( \text{Total Quantity Buy}_i + \text{Total Quantity Sell}_i \right).
\]

This is a volume-based notion (no notion of notional). The script collects all \(Q\) across ISINs and plots their empirical distribution.

- Random variable: metaorder volume \(Q\) (in units of shares).
- Distribution: density of \(Q\) across all metaorders.

Figure: `images/{LEVEL}_{PROPRIETARY_TAG}/volumes_all.png`.

## Relative size: \(Q/V\)

To measure how large a metaorder is relative to the day's activity in a given ISIN, the script computes:

- the **daily traded volume** \(V_{\text{day}}\) for each date as
  \[
  V_{\text{day}} = \sum_{\text{that day}} \left( \text{Total Quantity Buy} + \text{Total Quantity Sell} \right);
  \]
- for each metaorder starting on date \(d\) with volume \(Q\), the **relative size**
  \[
  \frac{Q}{V_{\text{day}}}.
  \]

The histogram is plotted in percent \(100 \times Q/V_{\text{day}}\) (log-scale on the y-axis):

- Random variable: relative size \(Q/V_{\text{day}}\).
- Distribution: density of \(100 \times Q/V_{\text{day}}\) across metaorders.

Figure: `images/{LEVEL}_{PROPRIETARY_TAG}/q_over_v_all.png`.

## Participation rate within the execution window

For each metaorder, the script compares its volume to the total volume traded in the same time window \([t_s, t_e]\). Let

- \(Q\) be the metaorder volume (as above),
- \(V_{\text{window}}\) be the total market volume (buy + sell) between \(t_s\) and \(t_e\).

Then the **participation rate** is
\[
\eta = \frac{Q}{V_{\text{window}}}.
\]

The histogram is plotted for \(100 \times \eta\) (in percent), with log-scale on the y-axis:

- Random variable: participation rate \(\eta\).
- Distribution: density of \(100 \times \eta\) across metaorders.

Figure: `images/{LEVEL}_{PROPRIETARY_TAG}/participation_rate_all.png`.

## Volatility signature plots

`metaorder_computation.py` can generate **volatility signature plots** per ISIN when `RUN_SIGNATURE_PLOTS=True`:

1. Resample the trade-time price series at interval \(\Delta\), compute log-returns for each day, and compute per-day volatility estimators:
   - realized variance (RV),
   - bipower variation (BPV),
   - realized kernel (RK).
2. For each \(\Delta\), aggregate daily values across days and compute the mean and the standard error of the mean.

Each figure shows mean \(\pm 2\,\text{SE}\) versus \(\Delta\) for the three estimators. Outputs live under `images/{LEVEL}_{PROPRIETARY_TAG}/signature_plots/`.

## Normalized impact paths

When `COMPUTE_IMPACT_PATHS=True` (default), `metaorder_computation.py` stores compact partial and aftermath impact paths for each metaorder (packed float32 bytes). `plot_normalized_impact_path`, triggered by `RUN_IMPACT_PATH_PLOT=True`, unpacks and interpolates these paths on a common grid to show:

- how impact accumulates during execution, and
- how it relaxes or overshoots after completion (up to `AFTERMATH_DURATION_MULTIPLIER` times the duration, with `AFTERMATH_NUM_SAMPLES` evenly spaced samples).

Figure: `images/{LEVEL}_{PROPRIETARY_TAG}/normalized_impact_path_{LEVEL}_{PROPRIETARY_TAG}.png`.

## Impact vs size: power-law and logarithmic fits

The WLS section of `metaorder_computation.py` studies the relationship between price impact and relative size \(Q/V\). Key steps:

1. Build `metaorders_info_sameday_{LEVEL}_{PROPRIETARY_TAG}.parquet`, then filter to `Q/V > MIN_QV` (default `1e-5`), drop non-finite values, and compute `Impact = Price Change * Direction / Daily Vol`. The `Q/V` denominator can be chosen via `Q_V_DENOMINATOR_MODE` (`same_day`, `prev_day`, `avg_5d`, default `avg_5d`).
2. Log-bin \(Q/V\) into `n_logbins=30` bins, keep bins with at least `min_count=20` metaorders and finite standard errors, and compute mean impacts and SEMs per bin.
3. Fit a weighted least-squares regression in log space to the **power-law** model
   \[
   \frac{I}{\sigma_{\text{day}}} \approx Y \left(\frac{Q}{V_{\text{day}}}\right)^{\gamma},
   \]
   optionally with additional bin-level controls (`fit_power_law_logbins_wls_new`).
4. Using the same bins, optionally fit a **logarithmic** model
   \[
   \frac{I}{\sigma_{\text{day}}} \approx a \,\log_{10}\!\bigl(1 + b\,Q/V_{\text{day}}\bigr).
   \]

Outputs:

- Overall fits: `images/{LEVEL}_{PROPRIETARY_TAG}/power_law_fit_overall_{LEVEL}.png` (power-law plus logarithmic overlay).
- Fits conditioned on participation quantiles: `images/{LEVEL}_{PROPRIETARY_TAG}/power_law_fits_by_participation_rate_{LEVEL}.png`.

These figures summarise how impact scales with size across the detected metaorders.
