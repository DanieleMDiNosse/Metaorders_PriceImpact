# Crowding section review (paper/main.tex)

Scope: `paper/main.tex` Section~\ref{sec:crowding} (roughly lines 940--1375 in the current working tree).

This note lists **inconsistencies and logical/statistical issues** I still see in the crowding section, plus **concrete fixes**.

---

## 1) Section title mentions “Co-Impact” but no co-impact analysis is shown

**Where:** `paper/main.tex:940` and the rest of Section~\ref{sec:crowding}.

**Issue:** The section is titled *“Crowding and Co-Impact …”* and motivates crowding as a driver of co-impact, but the section only reports crowding correlations. There is no definition/estimation of co-impact (e.g., impact conditional on crowding, or a co-impact model) in the section.

**Suggested fix:**cor
- Either **rename** the section to match the content (e.g., “Crowding of Proprietary vs Client Flow”), **or**
- Add a minimal co-impact analysis (even one paragraph + one figure/table), e.g. regress/stratify impact on a crowding variable:
  - Define a crowding covariate (e.g., within-group `imb_i^{k,d}` and cross-group `imb_i^{env}`).
  - Show that impact (or peak impact / temporary impact) is larger when `\varepsilon_i` aligns with the relevant imbalance.

---

## 2) “Cross-group imbalance” is referenced before it is defined

**Where:** `paper/main.tex:983` (KDE paragraph) vs definition of environment imbalance later at `paper/main.tex:1106`.

**Issue:** The text discusses “cross-group imbalance” in the within-group imbalance subsection, but the *cross-group* object is only defined later (environment imbalance). This reads as a forward reference without warning, and it is ambiguous what exact cross-group definition is being plotted in Figure~\ref{fig:imbalance_distributions}.

**Suggested fix:**
- Either move the KDE paragraph/figure to the cross-group subsection (after `\mathrm{imb}^{env}` is defined), **or**
- Keep it where it is but explicitly say “cross-group (environment) imbalance defined in the next subsection” and use consistent notation (`\mathrm{imb}^{env}`).

---

## 3) KDE interpretation is not clearly tied to what is actually plotted

**Where:** `paper/main.tex:983`–`paper/main.tex:987` and Figure~\ref{fig:imbalance_distributions}.

**Issue (conceptual/clarity):**
- The paragraph interprets “skewness towards positive values for buy, towards negative for sell”, which is inherently a *conditional-on-direction* statement, but the caption suggests unconditional KDEs (“distributions of the imbalances within-group and cross-group”).
- If buys and sells are pooled, symmetry around 0 can arise mechanically even under strong crowding (mixing a positive-shifted buy distribution with a negative-shifted sell distribution).

**Suggested fix:**
- Make the plot/paragraph consistent:
  - Either plot KDEs **conditional on** `\varepsilon_i=+1` and `\varepsilon_i=-1` (four curves: within/cross × buy/sell), or
  - Remove the buy/sell skewness language and interpret only the unconditional distributions (and be explicit about the weighting: per metaorder? per stock-day? per unit volume?).

---

## 4) Inference for “global” correlations is undocumented and inconsistent with later clustering logic

**Where:** reported CIs at `paper/main.tex:1042`–`paper/main.tex:1058`; commented-out Fisher CI discussion at `paper/main.tex:1009`–`paper/main.tex:1033`; day-cluster bootstrap approach later at `paper/main.tex:1185`–`paper/main.tex:1202`.

**Issue (statistical/logical):**
- The text reports 95% CIs for the *global* correlation `r` but does not state how they are computed (the only method discussion is commented out).
- Any iid-based CI is questionable here because observations are strongly clustered (at least by day, and also by stock-day), which the manuscript explicitly acknowledges later when it switches to day-cluster bootstrap for `\rho_b`.

**Suggested fix:**
- Pick one inference approach and apply it consistently:
  - Preferred: compute CIs for global `r` via a **day-cluster bootstrap** (same dependence logic as in Section~\ref{subsec:crowding_eta}), or
  - Alternatively: compute one correlation per day and report the mean across days with a CI on the mean (still day-cluster by construction).
- In either case, explicitly state the method (resampling unit, # replicates) in the “Empirical within-group crowding” paragraph.

---

## 5) “Minimum number of metaorders per day” threshold is vague/inconsistent

**Where:** `paper/main.tex:1007` vs explicit “(100)” later at `paper/main.tex:1099`–`paper/main.tex:1102`.

**Issue:** The within-group subsection says daily correlations are computed “whenever there are at least a minimum number” but doesn’t specify the threshold; later the time-series subsection states 100.

**Suggested fix:**
- State the threshold once, clearly, and use it for all daily-correlation computations (within-group and cross-group).
- Clarify whether the threshold is:
  - per group (prop/client separately),
  - per day across all instruments, or
  - per day *and* per instrument (if applicable).

---

## 6) Cross-group results still contain a TODO and are interpreted anyway

**Where:** `paper/main.tex:1143`.

**Issue:** The text literally contains “(Add CI)” and then interprets the sign of the estimates as “mild tendency” to trade against the other group.

**Suggested fix:**
- Either add a CI / p-value (e.g., bootstrap days for the mean daily correlation), or remove/soften the interpretation until inference is provided.

---

## 7) “Confidence bands” are referenced in figure captions without a method description

**Where:** within-group caption `paper/main.tex:1086`–`paper/main.tex:1089` and cross-group caption `paper/main.tex:1161`–`paper/main.tex:1164`.

**Issue:** Captions mention “confidence bands” for daily correlations, but the text does not define how those bands are computed (Fisher transform per day? bootstrap over days? something else).

**Suggested fix:**
- Add 1–2 sentences specifying the construction of these bands (and what they apply to: per-day `r_d` or rolling mean).

---

## 8) Notation in the `\eta`-conditioning figure captions is inconsistent with the definition of `\rho_b`

**Where:** `paper/main.tex:1181` (definition uses `\eta_i \in b`) vs captions `paper/main.tex:1245` and `paper/main.tex:1257` (uses “`\mid \eta`”).

**Issue:** The statistic is defined on **deciles/bins**, but captions read like conditioning on a continuous variable “`\eta`”.

**Suggested fix:**
- Replace “`\mid \eta`” in captions with “`\mid \eta_i \in b`” or explicitly say “as a function of the `\eta` decile”.

---

## 9) Environment/all-others imbalance edge cases are not specified

**Where:** environment imbalance definition `paper/main.tex:1113`–`paper/main.tex:1121`; all-others imbalance definition `paper/main.tex:1271`–`paper/main.tex:1276`.

**Issue:** For within-group imbalance you explicitly handle the zero-denominator case (single metaorder on `(k,d)`). For:
- cross-group environment imbalance, the denominator vanishes if there are **no source-group** metaorders on `(k,d)` but there are target metaorders;
- all-others imbalance, the denominator vanishes if there is only **one total** metaorder on `(k,d)`.

**Suggested fix:**
- Mirror the within-group rule: define the imbalance as NaN when the denominator is 0 and state that such observations are excluded from correlations/plots.

---

## 10) Member-level imbalance aggregates across all ISINs (conceptual mismatch with earlier “same stock & day” crowding)

**Where:** `paper/main.tex:1321`–`paper/main.tex:1323`.

**Issue:** Earlier crowding measures are explicitly stock-day specific. Member-level crowding instead aggregates a member’s client flow across *all* instruments for the day. This can:
- dilute strong instrument-level relationships (buys in one name offset sells in another),
- create interpretational ambiguity about “liquidity provision” vs “front-running” (which are usually instrument-level notions).

**Suggested fix (choose one):**
- Justify why cross-ISIN aggregation is the right object for the regulatory question, **or**
- Add an instrument-level variant, e.g. define `\text{imb}_{m,k,d}` using the member’s clients on instrument `k` and day `d`, and correlate proprietary directions on the same `(m,k,d)`.

---

## 11) Front-running vs liquidity provision is inferred from the sign of `\text{imb}_{m,d}` (wrong object)

**Where:** `paper/main.tex:1347`–`paper/main.tex:1349`.

**Issue (logical):** The sign of `\text{imb}_{m,d}` only tells you whether the *clients* net bought or sold that day. It does **not** tell you whether the member’s proprietary desk traded with or against clients. That is captured by the **sign of the correlation/alignment** between proprietary direction and `\text{imb}_{m,d}`.

**Suggested fix:**
- Rewrite this interpretation in terms of the sign of the estimated correlation (or rolling-window correlation):
  - positive correlation: proprietary aligns with client imbalance (potential amplification / potential front-running narrative),
  - negative correlation: proprietary opposes client imbalance (liquidity provision / intermediation narrative).

---

## 12) “Statistically significant correlation patterns” is asserted for the heatmap without showing significance

**Where:** `paper/main.tex:1352`–`paper/main.tex:1354` and Figure~\ref{fig:heatmap_imbcorr}.

**Issue:** The text claims statistical significance, but the heatmap figure/caption only describes correlations and a minimum-count filter (≥5 metaorders in a 3-day window). With such small windows, significance is not guaranteed and can be fragile.

**Suggested fix:**
- Either remove “statistically significant” from the narrative, **or**
- Add significance annotation/thresholding to the heatmap (e.g., show only cells with permutation p-value < 5% and enough observations, or add a separate panel with p-values).

---

## 13) Minor but visible notation/style inconsistency: `Corr` vs `corr`

**Where:** heatmap caption `paper/main.tex:1371` uses `\mathrm{corr}(\cdot,\cdot)` while the rest uses `\mathrm{Corr}`.

**Suggested fix:**
- Use one convention everywhere (prefer `\mathrm{Corr}`).

---

## 14) Notation reuse: `G_{k,d}` changes meaning in “all others”

**Where:** within-group definition `paper/main.tex:956`–`paper/main.tex:958` vs “all others” at `paper/main.tex:1276`.

**Issue:** Early in the section, `G_{k,d}` denotes “metaorders in group `G` on `(k,d)`”. In the “all others” subsection, `G_{k,d}` is redefined to contain *both* groups. This is easy to miss and can confuse readers.

**Suggested fix:**
- Use a different symbol for the union set, e.g. `\mathcal{M}_{k,d}` for “all metaorders on `(k,d)`”, and keep `G_{k,d}` reserved for group-specific sets.

---

## 15) Some summary crowding estimates are reported without uncertainty (while others have CIs)

**Where:** mean daily correlations at `paper/main.tex:1050`–`paper/main.tex:1066`, `paper/main.tex:1136`–`paper/main.tex:1141`, and `paper/main.tex:1282`–`paper/main.tex:1289`.

**Issue:** The section mixes:
- point estimates + 95% CIs (global within-group `r`),
- point estimates without any uncertainty (mean daily correlations, cross-group means, “all others” means),
- and later, bootstrap CIs for `\eta`-binned curves.

**Suggested fix:**
- Add day-cluster bootstrap CIs for the mean daily correlations (and/or for cross-group/all-others aggregates), or explicitly state that these are descriptive and defer inference to the bootstrap-based parts.
