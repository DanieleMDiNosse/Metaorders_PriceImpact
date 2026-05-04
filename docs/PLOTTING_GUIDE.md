# Plotting Guide

The repository uses a Plotly-first workflow with shared helpers in
`moimpact/plot_style.py` and `moimpact/plotting.py`. Plot typography and export
defaults are now centralized in `config_ymls/plot_style.yml`. This note
documents the current conventions so new analyses match the existing output tree
and paper-oriented styling.

## Shared plotting modules

`moimpact/plot_style.py`

- loads the central style spec from `config_ymls/plot_style.yml`
- defines the repository theme colors
- registers the `moimpact_white` Plotly template
- exposes `load_plot_style(...)`
- exposes `apply_shared_plotly_style(...)`
- exposes `apply_matplotlib_style(...)`

`moimpact/plotting.py`

- defines canonical group colors:
  - `COLOR_PROPRIETARY`
  - `COLOR_CLIENT`
  - `COLOR_NEUTRAL`
  - confidence-band colors
- provides `PlotOutputDirs`
- provides:
  - `make_plot_output_dirs(...)`
  - `ensure_plot_dirs(...)`
  - `save_plotly_figure(...)`

## Default output layout

Most workflow modules call:

```python
dirs = make_plot_output_dirs(Path("images/my_analysis"), use_subdirs=True)
```

which implies:

- HTML files go to `.../html/`
- PNG files go to `.../png/`

The helper creates directories on demand through `ensure_plot_dirs(...)`.

## Shared save helper

Canonical export pattern:

```python
html_path, png_path = save_plotly_figure(
    fig,
    stem="my_plot",
    dirs=dirs,
    write_html=True,
    write_png=True,
    strict_png=False,
)
```

Important behavior in `save_plotly_figure(...)`:

- HTML export includes Plotly JS from CDN
- HTML export includes MathJax from CDN, so LaTeX labels render in saved HTML
- PNG export uses Plotly's static backend, typically `kaleido`
- optional PDF export writes `stem.pdf` beside `stem.png`
- if `strict_png=False`, failed PNG export does not block HTML export

Pipeline-level legend suppression:

- setting `DISABLE_PLOT_LEGENDS=true` hides legends for all figures saved
  through the shared helper

## Title handling

Several workflow modules wrap the shared save helper and remove top-level
titles before export so that paper captions carry the final title text. Current
wrappers are used by:

- `scripts/run_analysis.py metaorders compute`
- `scripts/run_analysis.py metaorders distributions`
- `scripts/run_analysis.py metaorders summary`
- `scripts/run_analysis.py metaorders intraday-impact`
- `scripts/run_analysis.py metaorders start-event`
- `scripts/run_analysis.py metaorders start-time`
- `scripts/run_analysis.py crowding daily`
- `scripts/run_analysis.py crowding eta`
- `scripts/run_analysis.py crowding impact`
- `scripts/run_analysis.py crowding intraday`
- `scripts/run_analysis.py execution schedule`
- `scripts/run_analysis.py execution cluster`
- `scripts/run_analysis.py execution typology`
- `scripts/run_analysis.py members stats`

If you want the exported file to keep its title, check whether the workflow
module uses a local `save_plotly_figure(...)` wrapper that calls
`fig.update_layout(title=None)`.

## Global style knobs

Paper-oriented typography and export defaults now live in:

- `config_ymls/plot_style.yml`

This file is the single source of truth for:

- `TICK_FONT_SIZE`
- `LABEL_FONT_SIZE`
- `TITLE_FONT_SIZE`
- `LEGEND_FONT_SIZE`
- `ANNOTATION_FONT_SIZE`
- theme colors, grid/background colors, legend styling, and static export size

Per-workflow YAML files no longer own plotting font sizes.

## Common Places To Edit Plots

- `moimpact/workflows/metaorders/compute.py`
  (`scripts/run_analysis.py metaorders compute`)
  - one-dimensional fits, surfaces, and impact paths
- `moimpact/workflows/metaorders/distributions.py`
  (`scripts/run_analysis.py metaorders distributions`)
  - multi-panel distribution figure and fit annotations
- `moimpact/workflows/metaorders/summary.py`
  (`scripts/run_analysis.py metaorders summary`)
  - nationality, member profile, and daily-share figures
- `moimpact/workflows/crowding/daily.py`
  (`scripts/run_analysis.py crowding daily`)
  - daily crowding, diagnostics, and member-level plots
- `moimpact/workflows/crowding/eta.py`
  (`scripts/run_analysis.py crowding eta`)
  - `eta` curves, noise bands, and optional heatmaps
- `moimpact/workflows/crowding/impact.py`
  (`scripts/run_analysis.py crowding impact`)
  - crowding-conditioned impact curves and regression figures
- `moimpact/workflows/crowding/intraday.py`
  (`scripts/run_analysis.py crowding intraday`)
  - intraday start-bin crowding profiles and heatmaps
- `moimpact/workflows/crowding/overlap.py`
  (`scripts/run_analysis.py crowding overlap`)
  - active-overlap distributions, intraday summaries, and regressions
- `moimpact/workflows/crowding/member_overlap.py`
  (`scripts/run_analysis.py crowding member-overlap`)
  - member-level active-overlap correlation summaries
- `moimpact/workflows/impact/overlay.py`
  (`scripts/run_analysis.py impact overlay`)
  - proprietary-vs-client power-law, log, and retention overlays
- `moimpact/workflows/metaorders/start_event_study.py`
  (`scripts/run_analysis.py metaorders start-event`)
  - treated-vs-control event curves
- `moimpact/workflows/metaorders/start_time_distribution.py`
  (`scripts/run_analysis.py metaorders start-time`)
  - intraday start-time distribution comparisons
- `moimpact/workflows/metaorders/intraday_impact.py`
  (`scripts/run_analysis.py metaorders intraday-impact`)
  - session counts and session fit comparisons
- `moimpact/workflows/execution/schedule.py`
  (`scripts/run_analysis.py execution schedule`)
  - execution-schedule curves, heatmaps, and scalar summaries
- `moimpact/workflows/execution/cluster.py`
  (`scripts/run_analysis.py execution cluster`)
  - PCA, silhouette, composition, and cluster-profile figures
- `moimpact/workflows/execution/typology.py`
  (`scripts/run_analysis.py execution typology`)
  - execution-type heatmaps, shares, PCA, impact, and schedule figures
- `moimpact/workflows/members/stats.py`
  (`scripts/run_analysis.py members stats`)
  - member coverage bars and ISIN/member heatmaps

## Minimal new-figure pattern

```python
from pathlib import Path
import plotly.graph_objects as go

from moimpact.plotting import make_plot_output_dirs, save_plotly_figure

fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 3], mode="lines+markers"))
fig.update_layout(xaxis_title="X", yaxis_title="Y")

dirs = make_plot_output_dirs(Path("images/example"), use_subdirs=True)
save_plotly_figure(fig, stem="example_plot", dirs=dirs, write_html=True, write_png=True, strict_png=False)
```

## Related docs

- [`index.md`](index.md)
- [`market_impact.md`](market_impact.md)
- [`imbalance_and_crowding.md`](imbalance_and_crowding.md)
