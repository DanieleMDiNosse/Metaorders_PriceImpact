# Plotting Guide

The repository uses a Plotly-first workflow with shared helpers in
`moimpact/plot_style.py` and `moimpact/plotting.py`. This note documents the
current conventions so new analyses match the existing output tree and styling.

## Shared plotting modules

`moimpact/plot_style.py`

- defines the repository theme colors
- registers the `moimpact_white` Plotly template
- exposes `apply_plotly_style(...)`

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

Most scripts call:

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
- if `strict_png=False`, failed PNG export does not block HTML export

Pipeline-level legend suppression:

- setting `DISABLE_PLOT_LEGENDS=true` hides legends for all figures saved
  through the shared helper

## Title handling

Several scripts wrap the shared save helper and remove top-level titles before
export so that paper captions carry the final title text. Current wrappers
exist in:

- `scripts/metaorder_computation.py`
- `scripts/metaorder_distributions.py`
- `scripts/metaorder_summary_statistics.py`
- `scripts/metaorder_intraday_analysis.py`
- `scripts/metaorder_start_event_study.py`
- `scripts/metaorder_clustering.py`

If you want the exported file to keep its title, check whether the script uses a
local `save_plotly_figure(...)` wrapper that calls `fig.update_layout(title=None)`.

## Global style knobs

Most plotting scripts read font sizes from YAML:

- `TICK_FONT_SIZE`
- `LABEL_FONT_SIZE`
- `TITLE_FONT_SIZE`
- `LEGEND_FONT_SIZE`
- `ANNOTATION_FONT_SIZE` in scripts that use subplot annotations

These values are passed into `apply_plotly_style(...)`.

Configs that currently expose these keys include:

- `config_ymls/metaorder_computation.yml`
- `config_ymls/metaorder_distributions.yml`
- `config_ymls/metaorder_summary_statistics.yml`
- `config_ymls/crowding_analysis.yml`
- `config_ymls/metaorder_start_event_study.yml`
- `config_ymls/metaorder_intraday_analysis.yml`
- `config_ymls/plot_prop_nonprop_fits.yml`
- `config_ymls/paper_figures.yml`

## Common places to edit plots

- `scripts/metaorder_computation.py`
  - one-dimensional fits, surfaces, and impact paths
- `scripts/metaorder_distributions.py`
  - multi-panel distribution figure and fit annotations
- `scripts/metaorder_summary_statistics.py`
  - nationality, member profile, and daily-share figures
- `scripts/crowding_analysis.py`
  - daily crowding, diagnostics, and member-level plots
- `scripts/crowding_vs_part_rate.py`
  - `eta` curves, noise bands, and optional heatmaps
- `scripts/metaorder_start_event_study.py`
  - treated-vs-control event curves
- `scripts/metaorder_intraday_analysis.py`
  - session counts and session fit comparisons
- `scripts/metaorder_clustering.py`
  - PCA, silhouette, composition, and cluster-profile figures

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
