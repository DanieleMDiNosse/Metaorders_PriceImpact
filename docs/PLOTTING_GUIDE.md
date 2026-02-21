# Plotting Guide (Titles, Labels, Styling, Exports)

This repository uses a Plotly-first workflow with shared helpers so plots look consistent across scripts.

## 1) Shared plotting modules

- `moimpact/plot_style.py`
  - Registers a global Plotly template via `apply_plotly_style(...)`.
  - Controls default font family, color palette, grid color, and default font sizes.
- `moimpact/plotting.py`
  - Provides canonical output folder handling:
    - `make_plot_output_dirs(...)`
    - `ensure_plot_dirs(...)`
  - Provides unified export:
    - `save_plotly_figure(...)` (writes HTML and/or PNG).
  - Defines canonical colors (for proprietary/client/bands).

## 2) Fastest way to change titles and axis labels

For any `go.Figure`:

```python
fig.update_layout(
    title="My New Title",
    xaxis_title="X label",
    yaxis_title="Y label",
)
```

For subplots:

```python
fig.update_xaxes(title_text="X label", row=1, col=1)
fig.update_yaxes(title_text="Y label", row=1, col=1)
```

For Plotly Express (e.g. `px.bar`, `px.line`), set labels at creation time:

```python
fig = px.bar(
    df,
    x="col_x",
    y="col_y",
    title="My New Title",
    labels={"col_x": "X label", "col_y": "Y label"},
)
```

## 3) Global font-size styling (recommended)

Most scripts load these values from YAML:

- `TICK_FONT_SIZE`
- `LABEL_FONT_SIZE`
- `TITLE_FONT_SIZE`
- `LEGEND_FONT_SIZE`

Config files:

- `config_ymls/metaorder_computation.yml`
- `config_ymls/metaorder_statistics.yml`
- `config_ymls/crowding_analysis.yml`

Each script then calls:

```python
apply_plotly_style(
    tick_font_size=...,
    label_font_size=...,
    title_font_size=...,
    legend_font_size=...,
    theme_colorway=THEME_COLORWAY,
    theme_grid_color=THEME_GRID_COLOR,
    theme_bg_color=THEME_BG_COLOR,
    theme_font_family=THEME_FONT_FAMILY,
)
```

If you want all plots in a script to use bigger/smaller text, change these four values first.

## 4) Changing colors

- Main palette: `THEME_COLORWAY` in `moimpact/plot_style.py`
- Group colors used across analyses: `COLOR_PROPRIETARY`, `COLOR_CLIENT`, and confidence-band colors in `moimpact/plotting.py`

If you change colors there, all scripts importing these constants will inherit the new palette.

## 5) Saving HTML/PNG consistently

Use the shared save helper instead of custom file writing:

```python
plot_dirs = make_plot_output_dirs(Path("images/my_analysis"), use_subdirs=True)
html_path, png_path = save_plotly_figure(
    fig,
    stem="my_plot_name",
    dirs=plot_dirs,
    write_html=True,
    write_png=True,
    strict_png=False,
)
```

Notes:

- With `use_subdirs=True`, files go into `.../html/` and `.../png/`.
- PNG export requires Plotly static export backend (`kaleido`).
- If PNG export fails and `strict_png=False`, HTML is still written.

## 6) Where to edit existing plots

- `scripts/metaorder_computation.py`
  - Edit `fig.update_layout(...)` in plot functions like `plot_fit`, `plot_normalized_impact_path`, and surface/heatmap helpers.
- `scripts/metaorder_statistics.py`
  - Edit titles/labels in `fig.update_layout(...)` and `px.*(..., labels=..., title=...)` inside stats plotting blocks.
- `scripts/crowding_analysis.py`
  - Edit titles/labels inside plotting helpers like `plot_daily_crowding`, `plot_imbalance_distributions`, and return/ACF plotting functions.
- `scripts/member_statistics.py`
  - Edit titles/labels in `px.bar(...)`, heatmap layout blocks, and helper `save_plotly`.
- `scripts/crowding_vs_part_rate.py`
  - Edit shared plot helpers `_plotly_curve_date_ci(...)` and `plotly_heatmap_align(...)`.
- `scripts/plot_prop_nonprop_fits.py`
  - Quick title override from CLI:
    - `python scripts/plot_prop_nonprop_fits.py --title "Your title"`

## 7) Minimal pattern for new plots

```python
import plotly.graph_objects as go
from pathlib import Path
from moimpact.plotting import make_plot_output_dirs, save_plotly_figure

fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 3], mode="lines+markers", name="Series"))
fig.update_layout(title="Example", xaxis_title="X", yaxis_title="Y")

dirs = make_plot_output_dirs(Path("images/example"), use_subdirs=True)
save_plotly_figure(fig, stem="example_plot", dirs=dirs, write_html=True, write_png=True, strict_png=False)
```

This keeps output structure and styling aligned with the rest of the repository.
