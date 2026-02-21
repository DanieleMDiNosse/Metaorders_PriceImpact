"""
Plot styling helpers shared across scripts.

The repository uses a Plotly-first workflow with a single theme for consistent
fonts, colors, and layout defaults.
"""

from __future__ import annotations

from typing import Sequence

THEME_COLORWAY: tuple[str, ...] = (
    "#5B8FF9",
    "#91CC75",
    "#EE6666",
    "#5470C6",
    "#FAC858",
    "#73C0DE",
)
THEME_GRID_COLOR = "#E5ECF6"
THEME_BG_COLOR = "#FFFFFF"
THEME_FONT_FAMILY = "DejaVu Sans"
PLOTLY_TEMPLATE_NAME = "moimpact_white"


def apply_plotly_style(
    *,
    tick_font_size: int,
    label_font_size: int,
    title_font_size: int,
    legend_font_size: int,
    theme_colorway: Sequence[str],
    theme_grid_color: str,
    theme_bg_color: str,
    theme_font_family: str,
    template_name: str = PLOTLY_TEMPLATE_NAME,
) -> None:
    """
    Summary
    -------
    Register and activate a shared Plotly template used across this repository.

    Parameters
    ----------
    tick_font_size : int
        Base font size for axis tick labels.
    label_font_size : int
        Axis label font size.
    title_font_size : int
        Title font size.
    legend_font_size : int
        Legend font size.
    theme_colorway : Sequence[str]
        Plotly color cycle used for discrete traces.
    theme_grid_color : str
        Gridline color.
    theme_bg_color : str
        Background color for paper/plot areas.
    theme_font_family : str
        Font family name.
    template_name : str, default=PLOTLY_TEMPLATE_NAME
        Name under which the template is registered in `plotly.io.templates`.

    Returns
    -------
    None

    Notes
    -----
    - This function updates Plotly's global template registry and default.
    - The template starts from `plotly_white` and applies repository-specific
      overrides so line/bar/scatter and 3D scene figures share a common look.

    Examples
    --------
    >>> apply_plotly_style(
    ...     tick_font_size=12,
    ...     label_font_size=14,
    ...     title_font_size=15,
    ...     legend_font_size=12,
    ...     theme_colorway=THEME_COLORWAY,
    ...     theme_grid_color=THEME_GRID_COLOR,
    ...     theme_bg_color=THEME_BG_COLOR,
    ...     theme_font_family=THEME_FONT_FAMILY,
    ... )
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    base_template = go.layout.Template()
    if "plotly_white" in pio.templates:
        base_template = go.layout.Template(pio.templates["plotly_white"])

    base_template.layout.update(
        go.Layout(
            colorway=list(theme_colorway),
            paper_bgcolor=theme_bg_color,
            plot_bgcolor=theme_bg_color,
            font=dict(family=theme_font_family, size=tick_font_size, color="#1F2937"),
            title=dict(font=dict(size=title_font_size)),
            legend=dict(font=dict(size=legend_font_size)),
            hoverlabel=dict(font=dict(family=theme_font_family, size=tick_font_size)),
            xaxis=dict(
                showgrid=True,
                gridcolor=theme_grid_color,
                zeroline=False,
                title_font=dict(size=label_font_size),
                tickfont=dict(size=tick_font_size),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=theme_grid_color,
                zeroline=False,
                title_font=dict(size=label_font_size),
                tickfont=dict(size=tick_font_size),
            ),
            scene=dict(
                xaxis=dict(
                    showgrid=True,
                    gridcolor=theme_grid_color,
                    showbackground=True,
                    backgroundcolor=theme_bg_color,
                    title_font=dict(size=label_font_size),
                    tickfont=dict(size=tick_font_size),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=theme_grid_color,
                    showbackground=True,
                    backgroundcolor=theme_bg_color,
                    title_font=dict(size=label_font_size),
                    tickfont=dict(size=tick_font_size),
                ),
                zaxis=dict(
                    showgrid=True,
                    gridcolor=theme_grid_color,
                    showbackground=True,
                    backgroundcolor=theme_bg_color,
                    title_font=dict(size=label_font_size),
                    tickfont=dict(size=tick_font_size),
                ),
            ),
        )
    )
    pio.templates[template_name] = base_template
    pio.templates.default = template_name
