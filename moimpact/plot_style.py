"""
Shared plotting-style configuration for scientific-paper figures.

The repository uses Plotly for most figures and a small matplotlib branch for a
few diagnostics. This module provides one central style specification so every
image-generating script can use the same typography, colors, legend treatment,
and export defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Sequence

from moimpact.config import load_yaml_mapping


PLOTLY_TEMPLATE_NAME = "moimpact_white"
PLOT_STYLE_CONFIG_ENV = "PLOT_STYLE_CONFIG"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STYLE_CONFIG_PATH = _REPO_ROOT / "config_ymls" / "plot_style.yml"

_DEFAULT_THEME_COLORWAY: tuple[str, ...] = (
    "#355C7D",
    "#6C8EAD",
    "#C06C84",
    "#F67280",
    "#F8B195",
    "#7FB3B3",
)
_DEFAULT_GRID_COLOR = "#D9E1EA"
_DEFAULT_BG_COLOR = "#FFFFFF"
_DEFAULT_FONT_FAMILY = "DejaVu Sans"
_DEFAULT_FONT_COLOR = "#1F2937"
_DEFAULT_AXIS_LINE_COLOR = "#6B7280"
_DEFAULT_LEGEND_BG_COLOR = "rgba(255,255,255,0.92)"
_DEFAULT_LEGEND_BORDER_COLOR = "#CBD5E1"
_DEFAULT_LEGEND_BORDER_WIDTH = 1
_DEFAULT_LEGEND_ORIENTATION = "v"
_DEFAULT_LEGEND_X = 0.985
_DEFAULT_LEGEND_Y = 0.02
_DEFAULT_LEGEND_XANCHOR = "right"
_DEFAULT_LEGEND_YANCHOR = "bottom"
_DEFAULT_TICK_FONT_SIZE = 18
_DEFAULT_LABEL_FONT_SIZE = 20
_DEFAULT_TITLE_FONT_SIZE = 19
_DEFAULT_LEGEND_FONT_SIZE = 18
_DEFAULT_ANNOTATION_FONT_SIZE = 18
_DEFAULT_LINE_WIDTH = 2.4
_DEFAULT_REFERENCE_LINE_WIDTH = 1.2
_DEFAULT_MARKER_SIZE = 8
_DEFAULT_EXPORT_WIDTH = 1600
_DEFAULT_EXPORT_HEIGHT = 900
_DEFAULT_EXPORT_SCALE = 3
_DEFAULT_MATPLOTLIB_DPI = 240


@dataclass(frozen=True)
class PlotStyleSpec:
    """
    Summary
    -------
    Central style contract shared by all plotting workflows.

    Parameters
    ----------
    theme_colorway : tuple[str, ...]
        Ordered discrete color palette used across scripts.
    grid_color : str
        Gridline color for 2D and 3D plots.
    bg_color : str
        Background color for paper and plot regions.
    font_family : str
        Default font family for labels, ticks, legends, and annotations.
    font_color : str
        Default text color.
    axis_line_color : str
        Color used for visible axis lines.
    legend_bg_color : str
        Legend background color.
    legend_border_color : str
        Legend border color.
    legend_border_width : int
        Legend border width in pixels.
    legend_orientation : str
        Plotly legend orientation, typically ``"v"`` or ``"h"``.
    legend_x : float
        Legend x-position in paper coordinates.
    legend_y : float
        Legend y-position in paper coordinates.
    legend_xanchor : str
        Horizontal anchor used for legend placement.
    legend_yanchor : str
        Vertical anchor used for legend placement.
    tick_font_size : int
        Base tick-label font size.
    label_font_size : int
        Axis-label font size.
    title_font_size : int
        Figure title font size.
    legend_font_size : int
        Legend font size.
    annotation_font_size : int
        Default annotation font size.
    default_line_width : float
        Default Plotly/Matplotlib line width for traces.
    reference_line_width : float
        Default width for guide and reference lines.
    default_marker_size : float
        Default marker size for scatter-like traces.
    export_width : int
        Default static-export width in pixels.
    export_height : int
        Default static-export height in pixels.
    export_scale : int
        Default Plotly static-export scale multiplier.
    matplotlib_dpi : int
        Default Matplotlib DPI for raster outputs.

    Returns
    -------
    PlotStyleSpec
        Immutable style specification.

    Notes
    -----
    The repository treats this dataclass as the single source of truth for
    publication-oriented figure styling.
    """

    theme_colorway: tuple[str, ...]
    grid_color: str
    bg_color: str
    font_family: str
    font_color: str
    axis_line_color: str
    legend_bg_color: str
    legend_border_color: str
    legend_border_width: int
    legend_orientation: str
    legend_x: float
    legend_y: float
    legend_xanchor: str
    legend_yanchor: str
    tick_font_size: int
    label_font_size: int
    title_font_size: int
    legend_font_size: int
    annotation_font_size: int
    default_line_width: float
    reference_line_width: float
    default_marker_size: float
    export_width: int
    export_height: int
    export_scale: int
    matplotlib_dpi: int


def _resolve_plot_style_config_path(config_path: Path | None = None) -> Path:
    if config_path is not None:
        candidate = Path(config_path).expanduser()
    else:
        override = os.environ.get(PLOT_STYLE_CONFIG_ENV)
        candidate = Path(override).expanduser() if override else _DEFAULT_STYLE_CONFIG_PATH
    if not candidate.is_absolute():
        candidate = (_REPO_ROOT / candidate).resolve()
    return candidate


@lru_cache(maxsize=8)
def _load_plot_style_from_path(config_path: str) -> PlotStyleSpec:
    cfg = load_yaml_mapping(Path(config_path))
    raw_colorway = cfg.get("THEME_COLORWAY", _DEFAULT_THEME_COLORWAY)
    if not isinstance(raw_colorway, Sequence) or isinstance(raw_colorway, (str, bytes, bytearray)):
        raise TypeError("THEME_COLORWAY must be a YAML list of color strings.")
    theme_colorway = tuple(str(color) for color in raw_colorway)
    if not theme_colorway:
        raise ValueError("THEME_COLORWAY must contain at least one color.")

    return PlotStyleSpec(
        theme_colorway=theme_colorway,
        grid_color=str(cfg.get("GRID_COLOR", _DEFAULT_GRID_COLOR)),
        bg_color=str(cfg.get("BG_COLOR", _DEFAULT_BG_COLOR)),
        font_family=str(cfg.get("FONT_FAMILY", _DEFAULT_FONT_FAMILY)),
        font_color=str(cfg.get("FONT_COLOR", _DEFAULT_FONT_COLOR)),
        axis_line_color=str(cfg.get("AXIS_LINE_COLOR", _DEFAULT_AXIS_LINE_COLOR)),
        legend_bg_color=str(cfg.get("LEGEND_BG_COLOR", _DEFAULT_LEGEND_BG_COLOR)),
        legend_border_color=str(cfg.get("LEGEND_BORDER_COLOR", _DEFAULT_LEGEND_BORDER_COLOR)),
        legend_border_width=int(cfg.get("LEGEND_BORDER_WIDTH", _DEFAULT_LEGEND_BORDER_WIDTH)),
        legend_orientation=str(cfg.get("LEGEND_ORIENTATION", _DEFAULT_LEGEND_ORIENTATION)),
        legend_x=float(cfg.get("LEGEND_X", _DEFAULT_LEGEND_X)),
        legend_y=float(cfg.get("LEGEND_Y", _DEFAULT_LEGEND_Y)),
        legend_xanchor=str(cfg.get("LEGEND_XANCHOR", _DEFAULT_LEGEND_XANCHOR)),
        legend_yanchor=str(cfg.get("LEGEND_YANCHOR", _DEFAULT_LEGEND_YANCHOR)),
        tick_font_size=int(cfg.get("TICK_FONT_SIZE", _DEFAULT_TICK_FONT_SIZE)),
        label_font_size=int(cfg.get("LABEL_FONT_SIZE", _DEFAULT_LABEL_FONT_SIZE)),
        title_font_size=int(cfg.get("TITLE_FONT_SIZE", _DEFAULT_TITLE_FONT_SIZE)),
        legend_font_size=int(cfg.get("LEGEND_FONT_SIZE", _DEFAULT_LEGEND_FONT_SIZE)),
        annotation_font_size=int(cfg.get("ANNOTATION_FONT_SIZE", _DEFAULT_ANNOTATION_FONT_SIZE)),
        default_line_width=float(cfg.get("DEFAULT_LINE_WIDTH", _DEFAULT_LINE_WIDTH)),
        reference_line_width=float(cfg.get("REFERENCE_LINE_WIDTH", _DEFAULT_REFERENCE_LINE_WIDTH)),
        default_marker_size=float(cfg.get("DEFAULT_MARKER_SIZE", _DEFAULT_MARKER_SIZE)),
        export_width=int(cfg.get("EXPORT_WIDTH", _DEFAULT_EXPORT_WIDTH)),
        export_height=int(cfg.get("EXPORT_HEIGHT", _DEFAULT_EXPORT_HEIGHT)),
        export_scale=int(cfg.get("EXPORT_SCALE", _DEFAULT_EXPORT_SCALE)),
        matplotlib_dpi=int(cfg.get("MATPLOTLIB_DPI", _DEFAULT_MATPLOTLIB_DPI)),
    )


def load_plot_style(config_path: Path | None = None) -> PlotStyleSpec:
    """
    Summary
    -------
    Load the shared plotting style from the central YAML configuration.

    Parameters
    ----------
    config_path : Path | None, default=None
        Optional explicit style-config path. When omitted, the function uses
        `PLOT_STYLE_CONFIG` if set, otherwise `config_ymls/plot_style.yml`.

    Returns
    -------
    PlotStyleSpec
        Parsed immutable plotting-style specification.

    Notes
    -----
    The result is cached per resolved config path, so repeated calls from many
    scripts are inexpensive.

    Examples
    --------
    >>> style = load_plot_style()
    >>> style.tick_font_size >= 12
    True
    """
    resolved = _resolve_plot_style_config_path(config_path)
    return _load_plot_style_from_path(str(resolved))


def apply_plotly_style(
    *,
    tick_font_size: int,
    label_font_size: int,
    title_font_size: int,
    legend_font_size: int,
    annotation_font_size: int,
    theme_colorway: Sequence[str],
    theme_grid_color: str,
    theme_bg_color: str,
    theme_font_family: str,
    theme_font_color: str = _DEFAULT_FONT_COLOR,
    axis_line_color: str = _DEFAULT_AXIS_LINE_COLOR,
    legend_bg_color: str = _DEFAULT_LEGEND_BG_COLOR,
    legend_border_color: str = _DEFAULT_LEGEND_BORDER_COLOR,
    legend_border_width: int = _DEFAULT_LEGEND_BORDER_WIDTH,
    legend_orientation: str = _DEFAULT_LEGEND_ORIENTATION,
    legend_x: float = _DEFAULT_LEGEND_X,
    legend_y: float = _DEFAULT_LEGEND_Y,
    legend_xanchor: str = _DEFAULT_LEGEND_XANCHOR,
    legend_yanchor: str = _DEFAULT_LEGEND_YANCHOR,
    default_line_width: float = _DEFAULT_LINE_WIDTH,
    default_marker_size: float = _DEFAULT_MARKER_SIZE,
    template_name: str = PLOTLY_TEMPLATE_NAME,
) -> None:
    """
    Summary
    -------
    Register and activate the repository Plotly template.

    Parameters
    ----------
    tick_font_size : int
        Base font size for axis tick labels.
    label_font_size : int
        Axis label font size.
    title_font_size : int
        Figure title font size.
    legend_font_size : int
        Legend font size.
    annotation_font_size : int
        Default font size for Plotly annotations.
    theme_colorway : Sequence[str]
        Plotly color cycle used for discrete traces.
    theme_grid_color : str
        Gridline color.
    theme_bg_color : str
        Background color for paper/plot areas.
    theme_font_family : str
        Font family name.
    theme_font_color : str, default=_DEFAULT_FONT_COLOR
        Default text color.
    axis_line_color : str, default=_DEFAULT_AXIS_LINE_COLOR
        Color used for 2D axis lines.
    legend_bg_color : str, default=_DEFAULT_LEGEND_BG_COLOR
        Legend background color.
    legend_border_color : str, default=_DEFAULT_LEGEND_BORDER_COLOR
        Legend border color.
    legend_border_width : int, default=_DEFAULT_LEGEND_BORDER_WIDTH
        Legend border width in pixels.
    legend_orientation : str, default=_DEFAULT_LEGEND_ORIENTATION
        Default legend orientation.
    legend_x : float, default=_DEFAULT_LEGEND_X
        Default legend x-position in paper coordinates.
    legend_y : float, default=_DEFAULT_LEGEND_Y
        Default legend y-position in paper coordinates.
    legend_xanchor : str, default=_DEFAULT_LEGEND_XANCHOR
        Default legend horizontal anchor.
    legend_yanchor : str, default=_DEFAULT_LEGEND_YANCHOR
        Default legend vertical anchor.
    default_line_width : float, default=_DEFAULT_LINE_WIDTH
        Default line width for Plotly scatter-like traces.
    default_marker_size : float, default=_DEFAULT_MARKER_SIZE
        Default marker size for scatter-like traces.
    template_name : str, default=PLOTLY_TEMPLATE_NAME
        Name under which the template is registered in `plotly.io.templates`.

    Returns
    -------
    None

    Notes
    -----
    - This function updates Plotly's global template registry and default.
    - The template starts from `plotly_white` and applies repository-specific
      overrides so 2D, 3D, and annotation-heavy figures share the same visual
      language in paper exports.

    Examples
    --------
    >>> style = load_plot_style()
    >>> apply_plotly_style(
    ...     tick_font_size=style.tick_font_size,
    ...     label_font_size=style.label_font_size,
    ...     title_font_size=style.title_font_size,
    ...     legend_font_size=style.legend_font_size,
    ...     annotation_font_size=style.annotation_font_size,
    ...     theme_colorway=style.theme_colorway,
    ...     theme_grid_color=style.grid_color,
    ...     theme_bg_color=style.bg_color,
    ...     theme_font_family=style.font_family,
    ... )
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    base_template = go.layout.Template()
    if "plotly_white" in pio.templates:
        base_template = go.layout.Template(pio.templates["plotly_white"])

    base_template.data.scatter = [
        go.Scatter(line=dict(width=default_line_width), marker=dict(size=default_marker_size))
    ]
    base_template.data.scattergl = [
        go.Scattergl(line=dict(width=default_line_width), marker=dict(size=default_marker_size))
    ]

    base_template.layout.update(
        go.Layout(
            colorway=list(theme_colorway),
            paper_bgcolor=theme_bg_color,
            plot_bgcolor=theme_bg_color,
            font=dict(family=theme_font_family, size=tick_font_size, color=theme_font_color),
            title=dict(font=dict(size=title_font_size)),
            hoverlabel=dict(font=dict(family=theme_font_family, size=tick_font_size)),
            legend=dict(
                font=dict(size=legend_font_size),
                bgcolor=legend_bg_color,
                bordercolor=legend_border_color,
                borderwidth=legend_border_width,
                orientation=legend_orientation,
                x=legend_x,
                y=legend_y,
                xanchor=legend_xanchor,
                yanchor=legend_yanchor,
            ),
            annotationdefaults=dict(
                font=dict(family=theme_font_family, size=annotation_font_size, color=theme_font_color)
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor=theme_grid_color,
                zeroline=False,
                showline=True,
                linecolor=axis_line_color,
                ticks="outside",
                ticklen=6,
                automargin=True,
                title_font=dict(size=label_font_size),
                tickfont=dict(size=tick_font_size),
                exponentformat="power",
                showexponent="all",
                minorloglabels="none",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=theme_grid_color,
                zeroline=False,
                showline=True,
                linecolor=axis_line_color,
                ticks="outside",
                ticklen=6,
                automargin=True,
                title_font=dict(size=label_font_size),
                tickfont=dict(size=tick_font_size),
                exponentformat="power",
                showexponent="all",
                minorloglabels="none",
            ),
            scene=dict(
                xaxis=dict(
                    showgrid=True,
                    gridcolor=theme_grid_color,
                    showbackground=True,
                    backgroundcolor=theme_bg_color,
                    title_font=dict(size=label_font_size),
                    tickfont=dict(size=tick_font_size),
                    exponentformat="power",
                    showexponent="all",
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=theme_grid_color,
                    showbackground=True,
                    backgroundcolor=theme_bg_color,
                    title_font=dict(size=label_font_size),
                    tickfont=dict(size=tick_font_size),
                    exponentformat="power",
                    showexponent="all",
                ),
                zaxis=dict(
                    showgrid=True,
                    gridcolor=theme_grid_color,
                    showbackground=True,
                    backgroundcolor=theme_bg_color,
                    title_font=dict(size=label_font_size),
                    tickfont=dict(size=tick_font_size),
                    exponentformat="power",
                    showexponent="all",
                ),
            ),
        )
    )
    pio.templates[template_name] = base_template
    pio.templates.default = template_name


def apply_shared_plotly_style(style: PlotStyleSpec | None = None) -> PlotStyleSpec:
    """
    Summary
    -------
    Activate the shared Plotly template from the central style specification.

    Parameters
    ----------
    style : PlotStyleSpec | None, default=None
        Optional explicit style specification. When omitted, the central style
        file is loaded automatically.

    Returns
    -------
    PlotStyleSpec
        The style specification that was applied.

    Notes
    -----
    This helper is the preferred entry point for scripts. It keeps script-level
    initialization small and makes the central YAML the single source of truth.

    Examples
    --------
    >>> style = apply_shared_plotly_style()
    >>> style.legend_font_size >= style.tick_font_size
    True
    """
    resolved_style = load_plot_style() if style is None else style
    apply_plotly_style(
        tick_font_size=resolved_style.tick_font_size,
        label_font_size=resolved_style.label_font_size,
        title_font_size=resolved_style.title_font_size,
        legend_font_size=resolved_style.legend_font_size,
        annotation_font_size=resolved_style.annotation_font_size,
        theme_colorway=resolved_style.theme_colorway,
        theme_grid_color=resolved_style.grid_color,
        theme_bg_color=resolved_style.bg_color,
        theme_font_family=resolved_style.font_family,
        theme_font_color=resolved_style.font_color,
        axis_line_color=resolved_style.axis_line_color,
        legend_bg_color=resolved_style.legend_bg_color,
        legend_border_color=resolved_style.legend_border_color,
        legend_border_width=resolved_style.legend_border_width,
        legend_orientation=resolved_style.legend_orientation,
        legend_x=resolved_style.legend_x,
        legend_y=resolved_style.legend_y,
        legend_xanchor=resolved_style.legend_xanchor,
        legend_yanchor=resolved_style.legend_yanchor,
        default_line_width=resolved_style.default_line_width,
        default_marker_size=resolved_style.default_marker_size,
    )
    return resolved_style


def plotly_legend_layout(
    style: PlotStyleSpec | None = None,
    **overrides: object,
) -> dict[str, object]:
    """
    Summary
    -------
    Return the shared Plotly legend layout dictionary.

    Parameters
    ----------
    style : PlotStyleSpec | None, default=None
        Optional explicit style specification. When omitted, the central style
        file is loaded automatically.
    **overrides : object
        Optional Plotly legend-layout keys to override for one figure.

    Returns
    -------
    dict[str, object]
        Legend layout mapping suitable for `fig.update_layout(legend=...)`.
    """
    resolved_style = load_plot_style() if style is None else style
    layout: dict[str, object] = {
        "font": {"size": resolved_style.legend_font_size},
        "bgcolor": resolved_style.legend_bg_color,
        "bordercolor": resolved_style.legend_border_color,
        "borderwidth": resolved_style.legend_border_width,
        "orientation": resolved_style.legend_orientation,
        "x": resolved_style.legend_x,
        "y": resolved_style.legend_y,
        "xanchor": resolved_style.legend_xanchor,
        "yanchor": resolved_style.legend_yanchor,
    }
    layout.update(overrides)
    return layout


def apply_matplotlib_style(style: PlotStyleSpec | None = None) -> PlotStyleSpec:
    """
    Summary
    -------
    Apply the shared style to Matplotlib rcParams.

    Parameters
    ----------
    style : PlotStyleSpec | None, default=None
        Optional explicit style specification. When omitted, the central style
        file is loaded automatically.

    Returns
    -------
    PlotStyleSpec
        The style specification that was applied.

    Notes
    -----
    The repository only uses Matplotlib in a small number of fallback and
    diagnostic plots. This function keeps those figures visually aligned with
    the Plotly outputs used in the paper.

    Examples
    --------
    >>> style = apply_matplotlib_style()
    >>> style.matplotlib_dpi >= 100
    True
    """
    import matplotlib as mpl

    resolved_style = load_plot_style() if style is None else style
    mpl.rcParams.update(
        {
            "figure.dpi": resolved_style.matplotlib_dpi,
            "savefig.dpi": resolved_style.matplotlib_dpi,
            "font.family": resolved_style.font_family,
            "font.size": resolved_style.tick_font_size,
            "axes.titlesize": resolved_style.title_font_size,
            "axes.labelsize": resolved_style.label_font_size,
            "axes.edgecolor": resolved_style.axis_line_color,
            "axes.labelcolor": resolved_style.font_color,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.facecolor": resolved_style.bg_color,
            "axes.linewidth": resolved_style.reference_line_width,
            "grid.color": resolved_style.grid_color,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "legend.fontsize": resolved_style.legend_font_size,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": resolved_style.bg_color,
            "legend.edgecolor": resolved_style.legend_border_color,
            "xtick.labelsize": resolved_style.tick_font_size,
            "ytick.labelsize": resolved_style.tick_font_size,
            "text.color": resolved_style.font_color,
            "axes.titlecolor": resolved_style.font_color,
            "xtick.color": resolved_style.font_color,
            "ytick.color": resolved_style.font_color,
        }
    )
    return resolved_style


_GLOBAL_PLOT_STYLE = load_plot_style()

THEME_COLORWAY: tuple[str, ...] = _GLOBAL_PLOT_STYLE.theme_colorway
THEME_GRID_COLOR = _GLOBAL_PLOT_STYLE.grid_color
THEME_BG_COLOR = _GLOBAL_PLOT_STYLE.bg_color
THEME_FONT_FAMILY = _GLOBAL_PLOT_STYLE.font_family
THEME_FONT_COLOR = _GLOBAL_PLOT_STYLE.font_color
THEME_AXIS_LINE_COLOR = _GLOBAL_PLOT_STYLE.axis_line_color
THEME_LEGEND_BG_COLOR = _GLOBAL_PLOT_STYLE.legend_bg_color
THEME_LEGEND_BORDER_COLOR = _GLOBAL_PLOT_STYLE.legend_border_color
THEME_LEGEND_BORDER_WIDTH = _GLOBAL_PLOT_STYLE.legend_border_width
