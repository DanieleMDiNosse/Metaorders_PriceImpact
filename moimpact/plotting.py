"""
Unified Plotly-first plotting helpers shared across repository scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import os
from pathlib import Path
from typing import Mapping, Optional, Tuple

from moimpact.paper_figure_styles import (
    apply_plotly_paper_figure_style,
    plotly_size_from_paper_style,
)
from moimpact.plot_style import THEME_COLORWAY, load_plot_style

_PLOT_STYLE = load_plot_style()

# Canonical group colors used across scripts.
COLOR_PROPRIETARY = THEME_COLORWAY[0]
COLOR_CLIENT = THEME_COLORWAY[2]
COLOR_NEUTRAL = "#6B7280"
COLOR_BAND_PROPRIETARY = "rgba(91,143,249,0.20)"
COLOR_BAND_CLIENT = "rgba(238,102,102,0.20)"

# Canonical static export settings (used for Plotly PNG outputs).
PLOTLY_EXPORT_WIDTH = _PLOT_STYLE.export_width
PLOTLY_EXPORT_HEIGHT = _PLOT_STYLE.export_height
PLOTLY_EXPORT_SCALE = _PLOT_STYLE.export_scale


def _env_flag(name: str, *, default: bool = False) -> bool:
    """
    Parse a boolean environment variable.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : bool, default=False
        Value returned when the variable is unset.

    Returns
    -------
    bool
        Parsed boolean value.

    Notes
    -----
    Accepted true values: ``1``, ``true``, ``yes``, ``on``.
    Accepted false values: ``0``, ``false``, ``no``, ``off``.
    Any other non-empty value falls back to ``default``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_optional_positive_int(value: object, *, key: str) -> Optional[int]:
    """Parse a nullable positive-integer figure-size config value."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer or null, got {value!r}.")
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
        if not normalized.isdigit():
            raise ValueError(f"{key} must be a positive integer or null, got {value!r}.")
        parsed = int(normalized)
    else:
        raise ValueError(f"{key} must be a positive integer or null, got {value!r}.")
    if parsed <= 0:
        raise ValueError(f"{key} must be a positive integer or null, got {value!r}.")
    return parsed


def plotly_figure_size_from_config(
    cfg: Mapping[str, object],
    *,
    width_key: str = "IMPACT_FIT_FIGURE_WIDTH",
    height_key: str = "IMPACT_FIT_FIGURE_HEIGHT",
) -> dict[str, int]:
    """
    Return optional Plotly canvas dimensions from a YAML mapping.

    The result uses Plotly's ``width``/``height`` keyword names. Missing or null
    config values are omitted so callers can preserve existing defaults.
    """
    size: dict[str, int] = {}
    width = _parse_optional_positive_int(cfg.get(width_key), key=width_key)
    height = _parse_optional_positive_int(cfg.get(height_key), key=height_key)
    if width is not None:
        size["width"] = width
    if height is not None:
        size["height"] = height
    return size


def plotly_layout_size_kwargs(
    figure_size: Optional[Mapping[str, int]],
    *,
    default_width: Optional[int] = None,
    default_height: Optional[int] = None,
) -> dict[str, int]:
    """Build Plotly layout size kwargs, honoring explicit overrides first."""
    width = figure_size.get("width") if figure_size is not None else None
    height = figure_size.get("height") if figure_size is not None else None
    if width is None:
        width = default_width
    if height is None:
        height = default_height

    out: dict[str, int] = {}
    if width is not None:
        out["width"] = int(width)
    if height is not None:
        out["height"] = int(height)
    return out


def plotly_export_size_kwargs(figure_size: Optional[Mapping[str, int]]) -> dict[str, int]:
    """Build static-export size kwargs for ``save_plotly_figure``."""
    if not figure_size:
        return {}
    return {key: int(value) for key, value in figure_size.items() if key in {"width", "height"}}


@dataclass(frozen=True)
class PlotOutputDirs:
    """
    Summary
    -------
    Container for the canonical output folders used by Plotly exports.

    Parameters
    ----------
    base_dir : Path
        Root output directory for the analysis.
    html_dir : Path
        Directory where interactive HTML figures are written.
    png_dir : Path
        Directory where static PNG figures are written.

    Returns
    -------
    PlotOutputDirs
        Dataclass instance with resolved directory paths.

    Notes
    -----
    This keeps file-path handling explicit and avoids per-script conventions.

    Examples
    --------
    >>> from pathlib import Path
    >>> d = PlotOutputDirs(base_dir=Path("images/demo"), html_dir=Path("images/demo/html"), png_dir=Path("images/demo/png"))
    >>> d.base_dir.name
    'demo'
    """

    base_dir: Path
    html_dir: Path
    png_dir: Path


def make_plot_output_dirs(base_dir: Path, *, use_subdirs: bool = True) -> PlotOutputDirs:
    """
    Summary
    -------
    Build canonical HTML/PNG output directories from a base folder.

    Parameters
    ----------
    base_dir : Path
        Root figure output directory for an analysis/script.
    use_subdirs : bool, default=True
        If True, place files in `base_dir/html` and `base_dir/png`.
        If False, both HTML and PNG are written directly under `base_dir`.

    Returns
    -------
    PlotOutputDirs
        Resolved output directory container.

    Notes
    -----
    This function does not create folders on disk; call `ensure_plot_dirs`
    before writing files.

    Examples
    --------
    >>> from pathlib import Path
    >>> dirs = make_plot_output_dirs(Path("images/ftsemib/member_proprietary"))
    >>> dirs.html_dir.name, dirs.png_dir.name
    ('html', 'png')
    """
    base = Path(base_dir)
    if use_subdirs:
        return PlotOutputDirs(base_dir=base, html_dir=base / "html", png_dir=base / "png")
    return PlotOutputDirs(base_dir=base, html_dir=base, png_dir=base)


def ensure_plot_dirs(dirs: PlotOutputDirs) -> None:
    """
    Summary
    -------
    Create output directories for Plotly figures if they do not exist.

    Parameters
    ----------
    dirs : PlotOutputDirs
        Directory container from `make_plot_output_dirs`.

    Returns
    -------
    None

    Notes
    -----
    Directory creation is idempotent (`exist_ok=True`).

    Examples
    --------
    >>> from pathlib import Path
    >>> d = make_plot_output_dirs(Path("images/demo"))
    >>> ensure_plot_dirs(d)
    """
    dirs.base_dir.mkdir(parents=True, exist_ok=True)
    dirs.html_dir.mkdir(parents=True, exist_ok=True)
    dirs.png_dir.mkdir(parents=True, exist_ok=True)


def save_plotly_figure(
    fig,
    *,
    stem: str,
    dirs: PlotOutputDirs,
    width: int = PLOTLY_EXPORT_WIDTH,
    height: int = PLOTLY_EXPORT_HEIGHT,
    scale: int = PLOTLY_EXPORT_SCALE,
    include_plotlyjs: str = "cdn",
    include_mathjax: str = "cdn",
    write_html: bool = True,
    write_png: bool = True,
    write_pdf: bool = False,
    strict_png: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Summary
    -------
    Save a Plotly figure with a unified HTML/PNG convention.

    Parameters
    ----------
    fig
        Plotly figure object (typically `plotly.graph_objects.Figure`).
    stem : str
        Output filename stem (without extension).
    dirs : PlotOutputDirs
        Output directory container.
    width : int, default=PLOTLY_EXPORT_WIDTH
        Static PNG export width in pixels.
    height : int, default=PLOTLY_EXPORT_HEIGHT
        Static PNG export height in pixels.
    scale : int, default=PLOTLY_EXPORT_SCALE
        Plotly export scale multiplier for PNG quality.
    include_plotlyjs : str, default="cdn"
        Plotly JS inclusion mode for HTML exports.
    include_mathjax : str, default="cdn"
        MathJax inclusion mode for HTML exports. The default ensures LaTeX
        text in annotations, titles, and axis labels renders in saved HTML.
    write_html : bool, default=True
        Whether to save the interactive HTML file.
    write_png : bool, default=True
        Whether to save the static PNG file.
    write_pdf : bool, default=False
        Whether to also save a vector PDF version beside the PNG file. This can
        also be enabled globally with `PLOTLY_WRITE_PDF=true`.
    strict_png : bool, default=False
        If True, re-raise static export errors (for PNG and optional PDF).

    Returns
    -------
    tuple[Optional[Path], Optional[Path]]
        `(html_path, png_path)` for files requested/successfully written.
        If PNG export fails with `strict_png=False`, returns `None` for `png_path`.

    Notes
    -----
    - Static PNG/PDF export relies on Plotly's image export backend (usually
      `kaleido`).
    - HTML exports load MathJax by default so saved figures preserve LaTeX
      formatting used across the research plots.
    - PDF files are written to the same directory as PNG files so LaTeX can
      reference extensionless stems and prefer the vector export automatically.

    Examples
    --------
    >>> # Example omitted: requires a Plotly Figure instance.
    """
    ensure_plot_dirs(dirs)
    html_path: Optional[Path] = None
    png_path: Optional[Path] = None
    write_pdf = bool(write_pdf or _env_flag("PLOTLY_WRITE_PDF", default=False))

    # Pipeline-level control: hide all figure legends when requested.
    if _env_flag("DISABLE_PLOT_LEGENDS", default=False):
        fig.update_layout(showlegend=False)

    paper_style = apply_plotly_paper_figure_style(fig, stem)
    paper_size = plotly_size_from_paper_style(paper_style)
    if "width" in paper_size:
        width = paper_size["width"]
    if "height" in paper_size:
        height = paper_size["height"]

    if write_html:
        html_path = dirs.html_dir / f"{stem}.html"
        fig.write_html(
            str(html_path),
            include_plotlyjs=include_plotlyjs,
            include_mathjax=include_mathjax,
        )

    if write_png:
        png_path_tmp = dirs.png_dir / f"{stem}.png"
        try:
            fig.write_image(str(png_path_tmp), width=width, height=height, scale=scale)
            png_path = png_path_tmp
        except Exception:
            if strict_png:
                raise
            png_path = None

    if write_pdf:
        pdf_path = dirs.png_dir / f"{stem}.pdf"
        try:
            fig.write_image(str(pdf_path), width=width, height=height)
        except Exception:
            if strict_png:
                raise

    return html_path, png_path
