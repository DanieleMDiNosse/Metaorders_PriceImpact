"""
Unified Plotly-first plotting helpers shared across repository scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from moimpact.plot_style import THEME_COLORWAY

# Canonical group colors used across scripts.
COLOR_PROPRIETARY = THEME_COLORWAY[0]
COLOR_CLIENT = THEME_COLORWAY[2]
COLOR_NEUTRAL = "#6B7280"
COLOR_BAND_PROPRIETARY = "rgba(91,143,249,0.20)"
COLOR_BAND_CLIENT = "rgba(238,102,102,0.20)"

# Canonical static export settings (used for Plotly PNG outputs).
PLOTLY_EXPORT_WIDTH = 1200
PLOTLY_EXPORT_HEIGHT = 700
PLOTLY_EXPORT_SCALE = 2


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
    write_html: bool = True,
    write_png: bool = True,
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
    write_html : bool, default=True
        Whether to save the interactive HTML file.
    write_png : bool, default=True
        Whether to save the static PNG file.
    strict_png : bool, default=False
        If True, re-raise PNG export errors (e.g., missing kaleido).

    Returns
    -------
    tuple[Optional[Path], Optional[Path]]
        `(html_path, png_path)` for files requested/successfully written.
        If PNG export fails with `strict_png=False`, returns `None` for `png_path`.

    Notes
    -----
    PNG export relies on Plotly's static export backend (usually `kaleido`).

    Examples
    --------
    >>> # Example omitted: requires a Plotly Figure instance.
    """
    ensure_plot_dirs(dirs)
    html_path: Optional[Path] = None
    png_path: Optional[Path] = None

    if write_html:
        html_path = dirs.html_dir / f"{stem}.html"
        fig.write_html(str(html_path), include_plotlyjs=include_plotlyjs)

    if write_png:
        png_path_tmp = dirs.png_dir / f"{stem}.png"
        try:
            fig.write_image(str(png_path_tmp), width=width, height=height, scale=scale)
            png_path = png_path_tmp
        except Exception:
            if strict_png:
                raise
            png_path = None

    return html_path, png_path
