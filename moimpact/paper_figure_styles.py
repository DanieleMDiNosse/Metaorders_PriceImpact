"""Per-figure style overrides for generated paper figures.

This module intentionally keeps the configuration generic: each generating
workflow can opt in by looking up its output stem and applying the keys it
supports.  That lets the paper use different canvas/font settings for figures
that are placed differently in LaTeX, without solving readability via
``\\includegraphics`` resizing.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral, Real
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from moimpact.config import load_yaml_mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_FIGURE_STYLES_PATH = _REPO_ROOT / "config_ymls" / "paper_figure_styles.yml"
PAPER_FIGURE_STYLES_ENV = "PAPER_FIGURE_STYLES_CONFIG"
PAPER_FIGURE_STYLE_MODE_ENV = "PAPER_FIGURE_STYLE_MODE"
PAPER_FIGURE_STYLE_MODE_GLOBAL = "global"
PAPER_FIGURE_STYLE_MODE_PER_FIGURE = "per-figure"
PAPER_FIGURE_STYLE_MODES = frozenset(
    {PAPER_FIGURE_STYLE_MODE_GLOBAL, PAPER_FIGURE_STYLE_MODE_PER_FIGURE}
)

_SIZE_KEYS = {"width", "height"}
_FONT_KEYS = {
    "tick_font_size",
    "label_font_size",
    "title_font_size",
    "legend_font_size",
    "annotation_font_size",
}
_FLOAT_KEYS = {"line_width", "reference_line_width"}
_BOOL_KEYS = {"showlegend"}
_COLOR_KEYS = {"buy_color", "sell_color", "mean_color", "line_color"}
_UNIT_INTERVAL_KEYS = {"band_alpha"}
_MARGIN_KEYS = {"l", "r", "t", "b", "pad"}
_MAPPING_KEYS = {"margin"}
_ALLOWED_KEYS = (
    _SIZE_KEYS
    | _FONT_KEYS
    | _FLOAT_KEYS
    | _BOOL_KEYS
    | _COLOR_KEYS
    | _UNIT_INTERVAL_KEYS
    | _MAPPING_KEYS
)


def resolve_paper_figure_styles_path(path: Optional[str | Path] = None) -> Path:
    """Return the active paper-figure override config path."""
    raw = path if path is not None else os.environ.get(PAPER_FIGURE_STYLES_ENV)
    candidate = Path(raw).expanduser() if raw else DEFAULT_PAPER_FIGURE_STYLES_PATH
    if not candidate.is_absolute():
        candidate = (_REPO_ROOT / candidate).resolve()
    return candidate


def resolve_paper_figure_style_mode(mode: Optional[str] = None) -> str:
    """Return the active paper-figure style mode.

    ``global`` means generated figures use only the normal workflow/global
    plotting style. ``per-figure`` additionally applies
    ``paper_figure_styles.yml`` defaults and per-output-stem overrides.
    """
    raw = mode if mode is not None else os.environ.get(PAPER_FIGURE_STYLE_MODE_ENV)
    if raw is None or str(raw).strip() == "":
        return PAPER_FIGURE_STYLE_MODE_GLOBAL
    normalized = str(raw).strip().lower().replace("_", "-")
    aliases = {
        "perfigure": PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
        "figure": PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
        "figures": PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
        "specific": PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
        "global-only": PAPER_FIGURE_STYLE_MODE_GLOBAL,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PAPER_FIGURE_STYLE_MODES:
        allowed = ", ".join(sorted(PAPER_FIGURE_STYLE_MODES))
        raise ValueError(
            f"Invalid paper figure style mode {raw!r}. Allowed values: {allowed}."
        )
    return normalized


@lru_cache(maxsize=8)
def _load_paper_figure_styles_cached(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        if path == DEFAULT_PAPER_FIGURE_STYLES_PATH:
            return {}
        raise FileNotFoundError(f"Missing paper figure styles config: {path}")
    cfg = load_yaml_mapping(path)
    return cfg


def load_paper_figure_styles(path: Optional[str | Path] = None) -> dict[str, Any]:
    """Load the active paper-figure style config as a YAML mapping."""
    resolved = resolve_paper_figure_styles_path(path)
    return dict(_load_paper_figure_styles_cached(str(resolved)))


def _parse_positive_int(value: Any, *, key: str, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a positive integer, got {value!r}.")
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text.isdigit():
            raise ValueError(f"{context}.{key} must be a positive integer, got {value!r}.")
        parsed = int(text)
    else:
        raise ValueError(f"{context}.{key} must be a positive integer, got {value!r}.")
    if parsed <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer, got {value!r}.")
    return parsed


def _parse_nonnegative_int(value: Any, *, key: str, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a non-negative integer, got {value!r}.")
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text.isdigit():
            raise ValueError(f"{context}.{key} must be a non-negative integer, got {value!r}.")
        parsed = int(text)
    else:
        raise ValueError(f"{context}.{key} must be a non-negative integer, got {value!r}.")
    if parsed < 0:
        raise ValueError(f"{context}.{key} must be a non-negative integer, got {value!r}.")
    return parsed


def _parse_positive_float(value: Any, *, key: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context}.{key} must be a positive number, got {value!r}.")
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{context}.{key} must be a positive number, got {value!r}.")
    return parsed


def _parse_unit_interval_float(value: Any, *, key: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context}.{key} must be a number in [0, 1], got {value!r}.")
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise ValueError(f"{context}.{key} must be a number in [0, 1], got {value!r}.")
    return parsed


def _parse_bool(value: Any, *, key: str, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{context}.{key} must be a boolean, got {value!r}.")


def _parse_color(value: Any, *, key: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty color string, got {value!r}.")
    return value.strip()


def _normalize_margin(value: Any, *, context: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.margin must be a mapping, got {value!r}.")
    out: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key not in _MARGIN_KEYS:
            allowed = ", ".join(sorted(_MARGIN_KEYS))
            raise ValueError(f"Unknown margin key {context}.margin.{key!r}. Allowed: {allowed}.")
        out[key] = _parse_nonnegative_int(raw_value, key=f"margin.{key}", context=context)
    return out


def normalize_paper_figure_style(style: Mapping[str, Any], *, context: str = "figure") -> dict[str, Any]:
    """Validate and normalize a single figure-style mapping."""
    out: dict[str, Any] = {}
    for raw_key, value in style.items():
        key = str(raw_key)
        if key not in _ALLOWED_KEYS:
            allowed = ", ".join(sorted(_ALLOWED_KEYS))
            raise ValueError(f"Unknown paper figure style key {context}.{key!r}. Allowed: {allowed}.")
        if key in _SIZE_KEYS or key in _FONT_KEYS:
            out[key] = _parse_positive_int(value, key=key, context=context)
        elif key in _FLOAT_KEYS:
            out[key] = _parse_positive_float(value, key=key, context=context)
        elif key in _UNIT_INTERVAL_KEYS:
            out[key] = _parse_unit_interval_float(value, key=key, context=context)
        elif key in _BOOL_KEYS:
            out[key] = _parse_bool(value, key=key, context=context)
        elif key in _COLOR_KEYS:
            out[key] = _parse_color(value, key=key, context=context)
        elif key == "margin":
            out[key] = _normalize_margin(value, context=context)
    return out


def _identifier_candidates(identifier: str | Path) -> tuple[str, ...]:
    raw = Path(identifier).as_posix() if isinstance(identifier, Path) else str(identifier).strip()
    path = Path(raw)
    no_suffix = path.with_suffix("").as_posix() if path.suffix else raw
    candidates = [raw, no_suffix, path.stem]
    # Preserve order while removing duplicates and empty values.
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def paper_figure_style(
    identifier: str | Path,
    *,
    path: Optional[str | Path] = None,
    mode: Optional[str] = None,
) -> dict[str, Any]:
    """
    Return merged style overrides for a generated paper figure.

    ``identifier`` may be a filename stem, filename, or paper image path. The
    lookup tries exact, suffix-stripped, and basename-stem keys in that order.
    In ``global`` style mode the per-figure config is intentionally ignored and
    an empty mapping is returned.
    """
    style_mode = resolve_paper_figure_style_mode(mode)
    if style_mode == PAPER_FIGURE_STYLE_MODE_GLOBAL:
        return {}

    cfg = load_paper_figure_styles(path)
    defaults_raw = cfg.get("defaults", {})
    if defaults_raw is None:
        defaults_raw = {}
    if not isinstance(defaults_raw, Mapping):
        raise TypeError("paper_figure_styles.yml key 'defaults' must be a mapping when present.")
    merged = normalize_paper_figure_style(defaults_raw, context="defaults")

    figures = cfg.get("figures", {})
    if figures is None:
        figures = {}
    if not isinstance(figures, Mapping):
        raise TypeError("paper_figure_styles.yml key 'figures' must be a mapping when present.")

    for candidate in _identifier_candidates(identifier):
        if candidate not in figures:
            continue
        figure_raw = figures[candidate]
        if figure_raw is None:
            figure_raw = {}
        if not isinstance(figure_raw, Mapping):
            raise TypeError(f"paper_figure_styles.yml figure entry {candidate!r} must be a mapping.")
        merged.update(normalize_paper_figure_style(figure_raw, context=f"figures.{candidate}"))
        break
    return merged


def plotly_size_from_paper_style(
    style: Mapping[str, Any],
    *,
    default_width: Optional[int] = None,
    default_height: Optional[int] = None,
) -> dict[str, int]:
    """Build Plotly ``width``/``height`` kwargs from a normalized style mapping."""
    width = style.get("width", default_width)
    height = style.get("height", default_height)
    out: dict[str, int] = {}
    if width is not None:
        out["width"] = int(width)
    if height is not None:
        out["height"] = int(height)
    return out


def apply_plotly_paper_figure_style(
    fig: Any,
    identifier: str | Path,
    *,
    default_width: Optional[int] = None,
    default_height: Optional[int] = None,
    default_tick_font_size: Optional[int] = None,
    default_label_font_size: Optional[int] = None,
    default_title_font_size: Optional[int] = None,
    default_legend_font_size: Optional[int] = None,
    default_annotation_font_size: Optional[int] = None,
    default_line_width: Optional[float] = None,
    default_reference_line_width: Optional[float] = None,
) -> dict[str, Any]:
    """Apply configured paper-figure overrides to a Plotly figure and return them.

    The function is intentionally conservative: only keys present in
    ``paper_figure_styles.yml`` (or explicit defaults passed by the caller) are
    applied. Static export dimensions are not returned as keyword arguments;
    callers can pass ``plotly_size_from_paper_style(style)`` to their export
    helper when they want PNG/PDF dimensions to match the layout.
    """
    style = paper_figure_style(identifier)
    size = plotly_size_from_paper_style(
        style,
        default_width=default_width,
        default_height=default_height,
    )

    tick_size = style.get("tick_font_size", default_tick_font_size)
    label_size = style.get("label_font_size", default_label_font_size)
    title_size = style.get("title_font_size", default_title_font_size)
    legend_size = style.get("legend_font_size", default_legend_font_size)
    annotation_size = style.get("annotation_font_size", default_annotation_font_size)
    line_width = style.get("line_width", default_line_width)
    reference_line_width = style.get("reference_line_width", default_reference_line_width)

    layout_updates: dict[str, Any] = {}
    layout_updates.update(size)
    if label_size is not None:
        layout_updates["font"] = {"size": int(label_size)}
    if title_size is not None:
        layout_updates["title_font"] = {"size": int(title_size)}
    if "showlegend" in style:
        layout_updates["showlegend"] = bool(style["showlegend"])
    if "margin" in style:
        layout_updates["margin"] = dict(style["margin"])
    if layout_updates:
        fig.update_layout(**layout_updates)
    if legend_size is not None:
        fig.update_layout(legend={"font": {"size": int(legend_size)}})
    if tick_size is not None or label_size is not None:
        axis_updates: dict[str, Any] = {}
        if tick_size is not None:
            axis_updates["tickfont"] = {"size": int(tick_size)}
        if label_size is not None:
            axis_updates["title_font"] = {"size": int(label_size)}
        fig.update_xaxes(**axis_updates)
        fig.update_yaxes(**axis_updates)
    if annotation_size is not None:
        fig.update_annotations(font_size=int(annotation_size))
    if line_width is not None:
        for trace in fig.data:
            mode = getattr(trace, "mode", None)
            if mode is None or "lines" not in str(mode):
                continue
            if getattr(trace, "line", None) is not None:
                existing_width = getattr(trace.line, "width", None)
                if existing_width == 0:
                    continue
                trace.line.width = float(line_width)
    if reference_line_width is not None:
        for shape in fig.layout.shapes or ():
            if getattr(shape, "type", None) != "line":
                continue
            if getattr(shape, "line", None) is not None:
                shape.line.width = float(reference_line_width)
    return style
