#!/usr/bin/env python3
"""Create a no-data visual preview build of paper figures.

This workflow is intentionally separate from ``paper figures``. It does not
load CONSOB data, parquet files, or generated result tables. Instead, it copies
LaTeX source files into a scratch directory, creates fake figure images at the
same relative paths used by ``\\includegraphics``, and optionally compiles the
scratch paper so figure typography can be inspected after LaTeX scaling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from moimpact.paper_figure_styles import (
    DEFAULT_PAPER_FIGURE_STYLES_PATH,
    PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
    PAPER_FIGURE_STYLE_MODES,
    paper_figure_style,
    resolve_paper_figure_style_mode,
)
from moimpact.plot_style import load_plot_style

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PAPER_TEX = _REPO_ROOT / "paper" / "main.tex"
_DEFAULT_PLOT_STYLE_CONFIG = _REPO_ROOT / "config_ymls" / "plot_style.yml"

_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
_KNOWN_IMAGE_SUFFIXES = frozenset({".png", ".pdf", ".jpg", ".jpeg", ".svg", ".eps"})
_SUPPORTED_PLACEHOLDER_SUFFIXES = frozenset({".png", ".pdf", ".jpg", ".jpeg"})
_BUILD_ARTIFACT_SUFFIXES = frozenset(
    {
        ".aux",
        ".bbl",
        ".bcf",
        ".blg",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".run.xml",
        ".synctex.gz",
        ".toc",
    }
)
_BUILD_ARTIFACT_NAMES = frozenset({"main.pdf"})


@dataclass(frozen=True)
class PlaceholderRecord:
    """Metadata for one generated placeholder figure."""

    include_path: str
    generated_path: str
    width: int
    height: int
    layout: str
    style: dict[str, Any]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a temporary no-data paper build with fake figures that use "
            "paper_figure_styles.yml typography and canvas settings."
        )
    )
    parser.add_argument(
        "--paper-tex",
        default=str(_DEFAULT_PAPER_TEX),
        help="LaTeX source to copy and inspect. Default: paper/main.tex.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Scratch output directory. Default: tmp_paper_style_preview_<timestamp> "
            "under the repository root."
        ),
    )
    parser.add_argument(
        "--style-config",
        "--config-path",
        dest="style_config",
        default=str(DEFAULT_PAPER_FIGURE_STYLES_PATH),
        help="Per-figure style YAML. Default: config_ymls/paper_figure_styles.yml.",
    )
    parser.add_argument(
        "--plot-style-config",
        default=str(_DEFAULT_PLOT_STYLE_CONFIG),
        help="Global plot style YAML used for fallback fonts/colors. Default: config_ymls/plot_style.yml.",
    )
    parser.add_argument(
        "--style-mode",
        choices=tuple(sorted(PAPER_FIGURE_STYLE_MODES)),
        default=PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
        help="Style lookup mode. Default: per-figure.",
    )
    parser.add_argument(
        "--figures",
        nargs="*",
        default=(),
        help=(
            "Optional figure paths or unique basenames to preview. By default every "
            "non-commented includegraphics path in the LaTeX file is generated."
        ),
    )
    parser.add_argument(
        "--list-figures",
        action="store_true",
        help="List detected figure paths and exit without writing files.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpg", "jpeg"),
        default="png",
        help="Format to use when includegraphics omits an extension. Default: png.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help=(
            "Raster DPI for placeholder generation. Font sizes from style YAML are "
            "interpreted as pixels and converted to Matplotlib points at this DPI."
        ),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the scratch paper with latexmk after generating placeholders. Default: true.",
    )
    parser.add_argument(
        "--latexmk",
        default="latexmk",
        help="latexmk executable used when --compile is enabled. Default: latexmk.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow deleting an existing --output-root before generating the preview.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the no-data paper style preview workflow."""

    args = _parse_args(argv)
    paper_tex = _resolve_path(args.paper_tex, base=_REPO_ROOT)
    if not paper_tex.exists():
        raise FileNotFoundError(f"Missing paper source: {paper_tex}")

    detected_figures = _paper_figure_paths(paper_tex)
    if args.list_figures:
        for figure in detected_figures:
            print(figure)
        return 0

    style_config = _resolve_path(args.style_config, base=_REPO_ROOT)
    plot_style_config = _resolve_path(args.plot_style_config, base=_REPO_ROOT)
    style_mode = resolve_paper_figure_style_mode(args.style_mode)

    if args.dpi <= 0:
        raise ValueError(f"--dpi must be positive, got {args.dpi}.")
    if not style_config.exists() and style_mode == PAPER_FIGURE_STYLE_MODE_PER_FIGURE:
        raise FileNotFoundError(f"Missing paper figure style config: {style_config}")
    if not plot_style_config.exists():
        raise FileNotFoundError(f"Missing plot style config: {plot_style_config}")

    selected_figures = _select_figures(detected_figures, args.figures)
    if not selected_figures:
        raise ValueError("No figures selected for preview.")

    output_root = _resolve_output_root(args.output_root)
    _validate_output_root(output_root, paper_source_dir=paper_tex.parent)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --overwrite to replace it."
            )
        _validate_overwrite_target(output_root)
        shutil.rmtree(output_root)

    scratch_paper_dir = output_root / "paper"
    scratch_paper_tex = scratch_paper_dir / paper_tex.name
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / ".paper_style_preview").write_text(
        "This directory was created by paper style-preview and is safe to overwrite.\n",
        encoding="utf-8",
    )

    _copy_paper_sources(paper_tex.parent, scratch_paper_dir)

    plot_style = load_plot_style(plot_style_config)
    records: list[PlaceholderRecord] = []
    for include_path in selected_figures:
        style = paper_figure_style(include_path, path=style_config, mode=style_mode)
        target_path = _placeholder_path(
            scratch_paper_dir,
            include_path,
            default_format=str(args.image_format),
        )
        record = _generate_placeholder(
            include_path=include_path,
            target_path=target_path,
            style=style,
            plot_style=plot_style,
            dpi=int(args.dpi),
        )
        records.append(record)

    compile_info: dict[str, Any] = {"requested": bool(args.compile), "success": None}
    if args.compile:
        compile_info = _compile_scratch_paper(
            scratch_paper_dir=scratch_paper_dir,
            tex_name=scratch_paper_tex.name,
            latexmk_executable=str(args.latexmk),
            log_path=output_root / "latexmk.log",
        )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_paper_tex": str(paper_tex),
        "scratch_paper_tex": str(scratch_paper_tex),
        "output_root": str(output_root),
        "style_config": str(style_config),
        "plot_style_config": str(plot_style_config),
        "style_mode": style_mode,
        "dpi": int(args.dpi),
        "compile": compile_info,
        "figures_detected": list(detected_figures),
        "figures_selected": list(selected_figures),
        "placeholders": [record.__dict__ for record in records],
    }
    manifest_path = output_root / "preview_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[preview] root: {output_root}")
    print(f"[preview] paper tex: {scratch_paper_tex}")
    print(f"[preview] placeholders: {len(records)}")
    print(f"[preview] manifest: {manifest_path}")
    if args.compile:
        pdf_path = scratch_paper_dir / f"{scratch_paper_tex.stem}.pdf"
        print(f"[preview] pdf: {pdf_path}")
    else:
        print("[preview] compile skipped (--no-compile)")
    return 0


def _resolve_path(raw: str | Path, *, base: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _resolve_output_root(raw: str | None) -> Path:
    if raw:
        return _resolve_path(raw, base=_REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (_REPO_ROOT / f"tmp_paper_style_preview_{timestamp}").resolve()


def _validate_output_root(output_root: Path, *, paper_source_dir: Path) -> None:
    resolved = output_root.resolve(strict=False)
    repo_root = _REPO_ROOT.resolve(strict=False)
    paper_dir = paper_source_dir.resolve(strict=False)
    paper_images_dir = (paper_source_dir / "images").resolve(strict=False)
    dangerous_exact = {repo_root, paper_dir, paper_images_dir}
    if resolved in dangerous_exact:
        raise ValueError(f"Refusing to use dangerous preview output root: {output_root}")
    if _is_relative_to(resolved, paper_dir):
        raise ValueError(f"Refusing to write preview output inside the paper source directory: {output_root}")
    if _is_relative_to(resolved, paper_images_dir):
        raise ValueError(f"Refusing to write preview output inside real paper/images: {output_root}")


def _validate_overwrite_target(output_root: Path) -> None:
    marker = output_root / ".paper_style_preview"
    if marker.exists() or output_root.name.startswith("tmp_paper_style_preview_"):
        return
    raise ValueError(
        "Refusing to overwrite a directory that does not look like a paper style-preview scratch root: "
        f"{output_root}. Use a tmp_paper_style_preview_* directory or an existing preview directory "
        "containing .paper_style_preview."
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _paper_figure_paths(paper_tex_path: Path) -> tuple[str, ...]:
    figures: list[str] = []
    for raw_line in paper_tex_path.read_text(encoding="utf-8").splitlines():
        active_line = _strip_latex_comment(raw_line)
        if not active_line.strip():
            continue
        for match in _INCLUDEGRAPHICS_RE.finditer(active_line):
            figures.append(match.group(1).strip())
    return tuple(dict.fromkeys(figures))


def _strip_latex_comment(line: str) -> str:
    for index, char in enumerate(line):
        if char == "%" and not _is_escaped_percent(line, index):
            return line[:index]
    return line


def _is_escaped_percent(line: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and line[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _figure_path_id(path_like: str | Path) -> str:
    path = Path(str(path_like).strip())
    return path.with_suffix("").as_posix() if path.suffix else path.as_posix()


def _figure_basename_id(path_like: str | Path) -> str:
    path = Path(str(path_like).strip())
    return path.stem if path.suffix else path.name


def _select_figures(all_figures: Sequence[str], requested_figures: Sequence[str]) -> tuple[str, ...]:
    if not requested_figures:
        return tuple(all_figures)
    selected: list[str] = []
    for requested in requested_figures:
        selected.append(_resolve_requested_figure(requested, all_figures))
    selected_set = set(selected)
    return tuple(figure for figure in all_figures if figure in selected_set)


def _resolve_requested_figure(requested: str, all_figures: Sequence[str]) -> str:
    requested_clean = requested.strip()
    if requested_clean in all_figures:
        return requested_clean

    normalized_matches = [
        figure for figure in all_figures if _figure_path_id(figure) == _figure_path_id(requested_clean)
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]
    if len(normalized_matches) > 1:
        raise ValueError(f"Ambiguous figure path {requested!r}: {normalized_matches}")

    basename_matches = [
        figure for figure in all_figures if _figure_basename_id(figure) == _figure_basename_id(requested_clean)
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        raise ValueError(f"Ambiguous figure basename {requested!r}: {basename_matches}")

    raise ValueError(f"Requested figure {requested!r} is not present in the paper source.")


def _copy_paper_sources(source_dir: Path, scratch_paper_dir: Path) -> None:
    scratch_paper_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.name == "images":
            continue
        if _is_build_artifact(item):
            continue
        destination = scratch_paper_dir / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                destination,
                ignore=lambda directory, names: [
                    name for name in names if _should_ignore_copy_item(Path(directory) / name)
                ],
            )
        elif item.is_file():
            shutil.copy2(item, destination)


def _should_ignore_copy_item(path: Path) -> bool:
    return path.name == "images" or _is_build_artifact(path) or path.name == "__pycache__"


def _is_build_artifact(path: Path) -> bool:
    return path.name in _BUILD_ARTIFACT_NAMES or any(
        path.name.endswith(suffix) for suffix in _BUILD_ARTIFACT_SUFFIXES
    )


def _placeholder_path(scratch_paper_dir: Path, include_path: str, *, default_format: str) -> Path:
    raw_path = Path(include_path)
    if raw_path.is_absolute():
        raise ValueError(f"Absolute includegraphics paths are not supported in preview mode: {include_path}")
    suffix = raw_path.suffix.lower()
    if suffix in _KNOWN_IMAGE_SUFFIXES:
        if suffix not in _SUPPORTED_PLACEHOLDER_SUFFIXES:
            raise ValueError(
                f"Cannot generate placeholder for {suffix!r} images. "
                "Use PNG/JPG/PDF paper figure paths or omit the extension."
            )
        target = scratch_paper_dir / raw_path
    else:
        target = (scratch_paper_dir / raw_path).with_suffix(f".{default_format}")
    resolved_target = target.resolve(strict=False)
    resolved_scratch = scratch_paper_dir.resolve(strict=False)
    if not _is_relative_to(resolved_target, resolved_scratch):
        raise ValueError(f"Refusing includegraphics path that escapes the scratch paper directory: {include_path}")
    return target


def _generate_placeholder(
    *,
    include_path: str,
    target_path: Path,
    style: Mapping[str, Any],
    plot_style: Any,
    dpi: int,
) -> PlaceholderRecord:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    width = int(style.get("width", plot_style.export_width))
    height = int(style.get("height", plot_style.export_height))
    tick_font_size = int(style.get("tick_font_size", plot_style.tick_font_size))
    label_font_size = int(style.get("label_font_size", plot_style.label_font_size))
    title_font_size = int(style.get("title_font_size", plot_style.title_font_size))
    legend_font_size = int(style.get("legend_font_size", plot_style.legend_font_size))
    annotation_font_size = int(style.get("annotation_font_size", plot_style.annotation_font_size))
    line_width = float(style.get("line_width", plot_style.default_line_width))
    reference_line_width = float(style.get("reference_line_width", plot_style.reference_line_width))
    showlegend = bool(style.get("showlegend", True))
    margin = dict(style.get("margin", {}))

    nrows, ncols, kind = _infer_placeholder_layout(include_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        squeeze=False,
        facecolor=plot_style.bg_color,
    )
    _apply_subplot_margins(fig, width=width, height=height, margin=margin, nrows=nrows, ncols=ncols)

    colors = list(plot_style.theme_colorway)
    if not colors:
        colors = ["#355C7D", "#C06C84"]
    stem = _figure_basename_id(include_path)

    for panel_index, ax in enumerate(axes.flat, start=1):
        _draw_panel(
            ax=ax,
            kind=kind,
            panel_index=panel_index,
            colors=colors,
            line_width=line_width,
            reference_line_width=reference_line_width,
            showlegend=showlegend and panel_index == 1,
            plot_style=plot_style,
        )
        ax.set_title(
            _panel_title(stem, panel_index=panel_index, total=nrows * ncols),
            fontsize=_px_to_pt(title_font_size, dpi),
            color=plot_style.font_color,
            pad=max(4.0, _px_to_pt(6, dpi)),
        )
        ax.set_xlabel(
            f"Axis label preview ({label_font_size}px)",
            fontsize=_px_to_pt(label_font_size, dpi),
            color=plot_style.font_color,
        )
        ax.set_ylabel(
            f"Tick/label scale ({tick_font_size}/{label_font_size}px)",
            fontsize=_px_to_pt(label_font_size, dpi),
            color=plot_style.font_color,
        )
        ax.tick_params(
            axis="both",
            labelsize=_px_to_pt(tick_font_size, dpi),
            colors=plot_style.font_color,
            length=6,
            width=1,
        )
        for spine in ax.spines.values():
            spine.set_color(plot_style.axis_line_color)

        if showlegend and panel_index == 1:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                legend = ax.legend(
                    handles,
                    labels,
                    fontsize=_px_to_pt(legend_font_size, dpi),
                    frameon=True,
                    loc="best",
                )
                if legend is not None:
                    legend.get_frame().set_facecolor("white")
                    legend.get_frame().set_edgecolor(plot_style.axis_line_color)

        ax.text(
            0.02,
            0.98,
            f"{_panel_letter(panel_index)} font preview",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=_px_to_pt(annotation_font_size, dpi),
            color=plot_style.font_color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 2},
        )

    fig.text(
        0.5,
        0.5,
        "FAKE DATA\nSTYLE PREVIEW",
        ha="center",
        va="center",
        fontsize=_px_to_pt(max(annotation_font_size, 18), dpi),
        color="#9CA3AF",
        alpha=0.22,
        rotation=18,
        weight="bold",
    )

    fig.savefig(target_path, dpi=dpi, facecolor=plot_style.bg_color)
    plt.close(fig)

    return PlaceholderRecord(
        include_path=include_path,
        generated_path=str(target_path),
        width=width,
        height=height,
        layout=f"{nrows}x{ncols}:{kind}",
        style=dict(style),
    )


def _px_to_pt(size_px: int | float, dpi: int) -> float:
    return max(1.0, float(size_px) * 72.0 / float(dpi))


def _infer_placeholder_layout(include_path: str) -> tuple[int, int, str]:
    stem = _figure_basename_id(include_path)
    if stem in {
        "eta_robustness_crowding_impact_curves",
    }:
        return 2, 2, "line"
    if stem in {
        "curve_mean_align_vs_eta_local",
        "curve_mean_abs_imb_vs_eta_local",
        "curve_mean_align_vs_eta_cross",
        "curve_mean_abs_imb_vs_eta_cross",
        "main_crowding_impact_curves",
        "execution_schedule_heatmap_prop_vs_client",
        "execution_schedule_heatmap_prop_vs_client_median",
        "member_metaorder_profiles_all_metaorders",
    }:
        kind = "heatmap" if "heatmap" in stem else "line"
        return 1, 2, kind
    if stem == "metaorder_distributions_prop_vs_client":
        return 2, 3, "distribution"
    if "heatmap" in stem:
        return 1, 1, "heatmap"
    if "hist" in stem or "distribution" in stem:
        return 1, 1, "distribution"
    if "coverage" in stem or "per_isin" in stem or "share" in stem:
        return 1, 1, "bar"
    return 1, 1, "line"


def _apply_subplot_margins(
    fig: Any,
    *,
    width: int,
    height: int,
    margin: Mapping[str, Any],
    nrows: int,
    ncols: int,
) -> None:
    left = _margin_fraction(margin.get("l"), width, fallback=0.10, low=0.04, high=0.35)
    right = 1.0 - _margin_fraction(margin.get("r"), width, fallback=0.04, low=0.02, high=0.35)
    top = 1.0 - _margin_fraction(margin.get("t"), height, fallback=0.07, low=0.02, high=0.28)
    bottom = _margin_fraction(margin.get("b"), height, fallback=0.11, low=0.04, high=0.35)

    if right <= left + 0.15:
        right = min(0.98, left + 0.15)
    if top <= bottom + 0.15:
        top = min(0.98, bottom + 0.15)

    fig.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=0.32 if ncols > 1 else 0.20,
        hspace=0.42 if nrows > 1 else 0.20,
    )


def _margin_fraction(raw_value: Any, size: int, *, fallback: float, low: float, high: float) -> float:
    if raw_value is None:
        return fallback
    try:
        value = float(raw_value) / float(size)
    except (TypeError, ValueError):
        return fallback
    return max(low, min(high, value))


def _draw_panel(
    *,
    ax: Any,
    kind: str,
    panel_index: int,
    colors: Sequence[str],
    line_width: float,
    reference_line_width: float,
    showlegend: bool,
    plot_style: Any,
) -> None:
    import numpy as np

    ax.set_facecolor(plot_style.bg_color)
    if kind == "heatmap":
        x = np.linspace(-2, 2, 40)
        y = np.linspace(-2, 2, 30)
        xx, yy = np.meshgrid(x, y)
        zz = np.sin(xx * panel_index) + np.cos(yy * 1.3)
        image = ax.imshow(zz, origin="lower", aspect="auto", cmap="viridis")
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=ax.xaxis.get_ticklabels()[0].get_fontsize() if ax.xaxis.get_ticklabels() else 8)
        return

    if kind == "bar":
        labels = ["A", "B", "C", "D", "E"]
        values = np.array([0.42, 0.71, 0.55, 0.84, 0.63]) * (1 + 0.03 * panel_index)
        ax.bar(labels, values, color=colors[0], label="Group A")
        ax.plot(labels, values * 0.82, color=colors[min(2, len(colors) - 1)], linewidth=line_width, label="Group B")
        ax.grid(True, axis="y", color=plot_style.grid_color, linewidth=0.8)
        return

    if kind == "distribution":
        x = np.logspace(-4, -1, 80)
        y1 = 0.06 * (x / x.min()) ** -0.32
        y2 = 0.04 * (x / x.min()) ** -0.24
        ax.loglog(x, y1, color=colors[0], linewidth=line_width, label="Proprietary")
        ax.loglog(x, y2, color=colors[min(2, len(colors) - 1)], linewidth=line_width, label="Client")
        ax.axvline(x[20], color="#6B7280", linestyle="--", linewidth=reference_line_width, label="cutoff")
        ax.grid(True, which="both", color=plot_style.grid_color, linewidth=0.8)
        return

    x = np.logspace(-4, -1, 60)
    y1 = 0.025 * (x / x.min()) ** (0.34 + 0.02 * panel_index)
    y2 = 0.018 * (x / x.min()) ** 0.50
    ax.loglog(x, y1, color=colors[0], linewidth=line_width, label="Proprietary")
    ax.loglog(x, y2, color=colors[min(2, len(colors) - 1)], linewidth=line_width, label="Client")
    ax.scatter(x[::9], y1[::9] * 1.08, color=colors[0], s=24, alpha=0.85)
    ax.axhline(y1[20], color="#6B7280", linestyle="--", linewidth=reference_line_width, label="reference")
    ax.grid(True, which="both", color=plot_style.grid_color, linewidth=0.8)


def _panel_title(stem: str, *, panel_index: int, total: int) -> str:
    short = stem.replace("_", " ")
    if len(short) > 42:
        short = short[:39] + "..."
    if total == 1:
        return short
    return f"{_panel_letter(panel_index)} {short}"


def _panel_letter(panel_index: int) -> str:
    return f"({chr(ord('a') + panel_index - 1)})"


def _compile_scratch_paper(
    *,
    scratch_paper_dir: Path,
    tex_name: str,
    latexmk_executable: str,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        latexmk_executable,
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_name,
    ]
    completed = subprocess.run(
        command,
        cwd=scratch_paper_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(
            "latexmk failed while compiling the scratch preview. "
            f"Log: {log_path}\n{tail}"
        )
    return {
        "requested": True,
        "success": True,
        "command": command,
        "log_path": str(log_path),
        "pdf_path": str(scratch_paper_dir / f"{Path(tex_name).stem}.pdf"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
