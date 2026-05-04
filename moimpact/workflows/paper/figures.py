#!/usr/bin/env python3
"""
Generate the figures referenced in ``paper/main.tex``.

This runner is paper-oriented rather than pipeline-oriented: it inspects the
LaTeX source, lets the user request all figures or only a subset, and then
drives the existing repository scripts with temporary YAML overrides so the
outputs land under the paper image tree by default.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from functools import partial
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise ImportError("Missing dependency: pyyaml is required to run the paper-figure generator.") from exc


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]

PAPER_TEX_DEFAULT = _REPO_ROOT / "paper" / "main.tex"
PAPER_IMAGES_DEFAULT = _REPO_ROOT / "paper" / "images"
PAPER_FIGURES_CFG_DEFAULT = _REPO_ROOT / "config_ymls" / "paper_figures.yml"

METAORDER_COMP_CFG = _REPO_ROOT / "config_ymls" / "metaorder_computation.yml"
METAORDER_DISTRIBUTIONS_CFG = _REPO_ROOT / "config_ymls" / "metaorder_distributions.yml"
METAORDER_SUMMARY_CFG = _REPO_ROOT / "config_ymls" / "metaorder_summary_statistics.yml"
METAORDER_EXECUTION_SCHEDULE_CFG = _REPO_ROOT / "config_ymls" / "metaorder_execution_schedule.yml"
METAORDER_START_EVENT_STUDY_CFG = _REPO_ROOT / "config_ymls" / "metaorder_start_event_study.yml"
PLOT_PROP_NONPROP_CFG = _REPO_ROOT / "config_ymls" / "plot_prop_nonprop_fits.yml"
CROWDING_CFG = _REPO_ROOT / "config_ymls" / "crowding_analysis.yml"
CROWDING_IMPACT_CFG = _REPO_ROOT / "config_ymls" / "crowding_impact_analysis.yml"
CROWDING_OVERLAP_CFG = _REPO_ROOT / "config_ymls" / "crowding_overlap_analysis.yml"
MEMBER_ACTIVE_OVERLAP_CROWDING_CFG = _REPO_ROOT / "config_ymls" / "member_active_overlap_crowding.yml"
PAPER_FIGURES_CONFIG_ENV = "PAPER_FIGURES_CONFIG"

INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
KNOWN_FIGURE_SUFFIXES = frozenset({".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps"})

CROWDING_ANALYSIS_BASENAMES = frozenset(
    {
        "daily_crowding_daily_corr.png",
        "daily_crowding_rolling_3d.png",
        "cross_crowding_daily_corr.png",
        "cross_crowding_rolling_3d.png",
        "all_vs_all_crowding_daily_corr.png",
        "all_vs_all_crowding_rolling_3d.png",
        "member_prop_client_crowding_hist.png",
        "member_prop_client_crowding_heatmap_3d.png",
    }
)
CROWDING_VS_ETA_BASENAMES = frozenset(
    {
        "curve_mean_align_vs_eta_local.png",
        "curve_mean_abs_imb_vs_eta_local.png",
        "curve_mean_align_vs_eta_cross.png",
        "curve_mean_abs_imb_vs_eta_cross.png",
    }
)
CROWDING_IMPACT_BASENAMES = frozenset(
    {
        "main_crowding_impact_curves.png",
        "eta_robustness_crowding_impact_curves.png",
    }
)
MEMBER_ACTIVE_OVERLAP_BASENAMES = frozenset(
    {
        "per_member_correlations_same_isin_all_active.png",
        "member_comovement_same_isin_all_active.png",
        "member_window_heatmap_same_isin_all_active.png",
    }
)
IMPACT_OVERLAY_BASENAMES = frozenset(
    {
        "power_law_prop_vs_nonprop.png",
        "logarithmic_prop_vs_nonprop.png",
    }
)
IMPACT_ALL_BASENAMES = frozenset(
    {
        "power_law_fits_by_participation_rate.png",
        "impact_surface_qv_participation_heatmap_member_non_proprietary.png",
        "impact_surface_qv_participation_heatmap_member_proprietary.png",
        "normalized_impact_path_member_proprietary.png",
        "normalized_impact_path_member_proprietary_by_side.png",
        "normalized_impact_path_member_non_proprietary.png",
        "normalized_impact_path_member_non_proprietary_by_side.png",
    }
)
IMPACT_IT_BASENAMES = frozenset(
    {
        "normalized_impact_path_member_proprietary_member_nationality_it.png",
        "normalized_impact_path_member_proprietary_by_side_member_nationality_it.png",
        "normalized_impact_path_member_non_proprietary_member_nationality_it.png",
        "normalized_impact_path_member_non_proprietary_by_side_member_nationality_it.png",
    }
)
_CROWDING_ANALYSIS_FIGURE_IDS = frozenset(Path(name).stem for name in CROWDING_ANALYSIS_BASENAMES)
_CROWDING_VS_ETA_FIGURE_IDS = frozenset(Path(name).stem for name in CROWDING_VS_ETA_BASENAMES)
_CROWDING_IMPACT_FIGURE_IDS = frozenset(Path(name).stem for name in CROWDING_IMPACT_BASENAMES)
_MEMBER_ACTIVE_OVERLAP_FIGURE_IDS = frozenset(Path(name).stem for name in MEMBER_ACTIVE_OVERLAP_BASENAMES)
_IMPACT_OVERLAY_FIGURE_IDS = frozenset(Path(name).stem for name in IMPACT_OVERLAY_BASENAMES)
_IMPACT_ALL_FIGURE_IDS = frozenset(Path(name).stem for name in IMPACT_ALL_BASENAMES)
_IMPACT_IT_FIGURE_IDS = frozenset(Path(name).stem for name in IMPACT_IT_BASENAMES)

TARGET_DESCRIPTIONS = {
    "member_stats": "Member-level descriptive plots from scripts/run_analysis.py members stats.",
    "summary": "Pooled member-profile figure plus mean daily metaorder-volume share.",
    "distributions": "Combined metaorder-distribution figure with tail-fit annotations.",
    "execution_schedule": "Proprietary-vs-client execution-schedule heatmap and overlay curves.",
    "impact": "Impact overlays plus per-group participation, surface, and path figures.",
    "crowding": "Within-group, cross-group, all-vs-all, eta-conditioned, overlap, and impact crowding figures.",
    "event_study": "Metaorder-start event-study curves for all-neighbor and same-actor-excluded variants.",
    "appendix_it": "Italian-member appendix normalized impact-path figures.",
    "all_main": "All figures from the main text, excluding the Italian-member appendix figure.",
    "all": "Every non-commented figure referenced in paper/main.tex.",
}


class MetaorderStage(IntEnum):
    """Minimal metaorder-computation stage needed for a figure subset."""

    NONE = 0
    DICTS = 1
    FILTERED = 2
    IMPACT = 3
    IMPACT_BY_SIDE = 4


@dataclass(frozen=True)
class SelectedWork:
    """Selected figures and the derived script-level requirements."""

    figures: tuple[str, ...]
    tasks: frozenset[str]
    stage_all: MetaorderStage
    stage_it: MetaorderStage


def _config_path_from_env(default_path: Path, env_var: str) -> Path:
    override = os.environ.get(env_var)
    if not override:
        return default_path
    candidate = Path(override).expanduser()
    if not candidate.is_absolute():
        candidate = (_REPO_ROOT / candidate).resolve()
    return candidate


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a YAML mapping: {path}")
    return data


def _write_yaml_mapping(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _default_dataset_name() -> str:
    cfg = _read_yaml_mapping(METAORDER_COMP_CFG)
    return str(cfg.get("DATASET_NAME") or "ftsemib")


_RUNNER_CONFIG_PATH = _config_path_from_env(PAPER_FIGURES_CFG_DEFAULT, PAPER_FIGURES_CONFIG_ENV)
_RUNNER_CFG = _read_yaml_mapping(_RUNNER_CONFIG_PATH) if _RUNNER_CONFIG_PATH.exists() else {}


def _resolve_runner_path(value: str | Path | None, default_path: Path) -> Path:
    raw = value if value is not None else default_path
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def _runner_default_targets() -> tuple[str, ...]:
    raw = _RUNNER_CFG.get("DEFAULT_TARGETS")
    if raw is None:
        return ("all",)
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
        return tuple(values) if values else ("all",)
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        values = [str(item).strip() for item in raw if str(item).strip()]
        return tuple(values) if values else ("all",)
    return ("all",)


def _style_updates_from_runner_cfg() -> dict[str, int]:
    """
    Summary
    -------
    Return legacy per-script style overrides requested by the paper runner.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, int]
        Always-empty mapping.

    Notes
    -----
    Plot typography is now controlled centrally by `config_ymls/plot_style.yml`.
    The paper runner no longer propagates font-size overrides into individual
    analysis YAML files.
    """
    return {}


def _runner_default_max_workers() -> int:
    raw = _RUNNER_CFG.get("MAX_WORKERS", 0)
    try:
        requested = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MAX_WORKERS value in {_RUNNER_CONFIG_PATH}: {raw!r}") from exc
    if requested < 0:
        raise ValueError(f"MAX_WORKERS must be >= 0 in {_RUNNER_CONFIG_PATH}, got {requested}.")
    return requested


def _runner_default_write_pdf() -> bool:
    raw = _RUNNER_CFG.get("WRITE_PDF", True)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid WRITE_PDF value in {_RUNNER_CONFIG_PATH}: {raw!r}")


def _resolve_max_workers(requested: int) -> int:
    if requested < 0:
        raise ValueError(f"max_workers must be >= 0, got {requested}.")
    if requested == 0:
        return max(1, min(os.cpu_count() or 1, 2))
    return requested


def _paper_figure_paths(paper_tex_path: Path) -> tuple[str, ...]:
    figures: list[str] = []
    for raw_line in paper_tex_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.lstrip()
        if stripped.startswith("%"):
            continue
        match = INCLUDEGRAPHICS_RE.search(raw_line)
        if match is not None:
            figures.append(match.group(1).strip())
    return tuple(dict.fromkeys(figures))


def _figure_path_id(path_like: str | Path) -> str:
    """Return a normalized figure path identifier without the image suffix."""

    path = Path(path_like)
    if path.suffix.lower() in KNOWN_FIGURE_SUFFIXES:
        path = path.with_suffix("")
    return path.as_posix()


def _figure_basename_id(path_like: str | Path) -> str:
    """Return a normalized figure basename without the image suffix."""

    path = Path(path_like)
    if path.suffix.lower() in KNOWN_FIGURE_SUFFIXES:
        return path.stem
    return path.name


def _classify_figure(figure_path: str) -> str:
    path = Path(figure_path)
    basename = _figure_basename_id(path)
    parent = path.parent.as_posix()

    if parent == "images/member_statistics/png":
        return "member_statistics"
    if basename == "member_metaorder_profiles_all_metaorders" and parent in {
        "images/member_metaorder_summary_statistics/png",
        "images/member_prop_vs_nonprop/png",
    }:
        return "metaorder_summary_pooled"
    if basename == "mean_daily_metaorder_volume_share" and parent in {
        "images/prop_vs_nonprop/png",
        "images/member_metaorder_summary_statistics/png",
        "images/member_prop_vs_nonprop/png",
    }:
        return "metaorder_summary_split"
    if parent == "images/member_metaorder_distributions/png":
        return "metaorder_distributions"
    if parent == "images/member_metaorder_execution_schedule/png" and basename.startswith(
        "execution_schedule_heatmap_prop_vs_client"
    ):
        return "execution_schedule"
    if basename in _IMPACT_OVERLAY_FIGURE_IDS and parent == "images/prop_vs_nonprop/png":
        return "impact_overlays"
    if basename in _IMPACT_IT_FIGURE_IDS and parent in {
        "images/member_proprietary/png",
        "images/member_non_proprietary/png",
    }:
        return "impact_it"
    if basename in _IMPACT_ALL_FIGURE_IDS and parent in {
        "images/member_proprietary/png",
        "images/member_non_proprietary/png",
    }:
        return "impact_main"
    if basename in _CROWDING_ANALYSIS_FIGURE_IDS and parent == "images/prop_vs_nonprop/png":
        return "crowding_analysis"
    if basename in _CROWDING_VS_ETA_FIGURE_IDS and parent == "images/crowding_vs_part_rate/png":
        return "crowding_vs_eta"
    if basename in _CROWDING_IMPACT_FIGURE_IDS and parent == "images/crowding_impact/png":
        return "crowding_impact"
    if basename in _MEMBER_ACTIVE_OVERLAP_FIGURE_IDS and parent == "images/member_active_overlap_crowding/png":
        return "member_active_overlap"
    if parent == "images/metaorder_start_event_study/png" and basename.startswith("event_curve_"):
        return "start_event_study"
    raise ValueError(f"Unmapped paper figure path: {figure_path}")


def _build_target_map(all_figures: Sequence[str]) -> dict[str, tuple[str, ...]]:
    by_task: dict[str, list[str]] = defaultdict(list)
    for figure in all_figures:
        by_task[_classify_figure(figure)].append(figure)

    all_main = tuple(fig for fig in all_figures if _classify_figure(fig) != "impact_it")
    return {
        "member_stats": tuple(by_task["member_statistics"]),
        "summary": tuple(by_task["metaorder_summary_pooled"] + by_task["metaorder_summary_split"]),
        "distributions": tuple(by_task["metaorder_distributions"]),
        "execution_schedule": tuple(by_task["execution_schedule"]),
        "impact": tuple(by_task["impact_overlays"] + by_task["impact_main"]),
        "crowding": tuple(
            by_task["crowding_analysis"]
            + by_task["crowding_vs_eta"]
            + by_task["crowding_impact"]
            + by_task["member_active_overlap"]
        ),
        "event_study": tuple(by_task["start_event_study"]),
        "appendix_it": tuple(by_task["impact_it"]),
        "all_main": all_main,
        "all": tuple(all_figures),
    }


def _resolve_requested_figure(
    requested: str,
    all_figures: Sequence[str],
) -> str:
    requested_clean = requested.strip()
    if requested_clean in all_figures:
        return requested_clean

    normalized_path_matches = [
        figure for figure in all_figures if _figure_path_id(figure) == _figure_path_id(requested_clean)
    ]
    if len(normalized_path_matches) == 1:
        return normalized_path_matches[0]
    if len(normalized_path_matches) > 1:
        raise ValueError(
            "Ambiguous figure path "
            f"{requested_clean!r}. Use the exact path from paper/main.tex."
        )

    basename_matches = [
        figure for figure in all_figures if _figure_basename_id(figure) == _figure_basename_id(requested_clean)
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        raise ValueError(
            "Ambiguous figure basename "
            f"{requested_clean!r}. Use the full path from paper/main.tex instead."
        )
    raise ValueError(f"Unknown paper figure request: {requested_clean}")


def _selected_work(
    all_figures: Sequence[str],
    target_map: Mapping[str, Sequence[str]],
    requested_targets: Sequence[str],
    requested_figures: Sequence[str],
) -> SelectedWork:
    selected: list[str] = []

    for target in requested_targets:
        if target not in target_map:
            valid = ", ".join(sorted(target_map))
            raise ValueError(f"Unknown target {target!r}. Valid targets: {valid}.")
        selected.extend(target_map[target])

    for requested_figure in requested_figures:
        selected.append(_resolve_requested_figure(requested_figure, all_figures))

    selected_unique = tuple(fig for fig in all_figures if fig in set(selected))
    tasks = frozenset(_classify_figure(fig) for fig in selected_unique)

    stage_all = MetaorderStage.NONE
    if {"metaorder_summary_pooled", "metaorder_summary_split", "metaorder_distributions"} & tasks:
        stage_all = max(stage_all, MetaorderStage.DICTS)
    if {
        "impact_overlays",
        "crowding_analysis",
        "crowding_vs_eta",
        "crowding_impact",
        "member_active_overlap",
        "execution_schedule",
        "start_event_study",
    } & tasks:
        stage_all = max(stage_all, MetaorderStage.FILTERED)

    impact_main_paths = {
        fig for fig in selected_unique if _classify_figure(fig) == "impact_main"
    }
    if impact_main_paths:
        if any(_figure_basename_id(fig).endswith("_by_side") for fig in impact_main_paths):
            stage_all = max(stage_all, MetaorderStage.IMPACT_BY_SIDE)
        else:
            stage_all = max(stage_all, MetaorderStage.IMPACT)

    impact_it_paths = {fig for fig in selected_unique if _classify_figure(fig) == "impact_it"}
    stage_it = MetaorderStage.NONE
    if impact_it_paths:
        if any(
            _figure_basename_id(fig).endswith("_by_side_member_nationality_it")
            for fig in impact_it_paths
        ):
            stage_it = MetaorderStage.IMPACT_BY_SIDE
        else:
            stage_it = MetaorderStage.IMPACT

    return SelectedWork(
        figures=selected_unique,
        tasks=tasks,
        stage_all=stage_all,
        stage_it=stage_it,
    )


@contextmanager
def _temporary_yaml_copy(base_config_path: Path, updates: Mapping[str, Any]) -> Iterator[Path]:
    base_cfg = _read_yaml_mapping(base_config_path)
    merged_cfg = dict(base_cfg)
    merged_cfg.update(dict(updates))
    with tempfile.TemporaryDirectory(prefix="paper_fig_cfg_") as tmpdir:
        tmp_path = Path(tmpdir) / base_config_path.name
        _write_yaml_mapping(tmp_path, merged_cfg)
        yield tmp_path


def _log_tail(log_path: Path, *, n_lines: int = 40) -> str:
    if not log_path.exists():
        return "<missing log file>"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-n_lines:]
    return "\n".join(tail)


def _run_logged_command(
    cmd: Sequence[str],
    *,
    log_path: Path,
    env_updates: Mapping[str, str] | None = None,
    dry_run: bool,
) -> None:
    env_updates = dict(env_updates or {})
    printable_env = " ".join(f"{key}={value}" for key, value in sorted(env_updates.items()))
    printable_cmd = " ".join(cmd)
    print(f"[run] {printable_env} {printable_cmd}".strip())
    print(f"[log] {log_path}")
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(env_updates)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(cmd),
            cwd=_REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        tail = _log_tail(log_path)
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {printable_cmd}\n"
            f"Log tail ({log_path}):\n{tail}"
        )


def _run_task_batch(
    task_specs: Sequence[tuple[str, Callable[[], None]]],
    *,
    max_workers: int,
) -> None:
    if not task_specs:
        return

    worker_count = min(max_workers, len(task_specs))
    if worker_count <= 1:
        for task_name, task in task_specs:
            print(f"[task] {task_name}")
            task()
        return

    print(f"[parallel] Running {len(task_specs)} task(s) with max_workers={worker_count}")
    first_failure: tuple[str, Exception] | None = None
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_name = {executor.submit(task): task_name for task_name, task in task_specs}
        for future in as_completed(future_to_name):
            task_name = future_to_name[future]
            if future.cancelled():
                print(f"[cancelled] {task_name}")
                continue
            try:
                future.result()
                print(f"[done] {task_name}")
            except Exception as exc:
                print(f"[failed] {task_name}: {exc}")
                if first_failure is None:
                    first_failure = (task_name, exc)
                for pending in future_to_name:
                    if pending is not future:
                        pending.cancel()

    if first_failure is not None:
        task_name, exc = first_failure
        raise RuntimeError(f"Paper-figure task failed: {task_name}") from exc


def _ensure_parent(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path, *, dry_run: bool) -> None:
    if not src.exists():
        return
    print(f"[copy] {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _summary_compatibility_copy(img_output_root: Path, *, dry_run: bool) -> None:
    src_png = img_output_root / "member_metaorder_summary_statistics" / "png" / "mean_daily_metaorder_volume_share.png"
    dst_png = img_output_root / "prop_vs_nonprop" / "png" / "mean_daily_metaorder_volume_share.png"
    _copy_if_exists(src_png, dst_png, dry_run=dry_run)

    src_html = (
        img_output_root
        / "member_metaorder_summary_statistics"
        / "html"
        / "mean_daily_metaorder_volume_share.html"
    )
    dst_html = img_output_root / "prop_vs_nonprop" / "html" / "mean_daily_metaorder_volume_share.html"
    _copy_if_exists(src_html, dst_html, dry_run=dry_run)


def _run_metaorder_intro(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH": str(img_output_root),
        "PROPRIETARY": False,
        "MEMBER_NATIONALITY": None,
        "RUN_INTRO": True,
        "RUN_METAORDER_COMPUTATION": False,
        "RUN_SQL_FITS": False,
        "RUN_WLS": False,
        "RUN_IMPACT_PATH_PLOT": False,
        "RUN_SIGNATURE_PLOTS": False,
        "SPLIT_BY_SIDE": False,
    }
    updates.update(style_updates)
    with _temporary_yaml_copy(METAORDER_COMP_CFG, updates) as cfg_path:
        _run_logged_command(
            [sys.executable, "scripts/run_analysis.py", "metaorders", "compute"],
            log_path=log_dir / "metaorder_intro.log",
            env_updates={"METAORDER_COMP_CONFIG": str(cfg_path)},
            dry_run=dry_run,
        )


def _run_metaorder_stage(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    member_nationality: str | None,
    stage: MetaorderStage,
    max_workers: int,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    if stage == MetaorderStage.NONE:
        return

    nationality_tag = member_nationality or "all"

    def _run_group(proprietary: bool) -> None:
        proprietary_tag = "proprietary" if proprietary else "client"
        run_sql_fits = stage >= MetaorderStage.FILTERED
        run_wls = stage >= MetaorderStage.IMPACT

        primary_updates = {
            "DATASET_NAME": dataset_name,
            "IMG_OUTPUT_PATH": str(img_output_root),
            "PROPRIETARY": proprietary,
            "MEMBER_NATIONALITY": member_nationality,
            "RUN_INTRO": False,
            "RUN_METAORDER_COMPUTATION": True,
            "RUN_SQL_FITS": run_sql_fits,
            "RUN_WLS": run_wls,
            "RUN_IMPACT_PATH_PLOT": run_wls,
            "RUN_SIGNATURE_PLOTS": False,
            "SPLIT_BY_SIDE": False,
        }
        primary_updates.update(style_updates)
        with _temporary_yaml_copy(METAORDER_COMP_CFG, primary_updates) as cfg_path:
            _run_logged_command(
                [sys.executable, "scripts/run_analysis.py", "metaorders", "compute"],
                log_path=log_dir / f"metaorder_{nationality_tag}_{proprietary_tag}_primary.log",
                env_updates={"METAORDER_COMP_CONFIG": str(cfg_path)},
                dry_run=dry_run,
            )

        if stage >= MetaorderStage.IMPACT_BY_SIDE:
            by_side_updates = dict(primary_updates)
            by_side_updates.update(
                {
                    "RUN_METAORDER_COMPUTATION": False,
                    "RUN_SQL_FITS": False,
                    "RUN_WLS": True,
                    "RUN_IMPACT_PATH_PLOT": True,
                    "SPLIT_BY_SIDE": True,
                }
            )
            with _temporary_yaml_copy(METAORDER_COMP_CFG, by_side_updates) as cfg_path:
                _run_logged_command(
                    [sys.executable, "scripts/run_analysis.py", "metaorders", "compute"],
                    log_path=log_dir / f"metaorder_{nationality_tag}_{proprietary_tag}_by_side.log",
                    env_updates={"METAORDER_COMP_CONFIG": str(cfg_path)},
                    dry_run=dry_run,
                )

    _run_task_batch(
        [
            (f"metaorder_stage_{nationality_tag}_proprietary", partial(_run_group, True)),
            (f"metaorder_stage_{nationality_tag}_client", partial(_run_group, False)),
        ],
        max_workers=min(max_workers, 2),
    )


def _run_metaorder_distributions(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH": str(img_output_root),
        "MEMBER_NATIONALITY": None,
    }
    updates.update(style_updates)
    with _temporary_yaml_copy(METAORDER_DISTRIBUTIONS_CFG, updates) as cfg_path:
        _run_logged_command(
            [sys.executable, "scripts/run_analysis.py", "metaorders", "distributions"],
            log_path=log_dir / "metaorder_distributions.log",
            env_updates={"METAORDER_DISTRIBUTIONS_CONFIG": str(cfg_path)},
            dry_run=dry_run,
        )


def _run_metaorder_summary(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    condition_on_client_proprietary: bool,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    mode_tag = "split" if condition_on_client_proprietary else "pooled"
    updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH": str(img_output_root),
        "MEMBER_NATIONALITY": None,
    }
    updates.update(style_updates)
    with _temporary_yaml_copy(METAORDER_SUMMARY_CFG, updates) as cfg_path:
        _run_logged_command(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "metaorders",
                "summary",
                "--condition-on-client-proprietary",
                "true" if condition_on_client_proprietary else "false",
            ],
            log_path=log_dir / f"metaorder_summary_{mode_tag}.log",
            env_updates={"METAORDER_SUMMARY_STATS_CONFIG": str(cfg_path)},
            dry_run=dry_run,
        )


def _run_member_statistics(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    env_updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH_OVERRIDE": str(img_output_root),
    }
    _run_logged_command(
        [sys.executable, "scripts/run_analysis.py", "members", "stats"],
        log_path=log_dir / "member_statistics.log",
        env_updates=env_updates,
        dry_run=dry_run,
    )


def _run_prop_vs_nonprop_overlays(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH": str(img_output_root),
    }
    updates.update(style_updates)
    with _temporary_yaml_copy(PLOT_PROP_NONPROP_CFG, updates) as cfg_path:
        _run_logged_command(
            [sys.executable, "scripts/run_analysis.py", "impact", "overlay"],
            log_path=log_dir / "plot_prop_nonprop_fits.log",
            env_updates={"PLOT_PROP_NONPROP_FITS_CONFIG": str(cfg_path)},
            dry_run=dry_run,
        )


def _run_crowding_analysis(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH": str(img_output_root),
    }
    updates.update(style_updates)
    with _temporary_yaml_copy(CROWDING_CFG, updates) as cfg_path:
        _run_logged_command(
            [sys.executable, "scripts/run_analysis.py", "crowding", "daily"],
            log_path=log_dir / "crowding_analysis.log",
            env_updates={"CROWDING_CONFIG": str(cfg_path)},
            dry_run=dry_run,
        )


def _run_crowding_vs_eta(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH": str(img_output_root),
    }
    updates.update(style_updates)
    with _temporary_yaml_copy(CROWDING_CFG, updates) as cfg_path:
        _run_logged_command(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "crowding",
                "eta",
                "--dataset-name",
                dataset_name,
                "--config-path",
                str(cfg_path),
                "--img-output-path",
                str(img_output_root),
            ],
            log_path=log_dir / "crowding_vs_part_rate.log",
            dry_run=dry_run,
        )


def _run_crowding_impact(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH": str(img_output_root),
    }
    updates.update(style_updates)
    with _temporary_yaml_copy(CROWDING_IMPACT_CFG, updates) as cfg_path:
        _run_logged_command(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "crowding",
                "impact",
                "--config-path",
                str(cfg_path),
            ],
            log_path=log_dir / "crowding_impact.log",
            dry_run=dry_run,
        )


def _run_crowding_overlap(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH": str(img_output_root),
        "PLOTS": False,
        "WRITE_PARQUET": True,
    }
    updates.update(style_updates)
    with _temporary_yaml_copy(CROWDING_OVERLAP_CFG, updates) as cfg_path:
        _run_logged_command(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "crowding",
                "overlap",
                "--config-path",
                str(cfg_path),
            ],
            log_path=log_dir / "crowding_overlap.log",
            dry_run=dry_run,
        )


def _run_member_active_overlap(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH": str(img_output_root),
    }
    updates.update(style_updates)
    with _temporary_yaml_copy(MEMBER_ACTIVE_OVERLAP_CROWDING_CFG, updates) as cfg_path:
        _run_logged_command(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "crowding",
                "member-overlap",
                "--config-path",
                str(cfg_path),
            ],
            log_path=log_dir / "member_active_overlap_crowding.log",
            dry_run=dry_run,
        )


def _run_metaorder_execution_schedule(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH": str(img_output_root),
        "LEVEL": "member",
        "MEMBER_NATIONALITY": None,
    }
    updates.update(style_updates)
    with _temporary_yaml_copy(METAORDER_EXECUTION_SCHEDULE_CFG, updates) as cfg_path:
        _run_logged_command(
            [sys.executable, "scripts/run_analysis.py", "execution", "schedule"],
            log_path=log_dir / "metaorder_execution_schedule.log",
            env_updates={"METAORDER_EXECUTION_SCHEDULE_CONFIG": str(cfg_path)},
            dry_run=dry_run,
        )


def _run_metaorder_start_event_study(
    *,
    dataset_name: str,
    img_output_root: Path,
    log_dir: Path,
    style_updates: Mapping[str, int],
    dry_run: bool,
) -> None:
    updates = {
        "DATASET_NAME": dataset_name,
        "IMG_OUTPUT_PATH": str(img_output_root),
    }
    updates.update(style_updates)
    with _temporary_yaml_copy(METAORDER_START_EVENT_STUDY_CFG, updates) as cfg_path:
        _run_logged_command(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "metaorders",
                "start-event",
                "--config-path",
                str(cfg_path),
            ],
            log_path=log_dir / "metaorder_start_event_study.log",
            dry_run=dry_run,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the figures referenced in paper/main.tex. By default this writes "
            "outputs under paper/images and selects every non-commented figure."
        )
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=(),
        help="One or more logical target groups. Use --list-targets to inspect them.",
    )
    parser.add_argument(
        "--figures",
        nargs="*",
        default=(),
        help=(
            "Optional exact paper figure paths or unique basenames from paper/main.tex. "
            "These are unioned with --targets."
        ),
    )
    parser.add_argument(
        "--paper-tex",
        default=str(_resolve_runner_path(_RUNNER_CFG.get("PAPER_TEX_PATH", PAPER_TEX_DEFAULT), PAPER_TEX_DEFAULT)),
        help="Path to the LaTeX source to inspect. Default: paper/main.tex.",
    )
    parser.add_argument(
        "--img-output-root",
        default=str(
            _resolve_runner_path(_RUNNER_CFG.get("IMG_OUTPUT_ROOT", PAPER_IMAGES_DEFAULT), PAPER_IMAGES_DEFAULT)
        ),
        help="Root directory where generated paper images should be written. Default: paper/images.",
    )
    parser.add_argument(
        "--dataset-name",
        default=str(_RUNNER_CFG.get("DATASET_NAME") or _default_dataset_name()),
        help="Dataset name propagated into the existing YAML configs. Default: DATASET_NAME from metaorder_computation.yml.",
    )
    parser.add_argument(
        "--list-figures",
        action="store_true",
        help="List the non-commented figure paths found in paper/main.tex and exit.",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List the available target groups and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=_runner_default_max_workers(),
        help=(
            "Maximum number of independent external figure-generation tasks to run in parallel. "
            "Use 1 for fully serial execution or 0 for conservative auto-selection."
        ),
    )
    parser.add_argument(
        "--write-pdf",
        action=argparse.BooleanOptionalAction,
        default=_runner_default_write_pdf(),
        help="Whether to export PDF sidecars for paper figures. Default: value from paper_figures.yml.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Summary
    -------
    Generate all or a selected subset of the figures referenced in ``paper/main.tex``.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI argument vector. When ``None``, ``sys.argv`` is used.

    Returns
    -------
    int
        Process exit code. Returns ``0`` on success.

    Notes
    -----
    - The runner does not edit repository YAML files in place. It writes
      temporary YAML copies and points the existing scripts at them via their
      documented environment variables.
    - The default image root is ``paper/images`` because those are the paths
      consumed by ``paper/main.tex``.
    - The legacy paper path
      ``images/prop_vs_nonprop/png/mean_daily_metaorder_volume_share.png`` is
      maintained via a compatibility copy after the summary-statistics step.

    Examples
    --------
    Generate every figure referenced in the paper:

    >>> # python scripts/run_analysis.py paper figures --targets all

    Generate only the impact figures:

    >>> # python scripts/run_analysis.py paper figures --targets impact

    Generate two explicit figures:

    >>> # python scripts/run_analysis.py paper figures --figures \\
    >>> #   images/prop_vs_nonprop/png/power_law_prop_vs_nonprop.png \\
    >>> #   images/member_non_proprietary/png/normalized_impact_path_member_non_proprietary.png
    """
    args = _parse_args(argv)

    paper_tex_path = Path(args.paper_tex).resolve()
    img_output_root = Path(args.img_output_root).resolve()
    dataset_name = str(args.dataset_name)
    style_updates = _style_updates_from_runner_cfg()
    max_workers = _resolve_max_workers(int(args.max_workers))
    write_pdf = bool(args.write_pdf)

    all_figures = _paper_figure_paths(paper_tex_path)

    if args.list_figures:
        for figure in all_figures:
            print(figure)
        return 0

    target_map = _build_target_map(all_figures)

    if args.list_targets:
        for target_name in (
            "member_stats",
            "summary",
            "distributions",
            "execution_schedule",
            "impact",
            "crowding",
            "event_study",
            "appendix_it",
            "all_main",
            "all",
        ):
            figure_count = len(target_map[target_name])
            description = TARGET_DESCRIPTIONS[target_name]
            print(f"{target_name}: {figure_count} figures")
            print(f"  {description}")
        return 0

    requested_targets: Sequence[str]
    if args.targets:
        requested_targets = tuple(args.targets)
    elif args.figures:
        requested_targets = ()
    else:
        requested_targets = _runner_default_targets()

    work = _selected_work(
        all_figures=all_figures,
        target_map=target_map,
        requested_targets=requested_targets,
        requested_figures=args.figures,
    )
    if not work.figures:
        raise ValueError("The requested targets/figures resolved to an empty figure set.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = _REPO_ROOT / "out_files" / dataset_name / "logs" / f"paper_figures_{timestamp}"
    _ensure_parent(log_dir, dry_run=args.dry_run)
    _ensure_parent(img_output_root, dry_run=args.dry_run)

    manifest = {
        "generated_at": timestamp,
        "paper_tex": str(paper_tex_path),
        "paper_figures_config": str(_RUNNER_CONFIG_PATH),
        "dataset_name": dataset_name,
        "img_output_root": str(img_output_root),
        "targets": list(requested_targets),
        "explicit_figures": list(args.figures),
        "selected_figures": list(work.figures),
        "tasks": sorted(work.tasks),
        "style_updates": style_updates,
        "max_workers": max_workers,
        "write_pdf": write_pdf,
        "stage_all": work.stage_all.name,
        "stage_it": work.stage_it.name,
        "python_executable": sys.executable,
        "dry_run": bool(args.dry_run),
    }
    manifest_path = log_dir / "run_manifest.json"
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[manifest] {manifest_path}")

    print("[selection] Figures to generate:")
    for figure in work.figures:
        print(f"  - {figure}")

    print(
        "[stages] "
        f"stage_all={work.stage_all.name}, "
        f"stage_it={work.stage_it.name}, "
        f"tasks={', '.join(sorted(work.tasks))}, "
        f"max_workers={max_workers}, "
        f"write_pdf={write_pdf}"
    )

    original_plotly_write_pdf = os.environ.get("PLOTLY_WRITE_PDF")
    if write_pdf:
        os.environ["PLOTLY_WRITE_PDF"] = "true"
    else:
        os.environ.pop("PLOTLY_WRITE_PDF", None)

    try:
        _run_metaorder_intro(
            dataset_name=dataset_name,
            img_output_root=img_output_root,
            log_dir=log_dir,
            style_updates=style_updates,
            dry_run=args.dry_run,
        )

        _run_metaorder_stage(
            dataset_name=dataset_name,
            img_output_root=img_output_root,
            log_dir=log_dir,
            member_nationality=None,
            stage=work.stage_all,
            max_workers=max_workers,
            style_updates=style_updates,
            dry_run=args.dry_run,
        )

        downstream_tasks: list[tuple[str, Callable[[], None]]] = []
        if "member_statistics" in work.tasks:
            downstream_tasks.append(
                (
                    "member_statistics",
                    partial(
                        _run_member_statistics,
                        dataset_name=dataset_name,
                        img_output_root=img_output_root,
                        log_dir=log_dir,
                        style_updates=style_updates,
                        dry_run=args.dry_run,
                    ),
                )
            )

        if "metaorder_distributions" in work.tasks:
            downstream_tasks.append(
                (
                    "metaorder_distributions",
                    partial(
                        _run_metaorder_distributions,
                        dataset_name=dataset_name,
                        img_output_root=img_output_root,
                        log_dir=log_dir,
                        style_updates=style_updates,
                        dry_run=args.dry_run,
                    ),
                )
            )

        if "metaorder_summary_pooled" in work.tasks:
            downstream_tasks.append(
                (
                    "metaorder_summary_pooled",
                    partial(
                        _run_metaorder_summary,
                        dataset_name=dataset_name,
                        img_output_root=img_output_root,
                        log_dir=log_dir,
                        condition_on_client_proprietary=False,
                        style_updates=style_updates,
                        dry_run=args.dry_run,
                    ),
                )
            )

        if "metaorder_summary_split" in work.tasks:
            def _run_split_summary_and_copy() -> None:
                _run_metaorder_summary(
                    dataset_name=dataset_name,
                    img_output_root=img_output_root,
                    log_dir=log_dir,
                    condition_on_client_proprietary=True,
                    style_updates=style_updates,
                    dry_run=args.dry_run,
                )
                _summary_compatibility_copy(img_output_root, dry_run=args.dry_run)

            downstream_tasks.append(("metaorder_summary_split", _run_split_summary_and_copy))

        if "execution_schedule" in work.tasks:
            downstream_tasks.append(
                (
                    "metaorder_execution_schedule",
                    partial(
                        _run_metaorder_execution_schedule,
                        dataset_name=dataset_name,
                        img_output_root=img_output_root,
                        log_dir=log_dir,
                        style_updates=style_updates,
                        dry_run=args.dry_run,
                    ),
                )
            )

        if "impact_overlays" in work.tasks:
            downstream_tasks.append(
                (
                    "plot_prop_nonprop_fits",
                    partial(
                        _run_prop_vs_nonprop_overlays,
                        dataset_name=dataset_name,
                        img_output_root=img_output_root,
                        log_dir=log_dir,
                        style_updates=style_updates,
                        dry_run=args.dry_run,
                    ),
                )
            )

        if "crowding_analysis" in work.tasks:
            downstream_tasks.append(
                (
                    "crowding_analysis",
                    partial(
                        _run_crowding_analysis,
                        dataset_name=dataset_name,
                        img_output_root=img_output_root,
                        log_dir=log_dir,
                        style_updates=style_updates,
                        dry_run=args.dry_run,
                    ),
                )
            )

        if "crowding_vs_eta" in work.tasks:
            downstream_tasks.append(
                (
                    "crowding_vs_part_rate",
                    partial(
                        _run_crowding_vs_eta,
                        dataset_name=dataset_name,
                        img_output_root=img_output_root,
                        log_dir=log_dir,
                        style_updates=style_updates,
                        dry_run=args.dry_run,
                    ),
                )
            )

        if "crowding_impact" in work.tasks:
            downstream_tasks.append(
                (
                    "crowding_impact",
                    partial(
                        _run_crowding_impact,
                        dataset_name=dataset_name,
                        img_output_root=img_output_root,
                        log_dir=log_dir,
                        style_updates=style_updates,
                        dry_run=args.dry_run,
                    ),
                )
            )

        if "start_event_study" in work.tasks:
            downstream_tasks.append(
                (
                    "metaorder_start_event_study",
                    partial(
                        _run_metaorder_start_event_study,
                        dataset_name=dataset_name,
                        img_output_root=img_output_root,
                        log_dir=log_dir,
                        style_updates=style_updates,
                        dry_run=args.dry_run,
                    ),
                )
            )

        _run_task_batch(downstream_tasks, max_workers=max_workers)

        if "member_active_overlap" in work.tasks:
            _run_crowding_overlap(
                dataset_name=dataset_name,
                img_output_root=img_output_root,
                log_dir=log_dir,
                style_updates=style_updates,
                dry_run=args.dry_run,
            )
            _run_member_active_overlap(
                dataset_name=dataset_name,
                img_output_root=img_output_root,
                log_dir=log_dir,
                style_updates=style_updates,
                dry_run=args.dry_run,
            )

        _run_metaorder_stage(
            dataset_name=dataset_name,
            img_output_root=img_output_root,
            log_dir=log_dir,
            member_nationality="it" if work.stage_it != MetaorderStage.NONE else None,
            stage=work.stage_it,
            max_workers=max_workers,
            style_updates=style_updates,
            dry_run=args.dry_run,
        )
    finally:
        if original_plotly_write_pdf is None:
            os.environ.pop("PLOTLY_WRITE_PDF", None)
        else:
            os.environ["PLOTLY_WRITE_PDF"] = original_plotly_write_pdf

    print("[done] Paper-figure generation plan completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
