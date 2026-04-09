#!/usr/bin/env python3
"""
Pooled execution typology for proprietary and client metaorders.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.config import format_path_template, load_yaml_mapping, resolve_repo_path
from moimpact.execution_typology import (
    PathFeatureSpec,
    aggregate_schedule_profiles_chunk,
    apply_type_label_overrides,
    assign_auto_type_labels,
    engineer_path_features_frame,
    order_clusters_for_display,
    period_duration_seconds,
)
from moimpact.logging_utils import PrintTee, setup_file_logger
from moimpact.plot_style import (
    THEME_BG_COLOR,
    THEME_COLORWAY,
    THEME_FONT_FAMILY,
    THEME_GRID_COLOR,
    apply_plotly_style,
)
from moimpact.plotting import (
    COLOR_CLIENT,
    COLOR_NEUTRAL,
    COLOR_PROPRIETARY,
    PlotOutputDirs,
    make_plot_output_dirs,
    save_plotly_figure as _save_plotly_figure,
)


_FEATURE_HEATMAP_STEM = "execution_typology_feature_heatmap"
_GROUP_SHARES_STEM = "execution_typology_group_shares"
_PCA_SCATTER_STEM = "execution_typology_pca_scatter"
_IMPACT_STEM = "execution_typology_impact_profiles"
_SCHEDULE_STEM = "execution_typology_schedule_profiles"

_ROW_FILE = "clustered_metaorders.parquet"
_SUMMARY_FILE = "cluster_summary.csv"
_SHARES_FILE = "group_type_shares.csv"
_IMPACT_FILE = "impact_profiles_by_type.csv"
_SCHEDULE_FILE = "schedule_profiles_by_type.csv"
_SILHOUETTE_FILE = "silhouette_scores.csv"
_PCA_REPORT_FILE = "pca_report.json"
_MANIFEST_FILE = "run_manifest.json"

_SUMMARY_METRIC_COLS = [
    "Q",
    "Q/V",
    "Participation Rate",
    "Vt/V",
    "N Child",
    "Daily Vol",
    "DurationSeconds",
    "schedule_front25_share",
    "schedule_front50_share",
    "schedule_back25_share",
    "schedule_center_of_mass",
    "schedule_hhi",
    "schedule_twap_l1_distance",
    "abs_impact_end",
    "abs_impact_1m",
    "abs_impact_10m",
    "abs_impact_30m",
    "abs_impact_60m",
    "retention_30_over_end",
    "retention_60_over_end",
    "peak_abs_partial_impact",
    "overshoot_peak_over_end",
]
_CLUSTER_FEATURE_COLS = [
    "log_Q",
    "log_Q_over_V",
    "log_participation_rate",
    "log_vt_over_v",
    "log_duration_seconds",
    "log_n_child",
    "asinh_daily_vol",
    "schedule_front25_share",
    "schedule_front50_share",
    "schedule_back25_share",
    "schedule_center_of_mass",
    "schedule_hhi",
    "schedule_twap_l1_distance",
    "abs_impact_end",
    "abs_impact_1m",
    "abs_impact_10m",
    "abs_impact_30m",
    "abs_impact_60m",
    "log1p_retention_30_over_end",
    "log1p_retention_60_over_end",
    "peak_abs_partial_impact",
    "log1p_overshoot_peak_over_end",
]
_PCA_SCATTER_COLORWAY = [
    "#355070",
    "#6D597A",
    "#B56576",
    "#E56B6F",
    "#EAAC8B",
    "#457B9D",
    "#1D3557",
    "#2A9D8F",
]


class _NullTqdm:
    def __enter__(self) -> "_NullTqdm":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def _make_tqdm(*, total: Optional[int], desc: str, disable: bool, unit: str):
    if disable or tqdm is None:
        return _NullTqdm()
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def save_plotly_figure(fig, *args, **kwargs):
    fig.update_layout(title=None)
    return _save_plotly_figure(fig, *args, **kwargs)


def _load_yaml_defaults(config_path: Path) -> dict[str, Any]:
    return load_yaml_mapping(config_path)


def _try_git_hash() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _resolve_level(args: argparse.Namespace, cfg: Mapping[str, Any]) -> str:
    value = str(args.level or cfg.get("LEVEL", "member")).strip().lower()
    if value not in {"member", "client"}:
        raise ValueError("LEVEL must be one of: member, client.")
    return value


def _default_input_paths(output_root: Path, level: str) -> tuple[Path, Path]:
    return (
        output_root / f"metaorders_info_sameday_filtered_{level}_proprietary.parquet",
        output_root / f"metaorders_info_sameday_filtered_{level}_non_proprietary.parquet",
    )


def _resolve_paths(cfg: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Path | str]:
    dataset_name = str(args.dataset_name or cfg.get("DATASET_NAME", "ftsemib"))
    level = _resolve_level(args, cfg)
    context = {"DATASET_NAME": dataset_name, "LEVEL": level}
    output_root = resolve_repo_path(
        _REPO_ROOT,
        format_path_template(str(args.output_file_path or cfg.get("OUTPUT_FILE_PATH", "out_files/{DATASET_NAME}")), context),
    )
    image_root = resolve_repo_path(
        _REPO_ROOT,
        format_path_template(str(args.img_output_path or cfg.get("IMG_OUTPUT_PATH", "images/{DATASET_NAME}")), context),
    )
    analysis_tag = str(args.analysis_tag or cfg.get("ANALYSIS_TAG", "execution_typology"))
    default_prop, default_client = _default_input_paths(output_root, level)
    prop_path = resolve_repo_path(_REPO_ROOT, args.prop_path or cfg.get("PROP_PATH") or default_prop)
    client_path = resolve_repo_path(_REPO_ROOT, args.client_path or cfg.get("CLIENT_PATH") or default_client)
    out_dir = resolve_repo_path(_REPO_ROOT, output_root / analysis_tag)
    img_dir = resolve_repo_path(_REPO_ROOT, image_root / analysis_tag)
    log_dir = output_root / "logs"
    log_path = log_dir / f"{analysis_tag}.log"
    return {
        "dataset_name": dataset_name,
        "level": level,
        "prop_path": prop_path,
        "client_path": client_path,
        "out_dir": out_dir,
        "img_dir": img_dir,
        "config_path": resolve_repo_path(_REPO_ROOT, args.config_path),
        "log_path": log_path,
    }


def _resolve_worker_count(requested_jobs: int, n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    requested = int(requested_jobs)
    if requested < 0:
        raise ValueError("N_JOBS must be >= 0.")
    if requested == 0:
        return max(1, min(n_tasks, min(os.cpu_count() or 1, 4)))
    return max(1, min(n_tasks, requested))


def _process_pool_context():
    if os.name != "posix":
        return None
    try:
        return mp.get_context("fork")
    except ValueError:  # pragma: no cover
        return None


def _safe_log_series(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=False)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr >= 0.0)
    if np.any(mask):
        positive = arr[mask & (arr > 0.0)]
        eps = max(1.0e-12, 0.5 * float(np.min(positive))) if positive.size else 1.0e-12
        out[mask] = np.log(arr[mask] + eps)
    return pd.Series(out, index=values.index)


def _safe_asinh_series(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=False)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    out[mask] = np.arcsinh(arr[mask])
    return pd.Series(out, index=values.index)


def _safe_log1p_series(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=False)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr >= 0.0)
    if np.any(mask):
        out[mask] = np.log1p(arr[mask])
    return pd.Series(out, index=values.index)


def _chunk_index_ranges(n_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    if n_rows <= 0:
        return []
    size = max(1, int(chunk_size))
    return [(start, min(start + size, n_rows)) for start in range(0, n_rows, size)]


def _extract_path_features_parallel(
    df: pd.DataFrame,
    *,
    spec: PathFeatureSpec,
    chunk_size: int,
    n_jobs: int,
    show_progress: bool,
) -> pd.DataFrame:
    source_cols = [
        "child_time_norm",
        "child_volume_fraction",
        "partial_impact",
        "Impact",
        "Impact_1m",
        "Impact_10m",
        "Impact_30m",
        "Impact_60m",
    ]
    ranges = _chunk_index_ranges(len(df), chunk_size)
    worker_count = _resolve_worker_count(n_jobs, len(ranges))
    if worker_count <= 1:
        pieces = []
        with _make_tqdm(total=len(ranges), desc="[Typology] path features", disable=not show_progress, unit="chunk") as pbar:
            for start, end in ranges:
                chunk = df.iloc[start:end][source_cols].copy()
                pieces.append(engineer_path_features_frame(chunk, spec=spec))
                pbar.update(1)
        return pd.concat(pieces, axis=0).sort_index()

    ctx = _process_pool_context()
    executor_kwargs: dict[str, Any] = {"max_workers": worker_count}
    if ctx is not None:
        executor_kwargs["mp_context"] = ctx
    pieces: list[pd.DataFrame] = []
    with _make_tqdm(total=len(ranges), desc="[Typology] path features", disable=not show_progress, unit="chunk") as pbar:
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            future_to_idx = {
                executor.submit(
                    engineer_path_features_frame,
                    df.iloc[start:end][source_cols].copy(),
                    spec=spec,
                ): idx
                for idx, (start, end) in enumerate(ranges)
            }
            ordered: list[Optional[pd.DataFrame]] = [None] * len(ranges)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                ordered[idx] = future.result()
                pbar.update(1)
        pieces = [piece for piece in ordered if piece is not None]
    return pd.concat(pieces, axis=0).sort_index()


def _engineer_features(
    df: pd.DataFrame,
    *,
    chunk_size: int,
    n_jobs: int,
    twap_grid_size: int,
    show_progress: bool,
) -> pd.DataFrame:
    out = df.copy()
    out["DurationSeconds"] = out["Period"].apply(period_duration_seconds)
    out["log_Q"] = _safe_log_series(out["Q"])
    out["log_Q_over_V"] = _safe_log_series(out["Q/V"])
    out["log_participation_rate"] = _safe_log_series(out["Participation Rate"])
    out["log_vt_over_v"] = _safe_log_series(out["Vt/V"])
    out["log_duration_seconds"] = _safe_log_series(out["DurationSeconds"])
    out["log_n_child"] = _safe_log_series(out["N Child"])
    out["asinh_daily_vol"] = _safe_asinh_series(out["Daily Vol"])
    path_features = _extract_path_features_parallel(
        out,
        spec=PathFeatureSpec(twap_grid_size=twap_grid_size),
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )
    out = out.join(path_features)
    out["log1p_retention_30_over_end"] = _safe_log1p_series(out["retention_30_over_end"])
    out["log1p_retention_60_over_end"] = _safe_log1p_series(out["retention_60_over_end"])
    out["log1p_overshoot_peak_over_end"] = _safe_log1p_series(out["overshoot_peak_over_end"])
    return out


def _prepare_clustering_matrix(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = df
    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    matrix = work[list(feature_cols)].to_numpy(dtype=float, copy=False)
    valid_mask = np.all(np.isfinite(matrix), axis=1)
    work["clustering_valid"] = valid_mask
    filtered = work.loc[valid_mask].reset_index(drop=False).rename(columns={"index": "_row_idx"})
    if len(filtered) < 3:
        raise ValueError("Not enough valid rows remain after feature engineering for clustering.")
    matrix_valid = filtered[list(feature_cols)].to_numpy(dtype=float, copy=False)
    return work, filtered, matrix_valid


def _fit_pca(
    matrix: np.ndarray,
    *,
    variance_threshold: float,
) -> tuple[np.ndarray, PCA]:
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("PCA_VARIANCE_THRESHOLD must be in (0, 1].")
    full = PCA()
    full.fit(matrix)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    n_components = max(2, min(n_components, matrix.shape[1]))
    model = PCA(n_components=n_components)
    scores = model.fit_transform(matrix)
    return scores, model


def _sample_indices_by_group(groups: pd.Series, *, max_size: int, seed: int) -> np.ndarray:
    if len(groups) <= max_size:
        return np.arange(len(groups), dtype=int)
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    total = len(groups)
    group_counts = groups.value_counts(sort=False)
    for group_name, count in group_counts.items():
        target = max(1, int(round(max_size * float(count) / float(total))))
        idx = np.flatnonzero(groups.to_numpy(dtype=object) == group_name)
        take = min(len(idx), target)
        parts.append(np.sort(rng.choice(idx, size=take, replace=False)))
    sample = np.unique(np.concatenate(parts))
    if sample.size > max_size:
        sample = np.sort(rng.choice(sample, size=max_size, replace=False))
    elif sample.size < max_size:
        remaining = np.setdiff1d(np.arange(len(groups), dtype=int), sample, assume_unique=False)
        if remaining.size > 0:
            extra = np.sort(rng.choice(remaining, size=min(max_size - sample.size, remaining.size), replace=False))
            sample = np.sort(np.concatenate([sample, extra]))
    return sample


def _evaluate_k_values(
    matrix: np.ndarray,
    *,
    k_values: Sequence[int],
    seed: int,
    minibatch_batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for k in k_values:
        model = MiniBatchKMeans(
            n_clusters=int(k),
            n_init=10,
            random_state=int(seed),
            batch_size=int(minibatch_batch_size),
        )
        labels = model.fit_predict(matrix)
        sil = float(silhouette_score(matrix, labels))
        rows.append(
            {
                "k": int(k),
                "silhouette": sil,
                "inertia": float(model.inertia_),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def _choose_k(curve: pd.DataFrame) -> int:
    if curve.empty:
        raise ValueError("Silhouette curve is empty.")
    best_row = curve.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(best_row["k"])


def _summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    group = df.groupby("Cluster", sort=True, dropna=False)
    summary = group.size().rename("n_metaorders").to_frame().reset_index()
    counts = (
        df.pivot_table(index="Cluster", columns="Group", values="ISIN", aggfunc="size", fill_value=0)
        .rename(columns={"client": "n_client", "proprietary": "n_proprietary"})
        .reset_index()
    )
    summary = summary.merge(counts, on="Cluster", how="left")
    for feature in _SUMMARY_METRIC_COLS:
        agg = group[feature].agg(
            median=lambda s: float(np.nanmedian(s.to_numpy(dtype=float))) if np.any(np.isfinite(s.to_numpy(dtype=float))) else float("nan"),
            q25=lambda s: float(np.nanquantile(s.to_numpy(dtype=float), 0.25)) if np.any(np.isfinite(s.to_numpy(dtype=float))) else float("nan"),
            q75=lambda s: float(np.nanquantile(s.to_numpy(dtype=float), 0.75)) if np.any(np.isfinite(s.to_numpy(dtype=float))) else float("nan"),
        ).reset_index()
        agg = agg.rename(
            columns={
                "median": f"{feature}_median",
                "q25": f"{feature}_q25",
                "q75": f"{feature}_q75",
            }
        )
        summary = summary.merge(agg, on="Cluster", how="left")
    summary = assign_auto_type_labels(summary)
    return order_clusters_for_display(summary)


def _build_group_type_shares(df: pd.DataFrame) -> pd.DataFrame:
    by_group = (
        df.groupby(["Group", "Cluster", "type_code", "auto_type_label", "type_label", "display_order"], dropna=False)
        .size()
        .rename("n_metaorders")
        .reset_index()
    )
    by_group["share_basis"] = "within_group"
    by_group["share"] = by_group["n_metaorders"] / by_group.groupby("Group")["n_metaorders"].transform("sum")

    by_type = by_group.copy()
    by_type["share_basis"] = "within_type"
    by_type["share"] = by_type["n_metaorders"] / by_type.groupby("Cluster")["n_metaorders"].transform("sum")
    return pd.concat([by_group, by_type], ignore_index=True)


def _build_impact_profiles(df: pd.DataFrame) -> pd.DataFrame:
    horizon_map = {
        "abs_impact_end": "end",
        "abs_impact_1m": "1m",
        "abs_impact_10m": "10m",
        "abs_impact_30m": "30m",
        "abs_impact_60m": "60m",
    }
    rows: list[dict[str, Any]] = []
    for (cluster, type_label, display_order, group_name), sub in df.groupby(
        ["Cluster", "type_label", "display_order", "Group"],
        sort=False,
        dropna=False,
    ):
        for col, horizon in horizon_map.items():
            values = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "Cluster": int(cluster),
                    "type_label": str(type_label),
                    "display_order": int(display_order),
                    "Group": str(group_name),
                    "horizon": horizon,
                    "median": float(np.nanmedian(finite)) if finite.size else float("nan"),
                    "q25": float(np.nanquantile(finite, 0.25)) if finite.size else float("nan"),
                    "q75": float(np.nanquantile(finite, 0.75)) if finite.size else float("nan"),
                    "n_metaorders": int(finite.size),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_schedule_profiles(
    df: pd.DataFrame,
    *,
    tau_grid: np.ndarray,
    chunk_size: int,
    n_jobs: int,
    show_progress: bool,
) -> pd.DataFrame:
    source_cols = ["Group", "Cluster", "child_time_norm", "child_volume_fraction"]
    ranges = _chunk_index_ranges(len(df), chunk_size)
    worker_count = _resolve_worker_count(n_jobs, len(ranges))
    partial_results: list[dict[tuple[str, int], dict[str, Any]]] = []
    if worker_count <= 1:
        with _make_tqdm(total=len(ranges), desc="[Typology] schedule profiles", disable=not show_progress, unit="chunk") as pbar:
            for start, end in ranges:
                chunk = df.iloc[start:end][source_cols].copy()
                partial_results.append(aggregate_schedule_profiles_chunk(chunk, tau_grid=tau_grid))
                pbar.update(1)
    else:
        ctx = _process_pool_context()
        executor_kwargs: dict[str, Any] = {"max_workers": worker_count}
        if ctx is not None:
            executor_kwargs["mp_context"] = ctx
        with _make_tqdm(total=len(ranges), desc="[Typology] schedule profiles", disable=not show_progress, unit="chunk") as pbar:
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                future_to_idx = {
                    executor.submit(
                        aggregate_schedule_profiles_chunk,
                        df.iloc[start:end][source_cols].copy(),
                        tau_grid=tau_grid,
                    ): idx
                    for idx, (start, end) in enumerate(ranges)
                }
                ordered: list[Optional[dict[tuple[str, int], dict[str, Any]]]] = [None] * len(ranges)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    ordered[idx] = future.result()
                    pbar.update(1)
            partial_results = [item for item in ordered if item is not None]

    combined: dict[tuple[str, int], dict[str, Any]] = {}
    for part in partial_results:
        for key, state in part.items():
            slot = combined.setdefault(
                key,
                {
                    "sum_curve": np.zeros_like(tau_grid, dtype=float),
                    "sumsq_curve": np.zeros_like(tau_grid, dtype=float),
                    "count_curve": np.zeros_like(tau_grid, dtype=float),
                    "n_valid_metaorders": 0,
                },
            )
            slot["sum_curve"] += state["sum_curve"]
            slot["sumsq_curve"] += state["sumsq_curve"]
            slot["count_curve"] += state["count_curve"]
            slot["n_valid_metaorders"] += int(state["n_valid_metaorders"])

    meta = (
        df[["Group", "Cluster", "type_label", "display_order"]]
        .drop_duplicates(subset=["Group", "Cluster"])
        .copy()
    )
    rows: list[dict[str, Any]] = []
    for _, meta_row in meta.iterrows():
        key = (str(meta_row["Group"]), int(meta_row["Cluster"]))
        state = combined.get(key)
        if state is None or int(state["n_valid_metaorders"]) <= 0:
            continue
        count_curve = np.asarray(state["count_curve"], dtype=float)
        mean_curve = np.divide(
            state["sum_curve"],
            count_curve,
            out=np.full(count_curve.shape, np.nan, dtype=float),
            where=count_curve > 0,
        )
        variance = np.full(count_curve.shape, np.nan, dtype=float)
        valid_var = count_curve > 1.0
        variance[valid_var] = (
            state["sumsq_curve"][valid_var]
            - np.square(state["sum_curve"][valid_var]) / count_curve[valid_var]
        ) / (count_curve[valid_var] - 1.0)
        variance[valid_var] = np.maximum(variance[valid_var], 0.0)
        sem_curve = np.full(count_curve.shape, np.nan, dtype=float)
        sem_curve[valid_var] = np.sqrt(variance[valid_var] / count_curve[valid_var])
        for tau, mean_value, sem_value, n_eff in zip(tau_grid, mean_curve, sem_curve, count_curve, strict=True):
            rows.append(
                {
                    "Group": str(meta_row["Group"]),
                    "Cluster": int(meta_row["Cluster"]),
                    "type_label": str(meta_row["type_label"]),
                    "display_order": int(meta_row["display_order"]),
                    "tau": float(tau),
                    "mean_cum_volume_fraction": float(mean_value),
                    "sem_cum_volume_fraction": float(sem_value) if np.isfinite(sem_value) else float("nan"),
                    "n_eff": int(n_eff),
                    "n_valid_metaorders": int(state["n_valid_metaorders"]),
                }
            )
    return pd.DataFrame(rows)


def _short_feature_label(name: str) -> str:
    mapping = {
        "Participation Rate_median": "eta",
        "Q/V_median": "Q/V",
        "DurationSeconds_median": "duration",
        "N Child_median": "child count",
        "schedule_front25_share_median": "front 25%",
        "schedule_front50_share_median": "front 50%",
        "schedule_back25_share_median": "back 25%",
        "schedule_center_of_mass_median": "center of mass",
        "schedule_hhi_median": "schedule HHI",
        "schedule_twap_l1_distance_median": "TWAP distance",
        "abs_impact_end_median": "|I_end|",
        "abs_impact_30m_median": "|I_30m|",
        "abs_impact_60m_median": "|I_60m|",
        "retention_60_over_end_median": "retention 60/end",
        "peak_abs_partial_impact_median": "peak partial",
        "overshoot_peak_over_end_median": "peak/end",
    }
    return mapping.get(name, name)


def _plot_feature_heatmap(summary_df: pd.DataFrame, *, dirs: PlotOutputDirs) -> None:
    plot_cols = [
        "Participation Rate_median",
        "Q/V_median",
        "DurationSeconds_median",
        "N Child_median",
        "schedule_front25_share_median",
        "schedule_back25_share_median",
        "schedule_center_of_mass_median",
        "schedule_twap_l1_distance_median",
        "abs_impact_end_median",
        "abs_impact_30m_median",
        "abs_impact_60m_median",
        "retention_60_over_end_median",
        "peak_abs_partial_impact_median",
        "overshoot_peak_over_end_median",
    ]
    frame = summary_df.sort_values("display_order").copy()
    values = frame[plot_cols].to_numpy(dtype=float, copy=True)
    col_means = np.nanmean(values, axis=0)
    col_stds = np.nanstd(values, axis=0)
    col_stds[col_stds <= 1.0e-12] = 1.0
    standardized = (values - col_means[None, :]) / col_stds[None, :]
    y_labels = [f"{row.display_order + 1}. {row.type_label}" for row in frame.itertuples(index=False)]
    x_labels = [_short_feature_label(col) for col in plot_cols]
    fig = go.Figure(
        go.Heatmap(
            z=standardized,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu",
            zmid=0.0,
            colorbar=dict(title="Cluster z-score"),
            hovertemplate="%{y}<br>%{x}<br>z=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=max(500, 120 + 90 * len(y_labels)), width=1400)
    fig.update_xaxes(title_text="Feature median", tickangle=-35)
    fig.update_yaxes(title_text="Execution type")
    save_plotly_figure(fig, stem=_FEATURE_HEATMAP_STEM, dirs=dirs, write_html=True, write_png=True)


def _plot_group_shares(shares_df: pd.DataFrame, *, dirs: PlotOutputDirs) -> None:
    within_group = shares_df.loc[shares_df["share_basis"].eq("within_group")].copy()
    within_type = shares_df.loc[shares_df["share_basis"].eq("within_type")].copy()
    order_df = (
        shares_df[["Cluster", "type_label", "display_order"]]
        .drop_duplicates()
        .sort_values("display_order")
    )
    type_labels = order_df["type_label"].tolist()
    color_map = {
        label: _PCA_SCATTER_COLORWAY[idx % len(_PCA_SCATTER_COLORWAY)]
        for idx, label in enumerate(type_labels)
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Type mix within each group", "Proprietary vs client composition within type"),
        horizontal_spacing=0.12,
    )
    for label in type_labels:
        sub = within_group.loc[within_group["type_label"].eq(label)]
        fig.add_trace(
            go.Bar(
                x=sub["Group"],
                y=sub["share"],
                name=label,
                marker=dict(color=color_map[label]),
                hovertemplate="%{x}<br>share=%{y:.3f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    composition = (
        within_type.pivot_table(index="type_label", columns="Group", values="share", aggfunc="first")
        .reindex(type_labels)
        .reset_index()
    )
    for group_name, color in [("proprietary", COLOR_PROPRIETARY), ("client", COLOR_CLIENT)]:
        if group_name not in composition.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=composition["type_label"],
                y=composition[group_name],
                name=group_name.capitalize(),
                marker=dict(color=color),
                hovertemplate="%{x}<br>share=%{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_layout(barmode="stack", height=650, width=1500, legend=dict(orientation="h", x=0.0, y=1.15))
    fig.update_yaxes(title_text="Share", row=1, col=1)
    fig.update_yaxes(title_text="Share", row=1, col=2)
    fig.update_xaxes(title_text="Group", row=1, col=1)
    fig.update_xaxes(title_text="Execution type", tickangle=-25, row=1, col=2)
    save_plotly_figure(fig, stem=_GROUP_SHARES_STEM, dirs=dirs, write_html=True, write_png=True)


def _plot_pca_scatter(
    scatter_df: pd.DataFrame,
    *,
    explained_variance_ratio: Sequence[float],
    dirs: PlotOutputDirs,
) -> None:
    fig = go.Figure()
    type_order = (
        scatter_df[["display_order", "type_label"]]
        .drop_duplicates()
        .sort_values("display_order")
        ["type_label"]
        .tolist()
    )
    color_map = {
        label: _PCA_SCATTER_COLORWAY[idx % len(_PCA_SCATTER_COLORWAY)]
        for idx, label in enumerate(type_order)
    }
    symbol_map = {"proprietary": "circle", "client": "diamond"}
    for type_label in type_order:
        for group_name in ["proprietary", "client"]:
            sub = scatter_df.loc[
                scatter_df["type_label"].eq(type_label) & scatter_df["Group"].eq(group_name)
            ]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["PC1"],
                    y=sub["PC2"],
                    mode="markers",
                    name=f"{type_label} | {group_name}",
                    marker=dict(
                        color=color_map[type_label],
                        symbol=symbol_map[group_name],
                        size=5,
                        opacity=0.35,
                    ),
                    hovertemplate=(
                        f"type={type_label}<br>"
                        f"group={group_name}<br>"
                        "PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>"
                    ),
                )
            )
    pc1_var = 100.0 * float(explained_variance_ratio[0]) if len(explained_variance_ratio) >= 1 else float("nan")
    pc2_var = 100.0 * float(explained_variance_ratio[1]) if len(explained_variance_ratio) >= 2 else float("nan")
    fig.update_layout(height=800, width=1300, legend=dict(orientation="h", x=0.0, y=1.14))
    fig.update_xaxes(title_text=f"PC1 ({pc1_var:.1f}% var)")
    fig.update_yaxes(title_text=f"PC2 ({pc2_var:.1f}% var)")
    save_plotly_figure(fig, stem=_PCA_SCATTER_STEM, dirs=dirs, write_html=True, write_png=True)


def _subplot_grid(n_panels: int) -> tuple[int, int]:
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(math.ceil(float(n_panels) / float(n_cols)))
    return n_rows, n_cols


def _plot_impact_profiles(impact_df: pd.DataFrame, *, dirs: PlotOutputDirs) -> None:
    order_df = (
        impact_df[["display_order", "type_label"]]
        .drop_duplicates()
        .sort_values("display_order")
        .reset_index(drop=True)
    )
    horizons = ["end", "1m", "10m", "30m", "60m"]
    n_rows, n_cols = _subplot_grid(len(order_df))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=order_df["type_label"].tolist(), shared_yaxes=False)
    for idx, row in order_df.iterrows():
        r = int(idx // n_cols) + 1
        c = int(idx % n_cols) + 1
        sub = impact_df.loc[impact_df["type_label"].eq(row["type_label"])]
        for group_name, color in [("proprietary", COLOR_PROPRIETARY), ("client", COLOR_CLIENT)]:
            grp = (
                sub.loc[sub["Group"].eq(group_name)]
                .set_index("horizon")
                .reindex(horizons)
                .reset_index()
            )
            fig.add_trace(
                go.Scatter(
                    x=horizons,
                    y=grp["median"],
                    mode="lines+markers",
                    line=dict(color=color, width=3),
                    name=group_name.capitalize(),
                    legendgroup=group_name,
                    showlegend=(idx == 0),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=np.maximum(grp["q75"].to_numpy(dtype=float) - grp["median"].to_numpy(dtype=float), 0.0),
                        arrayminus=np.maximum(grp["median"].to_numpy(dtype=float) - grp["q25"].to_numpy(dtype=float), 0.0),
                    ),
                    hovertemplate="%{x}<br>median=%{y:.4f}<extra></extra>",
                ),
                row=r,
                col=c,
            )
        fig.update_xaxes(title_text="Impact horizon", row=r, col=c)
    fig.update_yaxes(title_text="Median absolute impact", row=1, col=1)
    fig.update_layout(height=max(550, 340 * n_rows), width=1450, legend=dict(orientation="h", x=0.0, y=1.08))
    save_plotly_figure(fig, stem=_IMPACT_STEM, dirs=dirs, write_html=True, write_png=True)


def _add_sem_band(
    fig: go.Figure,
    *,
    x: np.ndarray,
    mean_curve: np.ndarray,
    sem_curve: np.ndarray,
    color: str,
    name: str,
    row: int,
    col: int,
    showlegend: bool,
) -> None:
    band_color = "rgba(53,80,112,0.16)" if color == COLOR_PROPRIETARY else "rgba(229,107,111,0.16)"
    upper = mean_curve + np.nan_to_num(sem_curve, nan=0.0)
    lower = mean_curve - np.nan_to_num(sem_curve, nan=0.0)
    fig.add_trace(
        go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=band_color,
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean_curve,
            mode="lines",
            line=dict(color=color, width=3),
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            hovertemplate="tau=%{x:.2f}<br>cum. vol=%{y:.3f}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def _plot_schedule_profiles(schedule_df: pd.DataFrame, *, dirs: PlotOutputDirs) -> None:
    order_df = (
        schedule_df[["display_order", "type_label"]]
        .drop_duplicates()
        .sort_values("display_order")
        .reset_index(drop=True)
    )
    n_rows, n_cols = _subplot_grid(len(order_df))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=order_df["type_label"].tolist(), shared_yaxes=True)
    for idx, row in order_df.iterrows():
        r = int(idx // n_cols) + 1
        c = int(idx % n_cols) + 1
        sub = schedule_df.loc[schedule_df["type_label"].eq(row["type_label"])]
        for group_name, color in [("proprietary", COLOR_PROPRIETARY), ("client", COLOR_CLIENT)]:
            grp = sub.loc[sub["Group"].eq(group_name)].sort_values("tau")
            if grp.empty:
                continue
            _add_sem_band(
                fig,
                x=grp["tau"].to_numpy(dtype=float),
                mean_curve=grp["mean_cum_volume_fraction"].to_numpy(dtype=float),
                sem_curve=grp["sem_cum_volume_fraction"].to_numpy(dtype=float),
                color=color,
                name=group_name.capitalize(),
                row=r,
                col=c,
                showlegend=(idx == 0),
            )
        tau_values = sub["tau"].drop_duplicates().sort_values().to_numpy(dtype=float)
        if tau_values.size:
            fig.add_trace(
                go.Scatter(
                    x=tau_values,
                    y=tau_values,
                    mode="lines",
                    line=dict(color=COLOR_NEUTRAL, width=2, dash="dash"),
                    name="TWAP",
                    showlegend=(idx == 0),
                    hovertemplate="tau=%{x:.2f}<br>TWAP=%{y:.2f}<extra></extra>",
                ),
                row=r,
                col=c,
            )
        fig.update_xaxes(title_text="Normalized execution time", range=[0.0, 1.0], row=r, col=c)
    fig.update_yaxes(title_text="Cumulative volume fraction", range=[0.0, 1.0], row=1, col=1)
    fig.update_layout(height=max(550, 340 * n_rows), width=1450, legend=dict(orientation="h", x=0.0, y=1.10))
    save_plotly_figure(fig, stem=_SCHEDULE_STEM, dirs=dirs, write_html=True, write_png=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pooled proprietary-vs-client execution typology.")
    parser.add_argument("--config-path", type=str, default="config_ymls/metaorder_execution_typology.yml")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--level", type=str, default=None)
    parser.add_argument("--prop-path", type=str, default=None)
    parser.add_argument("--client-path", type=str, default=None)
    parser.add_argument("--output-file-path", type=str, default=None)
    parser.add_argument("--img-output-path", type=str, default=None)
    parser.add_argument("--analysis-tag", type=str, default=None)
    parser.add_argument("--k-min", type=int, default=None)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--silhouette-sample-size", type=int, default=None)
    parser.add_argument("--pca-scatter-sample-size", type=int, default=None)
    parser.add_argument("--pca-variance-threshold", type=float, default=None)
    parser.add_argument("--minibatch-batch-size", type=int, default=None)
    parser.add_argument("--twap-grid-size", type=int, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write-parquet", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = _load_yaml_defaults(resolve_repo_path(_REPO_ROOT, args.config_path))

    tick_font_size = int(cfg.get("TICK_FONT_SIZE", 12))
    label_font_size = int(cfg.get("LABEL_FONT_SIZE", 14))
    title_font_size = int(cfg.get("TITLE_FONT_SIZE", 15))
    legend_font_size = int(cfg.get("LEGEND_FONT_SIZE", 12))
    apply_plotly_style(
        tick_font_size=tick_font_size,
        label_font_size=label_font_size,
        title_font_size=title_font_size,
        legend_font_size=legend_font_size,
        theme_colorway=THEME_COLORWAY,
        theme_grid_color=THEME_GRID_COLOR,
        theme_bg_color=THEME_BG_COLOR,
        theme_font_family=THEME_FONT_FAMILY,
    )

    paths = _resolve_paths(cfg, args)
    k_min = int(args.k_min if args.k_min is not None else cfg.get("K_MIN", 2))
    k_max = int(args.k_max if args.k_max is not None else cfg.get("K_MAX", 8))
    seed = int(args.seed if args.seed is not None else cfg.get("SEED", 0))
    n_jobs = int(args.n_jobs if args.n_jobs is not None else cfg.get("N_JOBS", 0))
    chunk_size = int(args.chunk_size if args.chunk_size is not None else cfg.get("CHUNK_SIZE", 50000))
    silhouette_sample_size = int(
        args.silhouette_sample_size if args.silhouette_sample_size is not None else cfg.get("SILHOUETTE_SAMPLE_SIZE", 50000)
    )
    pca_scatter_sample_size = int(
        args.pca_scatter_sample_size if args.pca_scatter_sample_size is not None else cfg.get("PCA_SCATTER_SAMPLE_SIZE", 40000)
    )
    pca_variance_threshold = float(
        args.pca_variance_threshold if args.pca_variance_threshold is not None else cfg.get("PCA_VARIANCE_THRESHOLD", 0.9)
    )
    minibatch_batch_size = int(
        args.minibatch_batch_size if args.minibatch_batch_size is not None else cfg.get("MINIBATCH_BATCH_SIZE", 4096)
    )
    twap_grid_size = int(args.twap_grid_size if args.twap_grid_size is not None else cfg.get("TWAP_GRID_SIZE", 21))
    plots_enabled = bool(args.plots if args.plots is not None else cfg.get("PLOTS", True))
    write_parquet = bool(args.write_parquet if args.write_parquet is not None else cfg.get("WRITE_PARQUET", True))
    show_progress = bool(args.progress if args.progress is not None else cfg.get("SHOW_PROGRESS", True))
    type_label_overrides = cfg.get("TYPE_LABEL_OVERRIDES") or {}

    manifest = {
        "run_timestamp": dt.datetime.utcnow().isoformat(),
        "git_hash": _try_git_hash(),
        "dataset_name": paths["dataset_name"],
        "level": paths["level"],
        "config_path": str(paths["config_path"]),
        "prop_path": str(paths["prop_path"]),
        "client_path": str(paths["client_path"]),
        "out_dir": str(paths["out_dir"]),
        "img_dir": str(paths["img_dir"]),
        "k_min": k_min,
        "k_max": k_max,
        "seed": seed,
        "n_jobs": n_jobs,
        "chunk_size": chunk_size,
        "silhouette_sample_size": silhouette_sample_size,
        "pca_scatter_sample_size": pca_scatter_sample_size,
        "pca_variance_threshold": pca_variance_threshold,
        "minibatch_batch_size": minibatch_batch_size,
        "twap_grid_size": twap_grid_size,
        "plots_enabled": plots_enabled,
        "write_parquet": write_parquet,
        "show_progress": show_progress,
        "type_label_overrides": {str(k): str(v) for k, v in dict(type_label_overrides).items()},
        "cluster_feature_columns": list(_CLUSTER_FEATURE_COLS),
        "summary_feature_columns": list(_SUMMARY_METRIC_COLS),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    out_dir = Path(paths["out_dir"])
    img_dirs = make_plot_output_dirs(Path(paths["img_dir"]), use_subdirs=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(paths["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    if plots_enabled:
        img_dirs.base_dir.mkdir(parents=True, exist_ok=True)
        img_dirs.html_dir.mkdir(parents=True, exist_ok=True)
        img_dirs.png_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_file_logger(Path(__file__).stem, Path(paths["log_path"]), mode="w", also_stdout=True, reset_handlers=True)
    with PrintTee(logger):
        print("[Typology] Execution typology run started.")
        print(f"[Typology] Proprietary path: {paths['prop_path']}")
        print(f"[Typology] Client path: {paths['client_path']}")

        prop_df = pd.read_parquet(Path(paths["prop_path"])).copy()
        client_df = pd.read_parquet(Path(paths["client_path"])).copy()
        prop_df["Group"] = "proprietary"
        client_df["Group"] = "client"
        pooled_df = pd.concat([prop_df, client_df], axis=0, ignore_index=True)
        print(f"[Typology] Loaded pooled sample: {len(pooled_df)} rows.")

        required_cols = {
            "ISIN",
            "Direction",
            "Q",
            "Q/V",
            "Participation Rate",
            "Vt/V",
            "N Child",
            "Period",
            "Daily Vol",
            "Impact",
            "Impact_1m",
            "Impact_10m",
            "Impact_30m",
            "Impact_60m",
            "child_time_norm",
            "child_volume_fraction",
            "partial_impact",
        }
        missing = sorted(required_cols.difference(pooled_df.columns))
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        engineered_df = _engineer_features(
            pooled_df,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            twap_grid_size=twap_grid_size,
            show_progress=show_progress,
        )
        print("[Typology] Engineered execution, schedule, and impact-shape features.")

        full_df, clustered_valid_df, matrix_valid = _prepare_clustering_matrix(
            engineered_df,
            feature_cols=_CLUSTER_FEATURE_COLS,
        )
        print(f"[Typology] Valid rows for clustering: {len(clustered_valid_df)}/{len(full_df)}.")

        scaler = RobustScaler()
        matrix_scaled = scaler.fit_transform(matrix_valid)
        matrix_pca, pca_model = _fit_pca(matrix_scaled, variance_threshold=pca_variance_threshold)
        clustered_valid_df["PC1"] = matrix_pca[:, 0]
        clustered_valid_df["PC2"] = matrix_pca[:, 1]

        if k_min < 2 or k_max < k_min:
            raise ValueError("Require 2 <= K_MIN <= K_MAX.")
        max_k = min(k_max, len(clustered_valid_df) - 1)
        if max_k < k_min:
            raise ValueError("Not enough valid rows for the requested k range.")
        k_values = list(range(k_min, max_k + 1))
        sample_idx = _sample_indices_by_group(
            clustered_valid_df["Group"],
            max_size=max(1000, silhouette_sample_size),
            seed=seed,
        )
        silhouette_curve = _evaluate_k_values(
            matrix_pca[sample_idx],
            k_values=k_values,
            seed=seed,
            minibatch_batch_size=minibatch_batch_size,
        )
        best_k = _choose_k(silhouette_curve)
        print(f"[Typology] Selected k={best_k} from silhouette sweep.")

        cluster_model = MiniBatchKMeans(
            n_clusters=best_k,
            n_init=10,
            random_state=seed,
            batch_size=minibatch_batch_size,
        )
        cluster_labels = cluster_model.fit_predict(matrix_pca)
        clustered_valid_df["Cluster"] = cluster_labels.astype(int)

        summary_df = _summarize_clusters(clustered_valid_df)
        summary_df = apply_type_label_overrides(summary_df, type_label_overrides)
        label_cols = ["Cluster", "type_code", "auto_type_label", "type_label", "display_order", "urgency_score", "front_loading_score", "persistence_score"]
        clustered_valid_df = clustered_valid_df.merge(summary_df[label_cols], on="Cluster", how="left")

        full_df["Cluster"] = pd.Series(pd.NA, index=full_df.index, dtype="Int64")
        for col in ["type_code", "auto_type_label", "type_label"]:
            full_df[col] = ""
        full_df["display_order"] = pd.Series(pd.NA, index=full_df.index, dtype="Int64")
        full_df.loc[clustered_valid_df["_row_idx"], "Cluster"] = clustered_valid_df["Cluster"].to_numpy(dtype=int)
        full_df.loc[clustered_valid_df["_row_idx"], "type_code"] = clustered_valid_df["type_code"].to_numpy(dtype=object)
        full_df.loc[clustered_valid_df["_row_idx"], "auto_type_label"] = clustered_valid_df["auto_type_label"].to_numpy(dtype=object)
        full_df.loc[clustered_valid_df["_row_idx"], "type_label"] = clustered_valid_df["type_label"].to_numpy(dtype=object)
        full_df.loc[clustered_valid_df["_row_idx"], "display_order"] = clustered_valid_df["display_order"].to_numpy(dtype=int)

        group_type_shares = _build_group_type_shares(clustered_valid_df)
        impact_profiles = _build_impact_profiles(clustered_valid_df)
        tau_grid = np.linspace(0.0, 1.0, twap_grid_size)
        schedule_profiles = _aggregate_schedule_profiles(
            clustered_valid_df,
            tau_grid=tau_grid,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )

        full_df.to_parquet(out_dir / _ROW_FILE, index=False)
        summary_df.to_csv(out_dir / _SUMMARY_FILE, index=False)
        group_type_shares.to_csv(out_dir / _SHARES_FILE, index=False)
        impact_profiles.to_csv(out_dir / _IMPACT_FILE, index=False)
        schedule_profiles.to_csv(out_dir / _SCHEDULE_FILE, index=False)
        silhouette_curve.to_csv(out_dir / _SILHOUETTE_FILE, index=False)
        if write_parquet:
            group_type_shares.to_parquet(out_dir / _SHARES_FILE.replace(".csv", ".parquet"), index=False)
            impact_profiles.to_parquet(out_dir / _IMPACT_FILE.replace(".csv", ".parquet"), index=False)
            schedule_profiles.to_parquet(out_dir / _SCHEDULE_FILE.replace(".csv", ".parquet"), index=False)
            summary_df.to_parquet(out_dir / _SUMMARY_FILE.replace(".csv", ".parquet"), index=False)

        pca_report = {
            "best_k": int(best_k),
            "k_values": k_values,
            "cluster_feature_columns": list(_CLUSTER_FEATURE_COLS),
            "rows_loaded": int(len(full_df)),
            "rows_valid_for_clustering": int(len(clustered_valid_df)),
            "pca_n_components": int(pca_model.n_components_),
            "pca_explained_variance_ratio": [float(x) for x in pca_model.explained_variance_ratio_],
            "pca_variance_threshold": float(pca_variance_threshold),
            "silhouette_sample_size_effective": int(len(sample_idx)),
        }
        _write_json(out_dir / _PCA_REPORT_FILE, pca_report)
        manifest.update(
            {
                "best_k": int(best_k),
                "rows_loaded": int(len(full_df)),
                "rows_valid_for_clustering": int(len(clustered_valid_df)),
                "pca_n_components": int(pca_model.n_components_),
                "pca_explained_variance_ratio": [float(x) for x in pca_model.explained_variance_ratio_],
            }
        )
        _write_json(out_dir / _MANIFEST_FILE, manifest)

        if plots_enabled:
            _plot_feature_heatmap(summary_df, dirs=img_dirs)
            _plot_group_shares(group_type_shares, dirs=img_dirs)
            scatter_idx = _sample_indices_by_group(
                clustered_valid_df["Group"],
                max_size=max(1000, pca_scatter_sample_size),
                seed=seed + 7,
            )
            _plot_pca_scatter(
                clustered_valid_df.iloc[scatter_idx].copy(),
                explained_variance_ratio=pca_model.explained_variance_ratio_,
                dirs=img_dirs,
            )
            _plot_impact_profiles(impact_profiles, dirs=img_dirs)
            _plot_schedule_profiles(schedule_profiles, dirs=img_dirs)

        print(f"[Typology] Wrote row-level clusters to {out_dir / _ROW_FILE}")
        print(f"[Typology] Wrote cluster summary to {out_dir / _SUMMARY_FILE}")
        print(f"[Typology] Wrote figures under {img_dirs.base_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
