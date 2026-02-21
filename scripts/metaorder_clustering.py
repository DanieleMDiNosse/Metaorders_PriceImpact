#!/usr/bin/env python3
"""
Cluster metaorders with PCA + k-means and silhouette-based k selection.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

# Ensure repository-root imports (e.g., `moimpact`) work when running
# `python scripts/metaorder_clustering.py` from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.config import format_path_template
from moimpact.plot_style import (
    PLOTLY_TEMPLATE_NAME,
    THEME_BG_COLOR,
    THEME_COLORWAY,
    THEME_FONT_FAMILY,
    THEME_GRID_COLOR,
    apply_plotly_style,
)
from moimpact.plotting import make_plot_output_dirs, save_plotly_figure as _save_plotly_figure


_IMPACT_HORIZON_PATTERN = re.compile(r"^Impact_(\d+)m$")
_EPS_FLOOR = 1.0e-12
_CLUSTERED_FILE = "clustered_metaorders.parquet"
_SILHOUETTE_FILE = "silhouette_scores.csv"
_SUMMARY_FILE = "cluster_summary.csv"
_PCA_REPORT_FILE = "pca_report.json"
_PCA_SCORES_FILE = "pca_scores.parquet"
_PCA_FEATURE_SELECTION_SUMMARY_FILE = "pca_feature_selection_summary.csv"
_PCA_FEATURE_SELECTION_PC_FILE = "pca_feature_selection_per_pc.csv"
_TICK_FONT_SIZE = 12
_LABEL_FONT_SIZE = 14
_TITLE_FONT_SIZE = 15
_LEGEND_FONT_SIZE = 12

apply_plotly_style(
    tick_font_size=_TICK_FONT_SIZE,
    label_font_size=_LABEL_FONT_SIZE,
    title_font_size=_TITLE_FONT_SIZE,
    legend_font_size=_LEGEND_FONT_SIZE,
    theme_colorway=THEME_COLORWAY,
    theme_grid_color=THEME_GRID_COLOR,
    theme_bg_color=THEME_BG_COLOR,
    theme_font_family=THEME_FONT_FAMILY,
)


def save_plotly_figure(fig, *args, **kwargs):
    """
    Summary
    -------
    Save a Plotly figure after removing the top-level title.

    Parameters
    ----------
    fig
        Plotly figure object.
    *args, **kwargs
        Forwarded to `moimpact.plotting.save_plotly_figure`.

    Returns
    -------
    tuple[Optional[Path], Optional[Path]]
        Output HTML/PNG paths returned by the shared plotting helper.

    Notes
    -----
    Clustering figures are exported without top titles for paper assembly.
    """
    fig.update_layout(title=None)
    return _save_plotly_figure(fig, *args, **kwargs)


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in config file, got {type(config).__name__}.")
    return config


def _setup_logger(log_path: Path) -> logging.Logger:
    """Create a run logger that writes both to stdout and to a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("metaorder_clustering")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file extension for {path}: expected .parquet/.csv/.pkl")


def _resolve_dataset_name(args: argparse.Namespace, cfg: Mapping[str, Any]) -> str:
    value = args.dataset_name or cfg.get("DATASET_NAME") or "ftsemib"
    return str(value)


def _resolve_level(args: argparse.Namespace, cfg: Mapping[str, Any]) -> str:
    value = args.level or cfg.get("LEVEL") or "member"
    value = str(value).strip().lower()
    if value not in {"member", "client"}:
        raise ValueError(f"Invalid level '{value}'. Expected one of: member, client.")
    return value


def _format_path_template(template: str, context: Mapping[str, str]) -> str:
    return format_path_template(template, context)


def _resolve_output_base(args: argparse.Namespace, cfg: Mapping[str, Any], dataset_name: str) -> Path:
    context = {"DATASET_NAME": dataset_name}
    value = str(args.output_file_path or cfg.get("OUTPUT_FILE_PATH") or "out_files/{DATASET_NAME}")
    path = Path(_format_path_template(value, context))
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def _resolve_image_base(args: argparse.Namespace, cfg: Mapping[str, Any], dataset_name: str) -> Path:
    context = {"DATASET_NAME": dataset_name}
    value = str(args.img_output_path or cfg.get("IMG_OUTPUT_PATH") or "images/{DATASET_NAME}")
    path = Path(_format_path_template(value, context))
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def _default_input_paths(output_base: Path, level: str) -> Tuple[Path, Path]:
    base = output_base
    proprietary = base / f"metaorders_info_sameday_filtered_{level}_proprietary.parquet"
    non_proprietary = base / f"metaorders_info_sameday_filtered_{level}_non_proprietary.parquet"
    return proprietary, non_proprietary


def _resolve_io_paths(args: argparse.Namespace, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    dataset_name = _resolve_dataset_name(args, cfg)
    level = _resolve_level(args, cfg)
    out_base = _resolve_output_base(args, cfg, dataset_name)
    img_base = _resolve_image_base(args, cfg, dataset_name)
    prop_default, client_default = _default_input_paths(out_base, level)

    prop_path = Path(args.prop_path) if args.prop_path else prop_default
    client_path = Path(args.client_path) if args.client_path else client_default
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else out_base / f"kmeans_pca_clustering_{level}_{args.group}"
    )
    img_dir = (
        Path(args.img_dir)
        if args.img_dir
        else img_base / f"kmeans_pca_clustering_{level}_{args.group}"
    )
    return {
        "dataset_name": dataset_name,
        "level": level,
        "out_base": out_base,
        "img_base": img_base,
        "prop_path": prop_path,
        "client_path": client_path,
        "out_dir": out_dir,
        "img_dir": img_dir,
    }


def _expected_output_paths(out_dir: Path) -> Dict[str, Path]:
    return {
        "clustered": out_dir / _CLUSTERED_FILE,
        "silhouette": out_dir / _SILHOUETTE_FILE,
        "summary": out_dir / _SUMMARY_FILE,
        "report": out_dir / _PCA_REPORT_FILE,
        "pca_scores": out_dir / _PCA_SCORES_FILE,
        "selection_summary": out_dir / _PCA_FEATURE_SELECTION_SUMMARY_FILE,
        "selection_per_pc": out_dir / _PCA_FEATURE_SELECTION_PC_FILE,
    }


def _has_required_outputs(out_dir: Path) -> bool:
    paths = _expected_output_paths(out_dir)
    return paths["clustered"].exists() and paths["silhouette"].exists()


def _discover_output_dir_for_load(args: argparse.Namespace, resolved: Mapping[str, Any]) -> Path:
    preferred = Path(args.out_dir) if args.out_dir else Path(resolved["out_dir"])
    dataset_dir = Path(resolved["out_base"])
    pattern = f"kmeans_pca_clustering_{resolved['level']}_*"
    candidates: List[Path] = [preferred]
    if dataset_dir.exists():
        candidates.extend(sorted([p for p in dataset_dir.glob(pattern) if p.is_dir()]))

    unique_candidates: List[Path] = []
    seen: set = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(cand)

    valid = [d for d in unique_candidates if d.is_dir() and _has_required_outputs(d)]
    if not valid:
        checked = [str(p) for p in unique_candidates]
        raise FileNotFoundError(
            "Could not find existing clustering outputs for --load. "
            f"Checked: {checked}. Required files: {_CLUSTERED_FILE} and {_SILHOUETTE_FILE}."
        )

    exact_name = f"kmeans_pca_clustering_{resolved['level']}_{args.group}"
    best = max(
        valid,
        key=lambda p: (
            int(p.name == exact_name),
            float(p.stat().st_mtime),
        ),
    )
    return best


def _resolve_effective_img_dir(
    args: argparse.Namespace,
    resolved: Mapping[str, Any],
    effective_out_dir: Path,
) -> Path:
    """Resolve where visualizations are written for the current run."""
    if args.img_dir:
        return Path(args.img_dir)
    img_base = Path(resolved["img_base"])
    return img_base / effective_out_dir.name


def _load_metaorders_by_group(args: argparse.Namespace, resolved: Mapping[str, Any]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    group = args.group
    prop_path: Path = resolved["prop_path"]
    client_path: Path = resolved["client_path"]

    if group in {"both", "proprietary"}:
        if not prop_path.exists():
            raise FileNotFoundError(
                f"Proprietary file not found at {prop_path}. "
                "Pass --prop-path or change --group."
            )
        df_prop = _load_table(prop_path).copy()
        df_prop["Group"] = "proprietary"
        frames.append(df_prop)

    if group in {"both", "client"}:
        if not client_path.exists():
            raise FileNotFoundError(
                f"Client/non-proprietary file not found at {client_path}. "
                "Pass --client-path or change --group."
            )
        df_client = _load_table(client_path).copy()
        df_client["Group"] = "client"
        frames.append(df_client)

    if not frames:
        raise ValueError("No input frames loaded. Check --group and provided paths.")
    return pd.concat(frames, axis=0, ignore_index=True)


def _period_duration_seconds(period: Any) -> float:
    try:
        start_ns, end_ns = period
    except Exception:
        return float("nan")

    try:
        start_ns_i = int(start_ns)
        end_ns_i = int(end_ns)
    except Exception:
        return float("nan")
    return float((end_ns_i - start_ns_i) * 1.0e-9)


def _ensure_impact_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Impact" in out.columns:
        out["Impact"] = pd.to_numeric(out["Impact"], errors="coerce")
        return out
    required = {"Price Change", "Direction", "Daily Vol"}
    if not required.issubset(out.columns):
        missing = sorted(required.difference(out.columns))
        raise KeyError(f"Cannot compute Impact. Missing columns: {missing}")
    out["Impact"] = pd.to_numeric(
        out["Price Change"] * out["Direction"] / out["Daily Vol"], errors="coerce"
    )
    return out


def _ordered_impact_columns(columns: Iterable[str]) -> List[str]:
    names = set(columns)
    horizons: List[Tuple[int, str]] = []
    for col in names:
        match = _IMPACT_HORIZON_PATTERN.match(col)
        if match:
            horizons.append((int(match.group(1)), col))
    horizons.sort(key=lambda x: x[0])

    ordered: List[str] = []
    if "Impact" in names:
        ordered.append("Impact")
    ordered.extend([name for _, name in horizons])
    return ordered


def _add_engineered_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = _ensure_impact_column(df)
    if "Period" not in out.columns:
        raise KeyError("Missing 'Period' column. It is needed to compute DurationSeconds.")
    out["DurationSeconds"] = out["Period"].apply(_period_duration_seconds)

    impact_cols = _ordered_impact_columns(out.columns)
    if not impact_cols:
        raise ValueError("No impact columns found (expected 'Impact' and/or 'Impact_{m}m').")

    abs_impact_cols: List[str] = []
    for col in impact_cols:
        abs_name = "AbsImpact" if col == "Impact" else f"Abs{col}"
        out[abs_name] = pd.to_numeric(out[col], errors="coerce").abs()
        abs_impact_cols.append(abs_name)
    return out, abs_impact_cols


def _default_feature_columns(abs_impact_cols: Sequence[str]) -> List[str]:
    base = [
        "Q",
        "Q/V",
        "Participation Rate",
        "Vt/V",
        "N Child",
        "Daily Vol",
        "DurationSeconds",
    ]
    return base + list(abs_impact_cols)


def _parse_feature_override(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    parts = [c.strip() for c in raw.split(",")]
    parts = [c for c in parts if c]
    if not parts:
        raise ValueError("--features was passed but no valid column names were found.")
    return parts


def _prepare_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    clip_quantiles: Optional[Tuple[float, float]],
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Dict[str, float]], List[str]]:
    work = df.copy()
    missing = [c for c in feature_cols if c not in work.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    work = work.reset_index(drop=True)
    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work[list(feature_cols)] = work[list(feature_cols)].replace([np.inf, -np.inf], np.nan)

    if clip_quantiles is not None:
        low_q, high_q = clip_quantiles
        for col in feature_cols:
            values = work[col].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            lo = float(np.quantile(finite, low_q))
            hi = float(np.quantile(finite, high_q))
            work[col] = work[col].clip(lower=lo, upper=hi)

    transformed_col_names: List[str] = []
    transform_meta: Dict[str, Dict[str, float]] = {}
    for col in feature_cols:
        arr = work[col].to_numpy(dtype=float)
        finite = np.isfinite(arr)
        arr_finite = arr[finite]
        target_col = f"__x_{col}"

        if arr_finite.size == 0:
            work[target_col] = np.nan
            transform_meta[col] = {"method": "log", "eps": float(_EPS_FLOOR)}
            transformed_col_names.append(target_col)
            continue

        nonnegative = bool(np.all(arr_finite >= 0.0))
        if nonnegative:
            positive = arr_finite[arr_finite > 0.0]
            if positive.size == 0:
                eps = float(_EPS_FLOOR)
            else:
                eps = float(max(_EPS_FLOOR, 0.5 * float(np.min(positive))))
            transformed = np.where(finite, np.log(arr + eps), np.nan)
            transform_meta[col] = {"method": "log", "eps": eps}
        else:
            transformed = np.where(finite, np.arcsinh(arr), np.nan)
            transform_meta[col] = {"method": "asinh", "eps": float("nan")}

        work[target_col] = transformed
        transformed_col_names.append(target_col)

    n_before = len(work)
    work = work.dropna(subset=transformed_col_names).reset_index(drop=True)
    n_after = len(work)
    if n_after < 3:
        raise ValueError(
            f"Not enough rows after preprocessing: {n_after} (before={n_before}). "
            "Check missingness and feature choices."
        )
    matrix = work[transformed_col_names].to_numpy(dtype=float)
    return work, matrix, transform_meta, transformed_col_names


def _scale_matrix(matrix: np.ndarray, scaler_kind: str) -> Tuple[np.ndarray, Any]:
    if scaler_kind == "standard":
        scaler = StandardScaler()
    elif scaler_kind == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler kind: {scaler_kind}")
    scaled = scaler.fit_transform(matrix)
    return scaled, scaler


def _validate_k_grid(k_min: int, k_max: int, n_rows: int) -> List[int]:
    if k_min < 2:
        raise ValueError("--k-min must be >= 2.")
    if k_max < k_min:
        raise ValueError("--k-max must be >= --k-min.")
    upper = min(k_max, n_rows - 1)
    if upper < k_min:
        raise ValueError(
            f"Invalid k range for sample size {n_rows}: k_min={k_min}, k_max={k_max}. "
            "Need at least k_min < n_rows."
        )
    return list(range(k_min, upper + 1))


def _parse_k_list(raw: Optional[str]) -> Optional[List[int]]:
    """
    Parse a user-provided cluster-count list.

    Accepted formats:
    - "3,4,5,6"
    - "[3,4,5,6]"
    - "3 4 5 6"
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not text:
        return None

    parts = [p for p in re.split(r"[,\s]+", text) if p]
    values: List[int] = []
    for token in parts:
        try:
            val = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid k value in --k-list: '{token}'") from exc
        if val < 2:
            raise ValueError(f"Invalid k value in --k-list: {val} (must be >= 2).")
        values.append(val)
    if not values:
        return None
    return sorted(set(values))


def _validate_k_list_values(k_list: Sequence[int], n_rows: int) -> List[int]:
    valid: List[int] = []
    for k in k_list:
        if int(k) >= n_rows:
            raise ValueError(
                f"Invalid k={int(k)} for sample size {n_rows}. "
                "Need k < number of valid rows."
            )
        valid.append(int(k))
    return sorted(set(valid))


def _build_scaled_clustering_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    transform_meta: Optional[Mapping[str, Any]],
    scaler_kind: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build transformed+scaled clustering matrix from dataframe features.

    Returns
    -------
    matrix_scaled_valid : np.ndarray
        Matrix containing only rows with finite transformed values.
    valid_mask : np.ndarray
        Boolean mask of rows kept from the input dataframe.
    """
    if len(feature_cols) == 0:
        raise ValueError("No feature columns provided to build clustering matrix.")

    transformed_cols: List[np.ndarray] = []
    for feat in feature_cols:
        if feat not in df.columns:
            raise KeyError(f"Feature '{feat}' not found in clustered dataframe.")
        transformed = _transform_series_for_plotting(
            df[feat],
            feature_name=feat,
            transform_meta=transform_meta if isinstance(transform_meta, dict) else None,
        )
        transformed_cols.append(transformed)

    matrix = np.column_stack(transformed_cols)
    valid_mask = np.all(np.isfinite(matrix), axis=1)
    if np.count_nonzero(valid_mask) < 3:
        raise ValueError("Not enough valid rows to build clustering matrix from existing outputs.")

    matrix_valid = matrix[valid_mask]
    if scaler_kind == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    matrix_scaled_valid = scaler.fit_transform(matrix_valid)
    return matrix_scaled_valid, valid_mask


def _build_scenario_dirs(base_out_dir: Path, base_img_dir: Path, k: int) -> Tuple[Path, Path]:
    out_dir = base_out_dir / "k_scenarios" / f"k_{int(k):02d}"
    img_dir = base_img_dir / "k_scenarios" / f"k_{int(k):02d}"
    return out_dir, img_dir


def _fit_labels_for_k(matrix: np.ndarray, k: int, n_init: int, seed: int) -> np.ndarray:
    model = KMeans(n_clusters=int(k), n_init=int(n_init), random_state=int(seed))
    return model.fit_predict(matrix)


def _evaluate_k_values(
    matrix_pca: np.ndarray,
    k_values: Sequence[int],
    n_init: int,
    seed: int,
    silhouette_sample_size: Optional[int],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for k in k_values:
        model = KMeans(n_clusters=int(k), n_init=n_init, random_state=seed)
        labels = model.fit_predict(matrix_pca)

        if silhouette_sample_size is not None and silhouette_sample_size < matrix_pca.shape[0]:
            sil = float(
                silhouette_score(
                    matrix_pca, labels, sample_size=silhouette_sample_size, random_state=seed
                )
            )
        else:
            sil = float(silhouette_score(matrix_pca, labels))
        rows.append({"k": int(k), "silhouette": sil, "inertia": float(model.inertia_)})
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def _choose_k(curve: pd.DataFrame) -> int:
    if curve.empty:
        raise ValueError("Silhouette curve is empty.")
    best_val = float(curve["silhouette"].max())
    tol = 1.0e-12
    winners = curve[curve["silhouette"] >= (best_val - tol)]
    return int(winners["k"].min())


def _select_features_from_pca(
    pca: PCA,
    feature_cols: Sequence[str],
    per_pc_contribution_threshold: float,
) -> Tuple[List[int], List[str], pd.DataFrame, pd.DataFrame]:
    """
    Select original-space features using retained PCA loadings.

    Parameters
    ----------
    pca : PCA
        Fitted PCA object containing retained components only.
    feature_cols : Sequence[str]
        Original feature names in the same order as PCA input columns.
    per_pc_contribution_threshold : float
        Target cumulative contribution per retained PC, in (0, 1].

    Returns
    -------
    selected_indices : list[int]
        Selected feature indices in the original feature order.
    selected_features : list[str]
        Selected feature names, ordered by global weighted contribution.
    summary_df : pandas.DataFrame
        Per-feature summary with selection flags and aggregate scores.
    per_pc_df : pandas.DataFrame
        Per-PC, per-feature contributions and whether selected for that PC.

    Notes
    -----
    Contributions are computed from squared loadings:
    c_{i,j} = l_{i,j}^2 / sum_k l_{k,j}^2.
    This is sign-invariant and aligns with variance contribution logic.
    """
    if not (0.0 < per_pc_contribution_threshold <= 1.0):
        raise ValueError("per_pc_contribution_threshold must be in (0, 1].")

    components = np.asarray(pca.components_, dtype=float)
    if components.ndim != 2:
        raise ValueError("PCA components must be a 2D array.")
    n_pcs, n_features = components.shape
    if n_features != len(feature_cols):
        raise ValueError(
            f"Feature length mismatch: PCA has {n_features} features, "
            f"feature_cols has {len(feature_cols)}."
        )

    evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
    if evr.shape[0] != n_pcs:
        raise ValueError("Explained variance ratio length does not match number of retained PCs.")

    selected_idx_set: set = set()
    selected_count = np.zeros(n_features, dtype=int)
    per_pc_rows: List[Dict[str, Any]] = []

    for pc_idx in range(n_pcs):
        loadings = components[pc_idx, :]
        sq_load = np.square(loadings)
        denom = float(np.sum(sq_load))
        if not np.isfinite(denom) or denom <= 0.0:
            continue
        contrib = sq_load / denom
        order = np.argsort(contrib)[::-1]
        cum = np.cumsum(contrib[order])
        n_keep = int(np.searchsorted(cum, per_pc_contribution_threshold, side="left") + 1)
        n_keep = max(1, min(n_keep, n_features))
        keep_set = set(order[:n_keep].tolist())

        for idx in keep_set:
            selected_idx_set.add(int(idx))
            selected_count[idx] += 1

        for rank, feat_idx in enumerate(order, start=1):
            feat_name = feature_cols[int(feat_idx)]
            per_pc_rows.append(
                {
                    "pc": int(pc_idx + 1),
                    "feature": feat_name,
                    "feature_index": int(feat_idx),
                    "rank": int(rank),
                    "loading": float(loadings[feat_idx]),
                    "squared_loading": float(sq_load[feat_idx]),
                    "contribution": float(contrib[feat_idx]),
                    "cumulative_contribution": float(cum[rank - 1]),
                    "selected_for_pc": bool(int(feat_idx) in keep_set),
                    "pc_explained_variance_ratio": float(evr[pc_idx]),
                }
            )

    if not selected_idx_set:
        fallback = int(np.argmax(np.sum(np.square(components), axis=0)))
        selected_idx_set.add(fallback)
        selected_count[fallback] = 1

    global_weighted = np.sum(np.square(components) * evr.reshape(-1, 1), axis=0)
    mean_abs_loading = np.mean(np.abs(components), axis=0)
    selected_indices = sorted(
        [int(i) for i in selected_idx_set],
        key=lambda i: (float(global_weighted[i]), float(mean_abs_loading[i])),
        reverse=True,
    )
    selected_features = [feature_cols[i] for i in selected_indices]

    summary_rows: List[Dict[str, Any]] = []
    for idx, feat_name in enumerate(feature_cols):
        summary_rows.append(
            {
                "feature": feat_name,
                "feature_index": int(idx),
                "selected": bool(idx in selected_idx_set),
                "selected_in_n_pcs": int(selected_count[idx]),
                "global_weighted_squared_loading": float(global_weighted[idx]),
                "mean_abs_loading": float(mean_abs_loading[idx]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["selected", "global_weighted_squared_loading"], ascending=[False, False]
    )
    if per_pc_rows:
        per_pc_df = pd.DataFrame(per_pc_rows).sort_values(["pc", "rank"], ascending=[True, True])
    else:
        per_pc_df = pd.DataFrame(
            columns=[
                "pc",
                "feature",
                "feature_index",
                "rank",
                "loading",
                "squared_loading",
                "contribution",
                "cumulative_contribution",
                "selected_for_pc",
                "pc_explained_variance_ratio",
            ]
        )
    return selected_indices, selected_features, summary_df, per_pc_df


def _summarize_clusters(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    summary_rows: List[Dict[str, Any]] = []
    for cluster_id in sorted(pd.unique(df["Cluster"])):
        sub = df[df["Cluster"] == cluster_id]
        row: Dict[str, Any] = {"Cluster": int(cluster_id), "n_metaorders": int(len(sub))}
        if "Group" in sub.columns:
            counts = sub["Group"].value_counts(dropna=False).to_dict()
            row["n_client"] = int(counts.get("client", 0))
            row["n_proprietary"] = int(counts.get("proprietary", 0))

        for col in feature_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                row[f"{col}_median"] = np.nan
                row[f"{col}_q25"] = np.nan
                row[f"{col}_q75"] = np.nan
            else:
                row[f"{col}_median"] = float(np.median(vals))
                row[f"{col}_q25"] = float(np.quantile(vals, 0.25))
                row[f"{col}_q75"] = float(np.quantile(vals, 0.75))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _write_plotly_figure(fig: go.Figure, html_path: Path, png_path: Optional[Path] = None) -> None:
    base_dir = html_path.parent
    dirs = make_plot_output_dirs(base_dir, use_subdirs=True)
    stem = html_path.stem
    save_plotly_figure(
        fig,
        stem=stem,
        dirs=dirs,
        write_html=True,
        write_png=png_path is not None,
        strict_png=False,
    )


def _plot_silhouette_curve(
    curve: pd.DataFrame,
    best_k: int,
    out_dir: Path,
    clustering_space_label: str = "selected original feature space",
) -> None:
    if curve.empty:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["k"],
            y=curve["silhouette"],
            mode="lines+markers",
            name="Silhouette",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
        )
    )
    if int(best_k) in set(pd.to_numeric(curve["k"], errors="coerce").dropna().astype(int).tolist()):
        best_row = curve[curve["k"] == int(best_k)].iloc[0]
    else:
        best_row = curve.iloc[int(np.argmax(pd.to_numeric(curve["silhouette"], errors="coerce").to_numpy()))]
        best_k = int(best_row["k"])
    fig.add_trace(
        go.Scatter(
            x=[best_k],
            y=[float(best_row["silhouette"])],
            mode="markers",
            name=f"Selected k={best_k}",
            marker=dict(color="#d62728", size=12, symbol="diamond"),
        )
    )
    fig.update_layout(
        title=f"Silhouette score vs k (k-means on {clustering_space_label})",
        xaxis_title="Number of clusters k",
        yaxis_title="Silhouette score",
        template=PLOTLY_TEMPLATE_NAME,
    )
    _write_plotly_figure(
        fig,
        out_dir / "silhouette_curve.html",
        out_dir / "silhouette_curve.png",
    )


def _plot_pca_cumulative_variance(cumulative: Sequence[float], out_dir: Path) -> None:
    cumulative_arr = np.asarray(cumulative, dtype=float)
    if cumulative_arr.size == 0:
        return
    cumulative_arr = cumulative_arr[np.isfinite(cumulative_arr)]
    if cumulative_arr.size == 0:
        return
    cumulative_arr = np.clip(cumulative_arr, 0.0, 1.0)
    x = np.arange(1, len(cumulative_arr) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cumulative_arr,
            mode="lines+markers",
            name="Cumulative explained variance",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=7),
        )
    )
    fig.update_layout(
        title="PCA cumulative explained variance",
        xaxis_title="Number of principal components",
        yaxis_title="Cumulative explained variance",
        template=PLOTLY_TEMPLATE_NAME,
    )
    _write_plotly_figure(
        fig,
        out_dir / "pca_cumulative_explained_variance.html",
        out_dir / "pca_cumulative_explained_variance.png",
    )


def _plot_pca_clusters(
    matrix_pca: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    seed: int,
    title: str = "Metaorders in PCA projection of clustering space (PC1 vs PC2)",
) -> None:
    if matrix_pca.shape[1] < 2:
        return
    max_points = min(50000, matrix_pca.shape[0])
    rng = np.random.default_rng(seed)
    if matrix_pca.shape[0] > max_points:
        idx = np.sort(rng.choice(matrix_pca.shape[0], size=max_points, replace=False))
    else:
        idx = np.arange(matrix_pca.shape[0], dtype=int)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=matrix_pca[idx, 0],
            y=matrix_pca[idx, 1],
            mode="markers",
            marker=dict(
                size=5,
                color=labels[idx],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Cluster"),
                opacity=0.7,
            ),
            text=[f"Cluster={int(c)}" for c in labels[idx]],
            hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>%{text}<extra></extra>",
            name="Metaorders",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="PC1",
        yaxis_title="PC2",
        template=PLOTLY_TEMPLATE_NAME,
    )
    _write_plotly_figure(
        fig,
        out_dir / "pca_clusters_pc1_pc2.html",
        out_dir / "pca_clusters_pc1_pc2.png",
    )


def _sanitize_token(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()
    return token if token else "feature"


def _transform_series_for_plotting(
    series: pd.Series,
    feature_name: str,
    transform_meta: Optional[Mapping[str, Any]],
) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if transform_meta is None:
        return values
    meta = transform_meta.get(feature_name)
    if not isinstance(meta, dict):
        return values
    method = meta.get("method")
    if method == "log":
        eps = float(meta.get("eps", _EPS_FLOOR))
        return np.where(np.isfinite(values), np.log(values + eps), np.nan)
    if method == "asinh":
        return np.where(np.isfinite(values), np.arcsinh(values), np.nan)
    return values


def _axis_label_for_feature(feature_name: str, transform_meta: Optional[Mapping[str, Any]]) -> str:
    if transform_meta is None:
        return feature_name
    meta = transform_meta.get(feature_name)
    if not isinstance(meta, dict):
        return feature_name
    method = meta.get("method")
    if method == "log":
        return f"log({feature_name} + eps)"
    if method == "asinh":
        return f"asinh({feature_name})"
    return feature_name


def _build_feature_triplets(feature_cols: Sequence[str], n_triplets: int) -> List[Tuple[str, str, str]]:
    if n_triplets <= 0:
        return []
    triplets: List[Tuple[str, str, str]] = []
    limit = min(len(feature_cols), n_triplets * 3)
    for idx in range(0, limit, 3):
        if idx + 2 < len(feature_cols):
            triplets.append((feature_cols[idx], feature_cols[idx + 1], feature_cols[idx + 2]))
    return triplets


def _plot_feature_space_triplets(
    clustered_df: pd.DataFrame,
    feature_cols_ranked: Sequence[str],
    out_dir: Path,
    seed: int,
    n_triplets: int,
    sample_size: int,
    transform_meta: Optional[Mapping[str, Any]],
) -> int:
    if "Cluster" not in clustered_df.columns:
        return 0
    if sample_size <= 0:
        sample_size = 50000

    triplets = _build_feature_triplets(feature_cols_ranked, n_triplets=n_triplets)
    if not triplets:
        return 0

    generated = 0
    base_rng = np.random.default_rng(seed)
    for triplet_idx, (f1, f2, f3) in enumerate(triplets, start=1):
        if not ({f1, f2, f3}.issubset(clustered_df.columns)):
            continue

        x = _transform_series_for_plotting(clustered_df[f1], f1, transform_meta=transform_meta)
        y = _transform_series_for_plotting(clustered_df[f2], f2, transform_meta=transform_meta)
        z = _transform_series_for_plotting(clustered_df[f3], f3, transform_meta=transform_meta)
        labels = pd.to_numeric(clustered_df["Cluster"], errors="coerce").to_numpy()

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(labels)
        n_valid = int(np.count_nonzero(valid))
        if n_valid < 2:
            continue

        valid_idx = np.where(valid)[0]
        if n_valid > sample_size:
            idx = np.sort(base_rng.choice(valid_idx, size=sample_size, replace=False))
        else:
            idx = valid_idx

        x_plot = x[idx]
        y_plot = y[idx]
        z_plot = z[idx]
        label_plot = labels[idx].astype(int)

        if "Group" in clustered_df.columns:
            group_plot = clustered_df["Group"].iloc[idx].astype(str).to_numpy()
            hover_text = [
                f"Cluster={int(c)}<br>Group={g}" for c, g in zip(label_plot, group_plot)
            ]
        else:
            hover_text = [f"Cluster={int(c)}" for c in label_plot]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=x_plot,
                y=y_plot,
                z=z_plot,
                mode="markers",
                marker=dict(
                    size=2.5,
                    color=label_plot,
                    colorscale="Viridis",
                    opacity=0.65,
                    colorbar=dict(title="Cluster"),
                ),
                text=hover_text,
                hovertemplate=(
                    f"{_axis_label_for_feature(f1, transform_meta)}=%{{x:.3f}}<br>"
                    f"{_axis_label_for_feature(f2, transform_meta)}=%{{y:.3f}}<br>"
                    f"{_axis_label_for_feature(f3, transform_meta)}=%{{z:.3f}}<br>"
                    "%{text}<extra></extra>"
                ),
                name="Metaorders",
            )
        )
        fig.update_layout(
            title=(
                f"Cluster structure in selected feature space "
                f"(triplet {triplet_idx}: {f1}, {f2}, {f3})"
            ),
            scene=dict(
                xaxis_title=_axis_label_for_feature(f1, transform_meta),
                yaxis_title=_axis_label_for_feature(f2, transform_meta),
                zaxis_title=_axis_label_for_feature(f3, transform_meta),
            ),
            template=PLOTLY_TEMPLATE_NAME,
        )

        file_stub = (
            f"cluster_feature_triplet_{triplet_idx:02d}_"
            f"{_sanitize_token(f1)}__{_sanitize_token(f2)}__{_sanitize_token(f3)}"
        )
        _write_plotly_figure(
            fig,
            out_dir / f"{file_stub}.html",
            out_dir / f"{file_stub}.png",
        )
        generated += 1
    return generated


def _resolve_tsne_feature_columns(
    clustered_df: pd.DataFrame,
    report: Optional[Mapping[str, Any]],
    selection_summary_df: Optional[pd.DataFrame],
    top_n: int,
) -> List[str]:
    if top_n <= 0:
        return []

    if report is not None and isinstance(report.get("clustering_feature_columns"), list):
        ordered = [str(c) for c in report["clustering_feature_columns"] if str(c) in clustered_df.columns]
        if ordered:
            return ordered[:top_n]

    if selection_summary_df is not None and not selection_summary_df.empty:
        needed = {"feature", "selected", "global_weighted_squared_loading"}
        if needed.issubset(selection_summary_df.columns):
            df = selection_summary_df.copy()
            selected_bool = df["selected"]
            if selected_bool.dtype == object:
                selected_bool = (
                    selected_bool.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
                )
            df = df[selected_bool]
            if not df.empty:
                df = df.sort_values("global_weighted_squared_loading", ascending=False)
                ordered = [str(c) for c in df["feature"].tolist() if str(c) in clustered_df.columns]
                if ordered:
                    return ordered[:top_n]

    fallback = _infer_feature_columns_for_visualization(clustered_df, summary_df=None, report=report)
    return fallback[:top_n]


def _resolve_clustering_feature_columns_from_report(
    clustered_df: pd.DataFrame,
    report: Optional[Mapping[str, Any]],
) -> List[str]:
    if report is not None and isinstance(report.get("clustering_feature_columns"), list):
        cols = [str(c) for c in report["clustering_feature_columns"] if str(c) in clustered_df.columns]
        if cols:
            return cols
    if report is not None and isinstance(report.get("feature_columns"), list):
        cols = [str(c) for c in report["feature_columns"] if str(c) in clustered_df.columns]
        if cols:
            return cols
    cols = _infer_feature_columns_for_visualization(clustered_df, summary_df=None, report=report)
    if cols:
        return cols
    raise ValueError("Unable to resolve clustering feature columns from saved outputs.")


def _plot_tsne_clusters_top_features(
    clustered_df: pd.DataFrame,
    out_dir: Path,
    seed: int,
    feature_cols_ranked: Sequence[str],
    top_n_features: int,
    sample_size: int,
    perplexity: float,
    max_iter: int,
    transform_meta: Optional[Mapping[str, Any]],
    scaler_kind: str,
) -> Optional[Dict[str, Any]]:
    if "Cluster" not in clustered_df.columns:
        return None

    feature_cols = [f for f in list(feature_cols_ranked)[:top_n_features] if f in clustered_df.columns]
    if len(feature_cols) < 2:
        return None

    arrays = [
        _transform_series_for_plotting(clustered_df[f], f, transform_meta=transform_meta)
        for f in feature_cols
    ]
    matrix = np.column_stack(arrays)
    labels = pd.to_numeric(clustered_df["Cluster"], errors="coerce").to_numpy()
    valid = np.all(np.isfinite(matrix), axis=1) & np.isfinite(labels)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 5:
        return None

    valid_idx = np.where(valid)[0]
    sample_size_eff = min(int(sample_size), n_valid)
    rng = np.random.default_rng(seed)
    if n_valid > sample_size_eff:
        idx = np.sort(rng.choice(valid_idx, size=sample_size_eff, replace=False))
    else:
        idx = valid_idx

    matrix_sample = matrix[idx, :]
    labels_sample = labels[idx].astype(int)
    if scaler_kind == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    matrix_sample = scaler.fit_transform(matrix_sample)

    max_valid_perplexity = max(1.0, (matrix_sample.shape[0] - 1.0) / 3.0)
    perplexity_eff = float(min(float(perplexity), max_valid_perplexity))
    if perplexity_eff < 1.0:
        return None

    tsne = TSNE(
        n_components=3,
        perplexity=perplexity_eff,
        max_iter=int(max_iter),
        learning_rate="auto",
        init="pca",
        random_state=int(seed),
        method="barnes_hut",
    )
    embedding = tsne.fit_transform(matrix_sample)

    if "Group" in clustered_df.columns:
        group_sample = clustered_df["Group"].iloc[idx].astype(str).to_numpy()
        hover_text = [f"Cluster={int(c)}<br>Group={g}" for c, g in zip(labels_sample, group_sample)]
    else:
        hover_text = [f"Cluster={int(c)}" for c in labels_sample]

    fig_2d = go.Figure()
    fig_2d.add_trace(
        go.Scattergl(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode="markers",
            marker=dict(
                size=5,
                color=labels_sample,
                colorscale="Viridis",
                opacity=0.7,
                showscale=True,
                colorbar=dict(title="Cluster"),
            ),
            text=hover_text,
            hovertemplate="tSNE-1=%{x:.3f}<br>tSNE-2=%{y:.3f}<br>%{text}<extra></extra>",
            name="Metaorders",
        )
    )
    fig_2d.update_layout(
        title=(
            "t-SNE projection (2D view) of selected clustering features "
            f"(top {len(feature_cols)} features)"
        ),
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template=PLOTLY_TEMPLATE_NAME,
    )
    _write_plotly_figure(
        fig_2d,
        out_dir / "tsne_clusters_top_features.html",
        out_dir / "tsne_clusters_top_features.png",
    )

    fig_3d = go.Figure()
    fig_3d.add_trace(
        go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode="markers",
            marker=dict(
                size=2.5,
                color=labels_sample,
                colorscale="Viridis",
                opacity=0.65,
                colorbar=dict(title="Cluster"),
            ),
            text=hover_text,
            hovertemplate=(
                "tSNE-1=%{x:.3f}<br>tSNE-2=%{y:.3f}<br>tSNE-3=%{z:.3f}<br>%{text}<extra></extra>"
            ),
            name="Metaorders",
        )
    )
    fig_3d.update_layout(
        title=(
            "t-SNE projection (3D view) of selected clustering features "
            f"(top {len(feature_cols)} features)"
        ),
        scene=dict(
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            zaxis_title="t-SNE 3",
        ),
        template=PLOTLY_TEMPLATE_NAME,
    )
    _write_plotly_figure(
        fig_3d,
        out_dir / "tsne_clusters_top_features_3d.html",
        out_dir / "tsne_clusters_top_features_3d.png",
    )

    return {
        "feature_cols": feature_cols,
        "n_valid_rows": int(n_valid),
        "n_used_rows": int(matrix_sample.shape[0]),
        "perplexity_effective": float(perplexity_eff),
        "max_iter": int(max_iter),
        "n_components": 3,
    }


def _plot_cluster_sizes(clustered_df: pd.DataFrame, out_dir: Path) -> None:
    counts = (
        clustered_df["Cluster"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Count")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts["Cluster"].astype(str),
            y=counts["Count"],
            marker=dict(color="#4c78a8"),
            name="Metaorders",
        )
    )
    fig.update_layout(
        title="Cluster sizes",
        xaxis_title="Cluster",
        yaxis_title="Number of metaorders",
        template=PLOTLY_TEMPLATE_NAME,
    )
    _write_plotly_figure(
        fig,
        out_dir / "cluster_sizes.html",
        out_dir / "cluster_sizes.png",
    )


def _plot_cluster_group_composition(clustered_df: pd.DataFrame, out_dir: Path) -> None:
    if "Group" not in clustered_df.columns:
        return
    counts = (
        clustered_df.groupby(["Cluster", "Group"], dropna=False)
        .size()
        .rename("Count")
        .reset_index()
        .sort_values(["Cluster", "Group"])
    )
    if counts.empty:
        return

    fig = go.Figure()
    for group_name in sorted(counts["Group"].dropna().astype(str).unique()):
        sub = counts[counts["Group"].astype(str) == group_name]
        fig.add_trace(
            go.Bar(
                x=sub["Cluster"].astype(str),
                y=sub["Count"],
                name=group_name,
            )
        )

    fig.update_layout(
        title="Cluster composition by metaorder group",
        xaxis_title="Cluster",
        yaxis_title="Number of metaorders",
        barmode="stack",
        template=PLOTLY_TEMPLATE_NAME,
    )
    _write_plotly_figure(
        fig,
        out_dir / "cluster_group_composition.html",
        out_dir / "cluster_group_composition.png",
    )


def _infer_feature_columns_for_visualization(
    clustered_df: pd.DataFrame,
    summary_df: Optional[pd.DataFrame],
    report: Optional[Mapping[str, Any]],
) -> List[str]:
    if report is not None:
        cols = report.get("clustering_feature_columns")
        if isinstance(cols, list):
            available = [str(c) for c in cols if str(c) in clustered_df.columns]
            if available:
                return available

        cols = report.get("feature_columns")
        if isinstance(cols, list):
            available = [str(c) for c in cols if str(c) in clustered_df.columns]
            if available:
                return available

    if summary_df is not None:
        median_cols = [
            c for c in summary_df.columns if c.endswith("_median") and c not in {"Cluster_median"}
        ]
        inferred = [c[: -len("_median")] for c in median_cols if c[: -len("_median")] in clustered_df.columns]
        if inferred:
            return inferred

    fallback_base = ["Q", "Q/V", "Participation Rate", "Vt/V", "N Child", "Daily Vol", "DurationSeconds"]
    fallback_abs = sorted([c for c in clustered_df.columns if c.startswith("AbsImpact")])
    return [c for c in fallback_base + fallback_abs if c in clustered_df.columns]


def _plot_cluster_feature_heatmap(
    summary_df: pd.DataFrame,
    feature_cols: Sequence[str],
    out_dir: Path,
) -> None:
    if summary_df.empty or not feature_cols:
        return
    median_cols = [f"{f}_median" for f in feature_cols if f"{f}_median" in summary_df.columns]
    if not median_cols:
        return

    frame = summary_df[["Cluster", *median_cols]].copy().sort_values("Cluster")
    values = frame[median_cols].to_numpy(dtype=float)
    means = np.nanmean(values, axis=0, keepdims=True)
    stds = np.nanstd(values, axis=0, keepdims=True)
    stds = np.where(stds > 0.0, stds, 1.0)
    zscores = (values - means) / stds

    feature_names = [c[: -len("_median")] for c in median_cols]
    fig = go.Figure(
        data=go.Heatmap(
            z=zscores,
            x=feature_names,
            y=frame["Cluster"].astype(str),
            colorscale="RdBu",
            zmid=0.0,
            colorbar=dict(title="z-score"),
            hovertemplate="Cluster=%{y}<br>Feature=%{x}<br>z=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Cluster feature profiles (median, z-score across clusters)",
        xaxis_title="Feature",
        yaxis_title="Cluster",
        template=PLOTLY_TEMPLATE_NAME,
    )
    _write_plotly_figure(
        fig,
        out_dir / "cluster_feature_profiles_heatmap.html",
        out_dir / "cluster_feature_profiles_heatmap.png",
    )


def _parse_abs_impact_horizon(name: str) -> int:
    if name == "AbsImpact":
        return 0
    match = re.match(r"^AbsImpact_(\d+)m$", name)
    if match:
        return int(match.group(1))
    return 10**9


def _plot_abs_impact_profiles(clustered_df: pd.DataFrame, out_dir: Path) -> None:
    abs_cols = [c for c in clustered_df.columns if c == "AbsImpact" or re.match(r"^AbsImpact_(\d+)m$", c)]
    abs_cols = sorted(abs_cols, key=_parse_abs_impact_horizon)
    if not abs_cols or "Cluster" not in clustered_df.columns:
        return

    stats = (
        clustered_df.groupby("Cluster", dropna=False)[abs_cols]
        .median(numeric_only=True)
        .reset_index()
        .sort_values("Cluster")
    )
    if stats.empty:
        return

    labels = ["end" if c == "AbsImpact" else c.replace("AbsImpact_", "") for c in abs_cols]
    fig = go.Figure()
    for _, row in stats.iterrows():
        cluster_id = int(row["Cluster"])
        y = [float(row[col]) if np.isfinite(row[col]) else np.nan for col in abs_cols]
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=y,
                mode="lines+markers",
                name=f"Cluster {cluster_id}",
            )
        )

    fig.update_layout(
        title="Median absolute impact profile by cluster",
        xaxis_title="Impact horizon",
        yaxis_title="Median absolute impact",
        template=PLOTLY_TEMPLATE_NAME,
    )
    _write_plotly_figure(
        fig,
        out_dir / "cluster_abs_impact_profiles.html",
        out_dir / "cluster_abs_impact_profiles.png",
    )


def _plot_selected_feature_global_contributions(
    selection_summary_df: Optional[pd.DataFrame],
    out_dir: Path,
) -> None:
    if selection_summary_df is None or selection_summary_df.empty:
        return
    required = {"feature", "selected", "global_weighted_squared_loading"}
    if not required.issubset(selection_summary_df.columns):
        return

    df = selection_summary_df.copy()
    df = df.sort_values("global_weighted_squared_loading", ascending=True)
    colors = np.where(df["selected"].astype(bool), "#1f77b4", "#bdbdbd")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["global_weighted_squared_loading"],
            y=df["feature"],
            orientation="h",
            marker=dict(color=colors),
            customdata=np.stack([df["selected"].astype(bool).to_numpy()], axis=1),
            hovertemplate=(
                "Feature=%{y}<br>"
                "Global weighted squared loading=%{x:.6f}<br>"
                "Selected=%{customdata[0]}<extra></extra>"
            ),
            name="Feature score",
        )
    )
    fig.update_layout(
        title="PCA-guided feature relevance (global weighted squared loading)",
        xaxis_title="Score",
        yaxis_title="Feature",
        template=PLOTLY_TEMPLATE_NAME,
    )
    _write_plotly_figure(
        fig,
        out_dir / "selected_feature_global_contributions.html",
        out_dir / "selected_feature_global_contributions.png",
    )


def _plot_per_pc_feature_contributions(
    selection_per_pc_df: Optional[pd.DataFrame],
    out_dir: Path,
) -> None:
    if selection_per_pc_df is None or selection_per_pc_df.empty:
        return
    required = {"pc", "feature", "contribution", "selected_for_pc"}
    if not required.issubset(selection_per_pc_df.columns):
        return

    top_selected = (
        selection_per_pc_df[selection_per_pc_df["selected_for_pc"]]
        .groupby("feature")["contribution"]
        .max()
        .sort_values(ascending=False)
    )
    if top_selected.empty:
        return
    top_features = top_selected.head(20).index.tolist()
    sub = selection_per_pc_df[selection_per_pc_df["feature"].isin(top_features)].copy()
    if sub.empty:
        return

    pivot = (
        sub.pivot_table(
            index="pc",
            columns="feature",
            values="contribution",
            aggfunc="max",
            fill_value=0.0,
        )
        .sort_index(axis=0)
        .reindex(columns=top_features)
    )
    y_labels = [f"PC{int(pc)}" for pc in pivot.index.tolist()]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=pivot.columns.tolist(),
            y=y_labels,
            colorscale="Viridis",
            colorbar=dict(title="Contribution"),
            hovertemplate="PC=%{y}<br>Feature=%{x}<br>Contribution=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Per-PC feature contributions (top selected features)",
        xaxis_title="Feature",
        yaxis_title="Principal component",
        template=PLOTLY_TEMPLATE_NAME,
    )
    _write_plotly_figure(
        fig,
        out_dir / "per_pc_feature_contributions_heatmap.html",
        out_dir / "per_pc_feature_contributions_heatmap.png",
    )


def _ordered_pc_columns(columns: Iterable[str]) -> List[str]:
    pcs: List[Tuple[int, str]] = []
    for col in columns:
        match = re.match(r"^PC(\d+)$", col)
        if match:
            pcs.append((int(match.group(1)), col))
    pcs.sort(key=lambda x: x[0])
    return [name for _, name in pcs]


def _fallback_transform_column(arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    finite = np.isfinite(arr)
    arr_finite = arr[finite]
    if arr_finite.size == 0:
        return np.where(finite, np.nan, np.nan), {"method": "log", "eps": float(_EPS_FLOOR)}

    nonnegative = bool(np.all(arr_finite >= 0.0))
    if nonnegative:
        positive = arr_finite[arr_finite > 0.0]
        eps = float(_EPS_FLOOR if positive.size == 0 else max(_EPS_FLOOR, 0.5 * float(np.min(positive))))
        transformed = np.where(finite, np.log(arr + eps), np.nan)
        return transformed, {"method": "log", "eps": eps}

    transformed = np.where(finite, np.arcsinh(arr), np.nan)
    return transformed, {"method": "asinh", "eps": float("nan")}


def _compute_projection_from_clustered_df(
    clustered_df: pd.DataFrame,
    report: Optional[Mapping[str, Any]],
) -> Optional[pd.DataFrame]:
    """
    Build a PCA projection for visualization from clustered data.

    This is used in load mode to regenerate PC scatter plots even when
    `pca_scores.parquet` is missing or outdated.
    """
    if report is not None and isinstance(report.get("clustering_feature_columns"), list):
        feature_cols = [str(c) for c in report["clustering_feature_columns"] if str(c) in clustered_df.columns]
    elif report is not None and isinstance(report.get("feature_columns"), list):
        feature_cols = [str(c) for c in report["feature_columns"] if str(c) in clustered_df.columns]
    else:
        feature_cols = []
    if len(feature_cols) < 1:
        return None

    transform_meta = report.get("transform_meta") if isinstance(report, dict) else None
    if not isinstance(transform_meta, dict):
        transform_meta = {}

    matrix_cols: List[np.ndarray] = []
    for col in feature_cols:
        arr = pd.to_numeric(clustered_df[col], errors="coerce").to_numpy(dtype=float)
        meta = transform_meta.get(col, {})
        method = meta.get("method") if isinstance(meta, dict) else None
        if method == "log":
            eps = float(meta.get("eps", _EPS_FLOOR))
            transformed = np.where(np.isfinite(arr), np.log(arr + eps), np.nan)
        elif method == "asinh":
            transformed = np.where(np.isfinite(arr), np.arcsinh(arr), np.nan)
        else:
            transformed, _ = _fallback_transform_column(arr)
        matrix_cols.append(transformed)

    matrix = np.column_stack(matrix_cols)
    valid = np.all(np.isfinite(matrix), axis=1)
    if np.count_nonzero(valid) < 3:
        return None
    matrix_valid = matrix[valid]

    scaler_kind = str(report.get("scaler", "standard")) if isinstance(report, dict) else "standard"
    if scaler_kind == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix_valid)

    n_components = min(10, matrix_scaled.shape[1], matrix_scaled.shape[0])
    if n_components < 1:
        return None
    projection = PCA(n_components=n_components, svd_solver="full").fit_transform(matrix_scaled)
    proj_df = pd.DataFrame(projection, columns=[f"PC{i + 1}" for i in range(n_components)])
    labels = pd.to_numeric(clustered_df.loc[valid, "Cluster"], errors="coerce")
    proj_df["Cluster"] = labels.to_numpy()
    if "Group" in clustered_df.columns:
        proj_df["Group"] = clustered_df.loc[valid, "Group"].to_numpy()
    return proj_df
def _load_report_if_available(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw
    return None


def _load_existing_outputs(out_dir: Path) -> Dict[str, Any]:
    paths = _expected_output_paths(out_dir)
    if not paths["clustered"].exists() or not paths["silhouette"].exists():
        raise FileNotFoundError(
            f"Missing required output files in {out_dir}. "
            f"Expected at least {paths['clustered']} and {paths['silhouette']}."
        )
    outputs: Dict[str, Any] = {
        "paths": paths,
        "clustered_df": pd.read_parquet(paths["clustered"]),
        "silhouette_curve": pd.read_csv(paths["silhouette"]),
        "summary_df": pd.read_csv(paths["summary"]) if paths["summary"].exists() else None,
        "selection_summary_df": (
            pd.read_csv(paths["selection_summary"]) if paths["selection_summary"].exists() else None
        ),
        "selection_per_pc_df": (
            pd.read_csv(paths["selection_per_pc"]) if paths["selection_per_pc"].exists() else None
        ),
        "report": _load_report_if_available(paths["report"]),
    }
    if paths["pca_scores"].exists():
        outputs["pca_scores_df"] = pd.read_parquet(paths["pca_scores"])
    else:
        outputs["pca_scores_df"] = None
    return outputs


def _render_result_visualizations(
    out_dir: Path,
    clustered_df: pd.DataFrame,
    silhouette_curve: pd.DataFrame,
    report: Optional[Mapping[str, Any]],
    summary_df: Optional[pd.DataFrame],
    selection_summary_df: Optional[pd.DataFrame],
    selection_per_pc_df: Optional[pd.DataFrame],
    pca_scores_df: Optional[pd.DataFrame],
    seed: int,
    n_feature_triplets: int,
    feature_triplet_sample_size: int,
    run_tsne: bool,
    tsne_top_n_features: int,
    tsne_sample_size: int,
    tsne_perplexity: float,
    tsne_max_iter: int,
    logger: Optional[logging.Logger],
    highlight_k: Optional[int] = None,
) -> pd.DataFrame:
    out_dirs = make_plot_output_dirs(out_dir, use_subdirs=True)

    # Remove deprecated projection scatter plots to avoid mixing old and new
    # semantics (current default visualization is feature-space triplets).
    for stale_path in [
        out_dir / "pca_clusters_pc1_pc2.html",
        out_dir / "pca_clusters_pc1_pc2.png",
        out_dirs.html_dir / "pca_clusters_pc1_pc2.html",
        out_dirs.png_dir / "pca_clusters_pc1_pc2.png",
    ]:
        if stale_path.exists():
            try:
                stale_path.unlink()
            except OSError:
                pass

    clustering_space_label = "selected original feature space"
    if report is not None:
        report_space = report.get("clustering_space")
        if isinstance(report_space, str) and report_space.strip():
            clustering_space_label = report_space.replace("_", " ")

    if highlight_k is None:
        best_k = _choose_k(silhouette_curve)
    else:
        best_k = int(highlight_k)
    _plot_silhouette_curve(
        silhouette_curve,
        best_k=best_k,
        out_dir=out_dir,
        clustering_space_label=clustering_space_label,
    )

    if report is not None:
        cumulative = report.get("pca_cumulative_explained_variance")
        if cumulative is not None:
            try:
                _plot_pca_cumulative_variance(cumulative=np.asarray(cumulative, dtype=float), out_dir=out_dir)
            except Exception:
                pass

    feature_cols = _infer_feature_columns_for_visualization(
        clustered_df=clustered_df,
        summary_df=summary_df,
        report=report,
    )
    transform_meta = report.get("transform_meta") if isinstance(report, dict) else None
    if not isinstance(transform_meta, dict):
        transform_meta = None

    _plot_feature_space_triplets(
        clustered_df=clustered_df,
        feature_cols_ranked=feature_cols,
        out_dir=out_dir,
        seed=seed,
        n_triplets=n_feature_triplets,
        sample_size=feature_triplet_sample_size,
        transform_meta=transform_meta,
    )

    if logger is not None:
        logger.info("Feature-space triplet plots generated using ranked features: %s", feature_cols)
    if summary_df is None:
        summary_df = _summarize_clusters(clustered_df, feature_cols=feature_cols)
        summary_df.to_csv(out_dir / _SUMMARY_FILE, index=False)

    _plot_cluster_sizes(clustered_df, out_dir=out_dir)
    _plot_cluster_group_composition(clustered_df, out_dir=out_dir)
    _plot_cluster_feature_heatmap(summary_df, feature_cols=feature_cols, out_dir=out_dir)
    _plot_abs_impact_profiles(clustered_df, out_dir=out_dir)
    _plot_selected_feature_global_contributions(selection_summary_df=selection_summary_df, out_dir=out_dir)
    _plot_per_pc_feature_contributions(selection_per_pc_df=selection_per_pc_df, out_dir=out_dir)

    if run_tsne:
        tsne_features = _resolve_tsne_feature_columns(
            clustered_df=clustered_df,
            report=report,
            selection_summary_df=selection_summary_df,
            top_n=int(tsne_top_n_features),
        )
        scaler_kind = "standard"
        if isinstance(report, dict):
            scaler_kind = str(report.get("scaler", "standard"))
        tsne_meta = _plot_tsne_clusters_top_features(
            clustered_df=clustered_df,
            out_dir=out_dir,
            seed=seed,
            feature_cols_ranked=tsne_features,
            top_n_features=int(tsne_top_n_features),
            sample_size=int(tsne_sample_size),
            perplexity=float(tsne_perplexity),
            max_iter=int(tsne_max_iter),
            transform_meta=transform_meta,
            scaler_kind=scaler_kind,
        )
        if logger is not None:
            if tsne_meta is None:
                logger.info("t-SNE plot skipped (insufficient valid data or features).")
            else:
                logger.info(
                    "t-SNE plots (2D+3D) generated with features=%s, used_rows=%d, valid_rows=%d, perplexity=%.3f, max_iter=%d, n_components=%d",
                    tsne_meta["feature_cols"],
                    int(tsne_meta["n_used_rows"]),
                    int(tsne_meta["n_valid_rows"]),
                    float(tsne_meta["perplexity_effective"]),
                    int(tsne_meta["max_iter"]),
                    int(tsne_meta["n_components"]),
                )
    return summary_df


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    return obj


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for PCA + k-means clustering.

    Parameters
    ----------
    None
        This function only defines and returns a configured parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all clustering, preprocessing, and I/O options.

    Notes
    -----
    Defaults are aligned with repository conventions:
    `out_files/{dataset_name}/metaorders_info_sameday_filtered_{level}_{group}.parquet`.
    By default, `--group both` concatenates proprietary and client metaorders.

    Examples
    --------
    >>> parser = build_arg_parser()
    >>> isinstance(parser, argparse.ArgumentParser)
    True
    """
    parser = argparse.ArgumentParser(
        description=(
            "Cluster metaorders with PCA-guided feature selection + k-means in "
            "selected original-feature space, choose k via silhouette, and export artifacts."
        )
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=str(_REPO_ROOT / "config_ymls" / "metaorder_computation.yml"),
        help="YAML config used for defaults (dataset, level, output/image base paths).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name used for path templates. Defaults to YAML DATASET_NAME or ftsemib.",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        choices=["member", "client"],
        help="Metaorder aggregation level. Defaults to YAML LEVEL or member.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="both",
        choices=["both", "client", "proprietary"],
        help="Which metaorder populations to cluster. Default: both.",
    )
    parser.add_argument(
        "--output-file-path",
        type=str,
        default=None,
        help="Base output folder. Defaults to YAML OUTPUT_FILE_PATH or out_files/{DATASET_NAME}.",
    )
    parser.add_argument(
        "--img-output-path",
        type=str,
        default=None,
        help="Base folder for figure outputs. Defaults to YAML IMG_OUTPUT_PATH or images/{DATASET_NAME}.",
    )
    parser.add_argument(
        "--prop-path",
        type=str,
        default=None,
        help="Optional explicit path to proprietary metaorders file.",
    )
    parser.add_argument(
        "--client-path",
        type=str,
        default=None,
        help="Optional explicit path to client/non-proprietary metaorders file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for clustering artifacts.",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default=None,
        help="Directory where figures are saved. Default: images/{dataset_name}/{out_dir_name}.",
    )

    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Optional comma-separated feature override. If omitted, uses the default engineered set.",
    )
    parser.add_argument(
        "--pca-variance-threshold",
        type=float,
        default=0.75,
        help="Cumulative explained variance target for PCA n_components. Default: 0.75.",
    )
    parser.add_argument(
        "--pc-feature-contrib-threshold",
        type=float,
        default=0.50,
        help=(
            "Per-retained-PC cumulative contribution threshold (from squared loadings) "
            "used to select original features. Default: 0.50."
        ),
    )
    parser.add_argument("--k-min", type=int, default=2, help="Minimum k for silhouette sweep. Default: 2.")
    parser.add_argument("--k-max", type=int, default=20, help="Maximum k for silhouette sweep. Default: 20.")
    parser.add_argument(
        "--k-list",
        type=str,
        default=None,
        help=(
            "Optional explicit cluster counts for scenario plots, e.g. '3,4,5,6' or '[3,4,5,6]'. "
            "When provided, generates the full visualization set for each listed k under "
            "k-specific subfolders."
        ),
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=20000,
        help="Sample size used inside silhouette_score for speed. Default: 20000.",
    )
    parser.add_argument("--n-init", type=int, default=50, help="k-means n_init. Default: 50.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility. Default: 0.")
    parser.add_argument(
        "--scaler",
        type=str,
        default="standard",
        choices=["standard", "robust"],
        help="Feature scaler after transformation. Default: standard.",
    )
    parser.add_argument(
        "--clip-quantiles",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW_Q", "HIGH_Q"),
        help="Optional winsorization quantiles before transform, e.g. --clip-quantiles 0.001 0.999.",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="plotly",
        choices=["plotly", "none"],
        help="Plot output mode. Default: plotly.",
    )
    parser.add_argument(
        "--n-feature-triplets",
        type=int,
        default=4,
        help=(
            "Number of 3D feature-triplet scatter plots to produce from ranked selected features. "
            "Default: 4."
        ),
    )
    parser.add_argument(
        "--feature-triplet-sample-size",
        type=int,
        default=50000,
        help="Maximum points per 3D triplet scatter plot. Default: 50000.",
    )
    parser.add_argument(
        "--run-tsne",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate t-SNE cluster visualization from top selected features. Default: enabled.",
    )
    parser.add_argument(
        "--tsne-top-features",
        type=int,
        default=10,
        help="Number of top PCA-selected features to use for t-SNE. Default: 10.",
    )
    parser.add_argument(
        "--tsne-sample-size",
        type=int,
        default=30000,
        help="Maximum number of rows used for t-SNE embedding. Default: 30000.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Requested t-SNE perplexity (automatically clipped to valid range). Default: 30.",
    )
    parser.add_argument(
        "--tsne-max-iter",
        type=int,
        default=1000,
        help="Maximum t-SNE optimization iterations. Default: 1000.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and features, then stop without fitting.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help=(
            "Load existing clustering outputs and regenerate visualizations without recomputing "
            "PCA/k-means. It searches the resolved output directory and matching run folders."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional explicit path for the run log file. Default: <out_dir>/logs/run_<timestamp>.log",
    )
    return parser


def main() -> None:
    """
    Run the end-to-end metaorder clustering workflow.

    Parameters
    ----------
    None
        Inputs are provided through command-line arguments.

    Returns
    -------
    None
        Writes clustering artifacts (parquet/csv/json/plots) to disk.

    Notes
    -----
    Pipeline:
    1) load selected metaorders (`both`/`client`/`proprietary`);
    2) engineer unsigned impact features and durations;
    3) apply heavy-tail transform + scaling;
    4) PCA at target cumulative variance;
    5) from retained PCs, select original features using squared-loadings contribution;
    6) run silhouette sweep + final k-means in selected original-feature space.
    Reproducibility depends on `--seed`.

    Examples
    --------
    >>> # Minimal run on default dataset/paths
    >>> # python scripts/metaorder_clustering.py --group both --k-min 2 --k-max 12
    """
    args = build_arg_parser().parse_args()

    if not (0.0 < args.pca_variance_threshold <= 1.0):
        raise ValueError("--pca-variance-threshold must be in (0, 1].")
    if not (0.0 < args.pc_feature_contrib_threshold <= 1.0):
        raise ValueError("--pc-feature-contrib-threshold must be in (0, 1].")
    if args.n_init <= 0:
        raise ValueError("--n-init must be strictly positive.")
    if args.n_feature_triplets < 0:
        raise ValueError("--n-feature-triplets must be >= 0.")
    if args.feature_triplet_sample_size <= 0:
        raise ValueError("--feature-triplet-sample-size must be > 0.")
    if args.tsne_top_features <= 0:
        raise ValueError("--tsne-top-features must be > 0.")
    if args.tsne_sample_size <= 0:
        raise ValueError("--tsne-sample-size must be > 0.")
    if args.tsne_perplexity <= 0.0:
        raise ValueError("--tsne-perplexity must be > 0.")
    if args.tsne_max_iter <= 0:
        raise ValueError("--tsne-max-iter must be > 0.")
    explicit_k_list = _parse_k_list(args.k_list)
    if args.silhouette_sample_size <= 0:
        silhouette_sample_size: Optional[int] = None
    else:
        silhouette_sample_size = int(args.silhouette_sample_size)

    clip_quantiles: Optional[Tuple[float, float]]
    if args.clip_quantiles is None:
        clip_quantiles = None
    else:
        low_q, high_q = float(args.clip_quantiles[0]), float(args.clip_quantiles[1])
        if not (0.0 <= low_q < high_q <= 1.0):
            raise ValueError("--clip-quantiles must satisfy 0 <= LOW_Q < HIGH_Q <= 1.")
        clip_quantiles = (low_q, high_q)

    cfg = _load_config(Path(args.config_path))
    resolved = _resolve_io_paths(args, cfg)
    out_dir = Path(resolved["out_dir"])
    if args.load:
        out_dir = _discover_output_dir_for_load(args, resolved)
    img_dir = _resolve_effective_img_dir(args=args, resolved=resolved, effective_out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_file) if args.log_file else out_dir / "logs" / f"run_{timestamp}.log"
    logger = _setup_logger(log_path)
    logger.info("Starting metaorder clustering run.")
    logger.info("Arguments: %s", vars(args))
    logger.info("Resolved paths: %s", {k: str(v) for k, v in resolved.items()})
    logger.info("Effective output directory: %s", out_dir)
    logger.info("Effective image directory: %s", img_dir)

    if args.load:
        if args.dry_run:
            paths = _expected_output_paths(out_dir)
            logger.info("Dry run (--load) summary:")
            logger.info("Discovered output directory: %s", out_dir)
            logger.info("Image output directory: %s", img_dir)
            logger.info("explicit_k_list=%s", explicit_k_list)
            for name, path in paths.items():
                logger.info("%s: %s (exists=%s)", name, path, path.exists())
            return

        outputs = _load_existing_outputs(out_dir)
        clustered_df = outputs["clustered_df"]
        silhouette_curve = outputs["silhouette_curve"]
        summary_df = outputs["summary_df"]
        report = outputs["report"]
        pca_scores_df = outputs["pca_scores_df"]

        logger.info("Loaded existing outputs.")
        logger.info("Rows in clustered parquet: %d", len(clustered_df))
        logger.info("Silhouette curve rows: %d", len(silhouette_curve))

        if explicit_k_list:
            feature_cols_for_k = _resolve_clustering_feature_columns_from_report(clustered_df, report=report)
            scaler_kind = str(report.get("scaler", args.scaler)) if isinstance(report, dict) else str(args.scaler)
            transform_meta = report.get("transform_meta") if isinstance(report, dict) else None
            matrix_valid, valid_mask = _build_scaled_clustering_matrix(
                clustered_df,
                feature_cols=feature_cols_for_k,
                transform_meta=transform_meta,
                scaler_kind=scaler_kind,
            )
            valid_df = clustered_df.loc[valid_mask].reset_index(drop=True)
            if len(valid_df) != len(clustered_df):
                logger.info(
                    "Load mode with --k-list dropped %d rows with invalid transformed features.",
                    int(len(clustered_df) - len(valid_df)),
                )
            scenario_k_values = _validate_k_list_values(explicit_k_list, n_rows=matrix_valid.shape[0])
            logger.info("Generating explicit-k scenarios in load mode: %s", scenario_k_values)

            n_proj_components = min(10, matrix_valid.shape[1], matrix_valid.shape[0])
            projection_matrix = PCA(n_components=n_proj_components, svd_solver="full").fit_transform(matrix_valid)
            projection_cols = [f"PC{i + 1}" for i in range(projection_matrix.shape[1])]

            for k in scenario_k_values:
                labels_k = _fit_labels_for_k(matrix_valid, k=int(k), n_init=int(args.n_init), seed=int(args.seed))
                scenario_df = valid_df.copy()
                scenario_df["Cluster"] = labels_k

                scenario_out_dir, scenario_img_dir = _build_scenario_dirs(out_dir, img_dir, k=int(k))
                scenario_out_dir.mkdir(parents=True, exist_ok=True)
                scenario_img_dir.mkdir(parents=True, exist_ok=True)
                scenario_paths = _expected_output_paths(scenario_out_dir)

                scenario_df.to_parquet(scenario_paths["clustered"], index=False)
                scenario_summary = _summarize_clusters(scenario_df, feature_cols=feature_cols_for_k)
                scenario_summary.to_csv(scenario_paths["summary"], index=False)
                silhouette_curve.to_csv(scenario_paths["silhouette"], index=False)
                if outputs.get("selection_summary_df") is not None:
                    outputs["selection_summary_df"].to_csv(scenario_paths["selection_summary"], index=False)
                if outputs.get("selection_per_pc_df") is not None:
                    outputs["selection_per_pc_df"].to_csv(scenario_paths["selection_per_pc"], index=False)

                scenario_pca_scores = pd.DataFrame(projection_matrix, columns=projection_cols)
                scenario_pca_scores["Cluster"] = labels_k
                if "Group" in scenario_df.columns:
                    scenario_pca_scores["Group"] = scenario_df["Group"].to_numpy()
                scenario_pca_scores.to_parquet(scenario_paths["pca_scores"], index=False)

                scenario_report: Dict[str, Any] = dict(report) if isinstance(report, dict) else {}
                scenario_report["k_selected"] = int(k)
                scenario_report["k_source"] = "explicit_cli_list"
                scenario_report["k_list"] = [int(v) for v in scenario_k_values]
                scenario_report["rows_after_preprocessing"] = int(len(scenario_df))
                scenario_report["clustering_feature_columns"] = list(feature_cols_for_k)
                scenario_report["n_init"] = int(args.n_init)
                scenario_report["seed"] = int(args.seed)
                scenario_report["output_paths"] = {
                    "clustered": str(scenario_paths["clustered"]),
                    "silhouette": str(scenario_paths["silhouette"]),
                    "summary": str(scenario_paths["summary"]),
                    "pca_scores": str(scenario_paths["pca_scores"]),
                    "selection_summary": str(scenario_paths["selection_summary"]),
                    "selection_per_pc": str(scenario_paths["selection_per_pc"]),
                }
                scenario_report["image_dir"] = str(scenario_img_dir)
                scenario_paths["report"].write_text(
                    json.dumps(_to_jsonable(scenario_report), indent=2),
                    encoding="utf-8",
                )

                if args.plots == "plotly":
                    _render_result_visualizations(
                        out_dir=scenario_img_dir,
                        clustered_df=scenario_df,
                        silhouette_curve=silhouette_curve,
                        report=scenario_report,
                        summary_df=scenario_summary,
                        selection_summary_df=outputs.get("selection_summary_df"),
                        selection_per_pc_df=outputs.get("selection_per_pc_df"),
                        pca_scores_df=scenario_pca_scores,
                        seed=int(args.seed),
                        n_feature_triplets=int(args.n_feature_triplets),
                        feature_triplet_sample_size=int(args.feature_triplet_sample_size),
                        run_tsne=bool(args.run_tsne),
                        tsne_top_n_features=int(args.tsne_top_features),
                        tsne_sample_size=int(args.tsne_sample_size),
                        tsne_perplexity=float(args.tsne_perplexity),
                        tsne_max_iter=int(args.tsne_max_iter),
                        logger=logger,
                        highlight_k=int(k),
                    )
                logger.info(
                    "Scenario k=%d completed in load mode. out_dir=%s | img_dir=%s",
                    int(k),
                    scenario_out_dir,
                    scenario_img_dir,
                )

            logger.info("Load mode completed (no recomputation) for explicit-k scenarios.")
            logger.info("Run log written to: %s", log_path)
            return

        if args.plots == "plotly":
            summary_df = _render_result_visualizations(
                out_dir=img_dir,
                clustered_df=clustered_df,
                silhouette_curve=silhouette_curve,
                report=report,
                summary_df=summary_df,
                selection_summary_df=outputs.get("selection_summary_df"),
                selection_per_pc_df=outputs.get("selection_per_pc_df"),
                pca_scores_df=pca_scores_df,
                seed=int(args.seed),
                n_feature_triplets=int(args.n_feature_triplets),
                feature_triplet_sample_size=int(args.feature_triplet_sample_size),
                run_tsne=bool(args.run_tsne),
                tsne_top_n_features=int(args.tsne_top_features),
                tsne_sample_size=int(args.tsne_sample_size),
                tsne_perplexity=float(args.tsne_perplexity),
                tsne_max_iter=int(args.tsne_max_iter),
                logger=logger,
            )
            logger.info("Plots regenerated from saved artifacts.")

        best_k = _choose_k(silhouette_curve)
        logger.info("Load mode completed (no recomputation).")
        logger.info("Selected k from silhouette CSV: %d", best_k)
        logger.info("Summary rows: %d", len(summary_df) if summary_df is not None else 0)
        logger.info("Run log written to: %s", log_path)
        return

    df = _load_metaorders_by_group(args, resolved)
    n_loaded = len(df)
    logger.info("Loaded metaorders rows: %d", n_loaded)
    if "Group" in df.columns:
        logger.info("Rows by group: %s", df["Group"].value_counts(dropna=False).to_dict())

    df, abs_impact_cols = _add_engineered_features(df)
    logger.info("Engineered features added: DurationSeconds and %d absolute impact columns.", len(abs_impact_cols))

    feature_override = _parse_feature_override(args.features)
    if feature_override is None:
        feature_cols = [c for c in _default_feature_columns(abs_impact_cols) if c in df.columns]
    else:
        feature_cols = feature_override
    logger.info("Candidate feature columns (%d): %s", len(feature_cols), feature_cols)

    if args.dry_run:
        logger.info("Dry run summary:")
        logger.info("dataset_name=%s", resolved["dataset_name"])
        logger.info("level=%s", resolved["level"])
        logger.info("group=%s", args.group)
        logger.info("rows_loaded=%d", n_loaded)
        logger.info("prop_path=%s", resolved["prop_path"])
        logger.info("client_path=%s", resolved["client_path"])
        logger.info("out_dir=%s", out_dir)
        logger.info("img_dir=%s", img_dir)
        logger.info("pca_variance_threshold=%.4f", float(args.pca_variance_threshold))
        logger.info("pc_feature_contrib_threshold=%.4f", float(args.pc_feature_contrib_threshold))
        logger.info("k_grid=[%d, %d]", int(args.k_min), int(args.k_max))
        logger.info("explicit_k_list=%s", explicit_k_list)
        logger.info(
            "feature_triplet_plots=%d (sample_size=%d)",
            int(args.n_feature_triplets),
            int(args.feature_triplet_sample_size),
        )
        logger.info(
            "run_tsne=%s | tsne_top_features=%d | tsne_sample_size=%d | tsne_perplexity=%.3f | tsne_max_iter=%d",
            bool(args.run_tsne),
            int(args.tsne_top_features),
            int(args.tsne_sample_size),
            float(args.tsne_perplexity),
            int(args.tsne_max_iter),
        )
        logger.info("Run log written to: %s", log_path)
        return

    prepared_df, matrix, transform_meta, transformed_cols = _prepare_matrix(
        df=df,
        feature_cols=feature_cols,
        clip_quantiles=clip_quantiles,
    )
    logger.info("Rows after preprocessing: %d", len(prepared_df))
    matrix_scaled, _ = _scale_matrix(matrix, scaler_kind=args.scaler)
    logger.info("Applied scaler: %s", args.scaler)

    pca = PCA(n_components=float(args.pca_variance_threshold), svd_solver="full")
    matrix_pca = pca.fit_transform(matrix_scaled)
    logger.info(
        "Retained %d PCs at variance threshold %.4f (cum=%.4f).",
        int(pca.n_components_),
        float(args.pca_variance_threshold),
        float(np.cumsum(pca.explained_variance_ratio_)[-1]),
    )

    selected_idx, selected_features, selection_summary_df, selection_per_pc_df = _select_features_from_pca(
        pca=pca,
        feature_cols=feature_cols,
        per_pc_contribution_threshold=float(args.pc_feature_contrib_threshold),
    )
    logger.info(
        "Selected %d original features for clustering (per-PC contribution threshold %.4f).",
        len(selected_features),
        float(args.pc_feature_contrib_threshold),
    )
    logger.info("Selected features: %s", selected_features)
    if not selection_per_pc_df.empty and "pc" in selection_per_pc_df.columns:
        for pc in sorted(selection_per_pc_df["pc"].unique().tolist()):
            pc_rows = selection_per_pc_df[
                (selection_per_pc_df["pc"] == pc) & (selection_per_pc_df["selected_for_pc"])
            ]
            pc_feats = pc_rows["feature"].tolist()
            pc_evr = float(pc_rows["pc_explained_variance_ratio"].iloc[0]) if not pc_rows.empty else float("nan")
            logger.info("PC%d (EVR=%.4f) selected features: %s", int(pc), pc_evr, pc_feats)

    cluster_matrix = matrix_scaled[:, selected_idx]
    transformed_selected_cols = [transformed_cols[i] for i in selected_idx]
    logger.info("Clustering matrix shape: %s", tuple(cluster_matrix.shape))
    logger.info("Selected transformed columns: %s", transformed_selected_cols)

    k_values = _validate_k_grid(k_min=args.k_min, k_max=args.k_max, n_rows=cluster_matrix.shape[0])
    silhouette_curve = _evaluate_k_values(
        matrix_pca=cluster_matrix,
        k_values=k_values,
        n_init=int(args.n_init),
        seed=int(args.seed),
        silhouette_sample_size=silhouette_sample_size,
    )
    logger.info("Silhouette scores by k: %s", silhouette_curve.to_dict(orient="records"))
    best_k = _choose_k(silhouette_curve)
    logger.info("Selected k (silhouette argmax): %d", best_k)
    scenario_k_values: List[int] = []
    if explicit_k_list:
        scenario_k_values = _validate_k_list_values(explicit_k_list, n_rows=cluster_matrix.shape[0])
        logger.info("Generating explicit-k scenarios in compute mode: %s", scenario_k_values)

    final_model = KMeans(n_clusters=best_k, n_init=int(args.n_init), random_state=int(args.seed))
    labels = final_model.fit_predict(cluster_matrix)
    prepared_df = prepared_df.copy()
    prepared_df["Cluster"] = labels

    paths = _expected_output_paths(out_dir)
    clustered_path = paths["clustered"]
    silhouette_csv_path = paths["silhouette"]
    summary_csv_path = paths["summary"]
    report_json_path = paths["report"]
    pca_scores_path = paths["pca_scores"]
    selection_summary_path = paths["selection_summary"]
    selection_per_pc_path = paths["selection_per_pc"]

    prepared_df.to_parquet(clustered_path, index=False)
    silhouette_curve.to_csv(silhouette_csv_path, index=False)
    summary = _summarize_clusters(prepared_df, feature_cols=feature_cols)
    summary.to_csv(summary_csv_path, index=False)
    selection_summary_df.to_csv(selection_summary_path, index=False)
    selection_per_pc_df.to_csv(selection_per_pc_path, index=False)

    n_proj_components = min(10, cluster_matrix.shape[1], cluster_matrix.shape[0])
    projection_matrix = PCA(n_components=n_proj_components, svd_solver="full").fit_transform(cluster_matrix)
    pc_cols = [f"PC{i + 1}" for i in range(projection_matrix.shape[1])]
    pca_scores_df = pd.DataFrame(projection_matrix, columns=pc_cols)
    pca_scores_df["Cluster"] = labels
    if "Group" in prepared_df.columns:
        pca_scores_df["Group"] = prepared_df["Group"].to_numpy()
    pca_scores_df.to_parquet(pca_scores_path, index=False)

    report = {
        "dataset_name": resolved["dataset_name"],
        "level": resolved["level"],
        "group": args.group,
        "rows_loaded": int(n_loaded),
        "rows_after_preprocessing": int(len(prepared_df)),
        "feature_columns": list(feature_cols),
        "clustering_feature_columns": list(selected_features),
        "clustering_feature_indices": [int(i) for i in selected_idx],
        "clustering_space": "selected_original_features_scaled",
        "clustering_projection_space": "pca_of_selected_original_features_scaled",
        "pc_feature_contrib_threshold": float(args.pc_feature_contrib_threshold),
        "transform_meta": transform_meta,
        "scaler": args.scaler,
        "clip_quantiles": list(clip_quantiles) if clip_quantiles is not None else None,
        "pca_variance_threshold": float(args.pca_variance_threshold),
        "pca_n_components": int(pca.n_components_),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "pca_cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        "pca_components": pca.components_,
        "k_values": list(k_values),
        "k_selected": int(best_k),
        "silhouette_sample_size": silhouette_sample_size,
        "n_init": int(args.n_init),
        "seed": int(args.seed),
        "n_feature_triplets": int(args.n_feature_triplets),
        "feature_triplet_sample_size": int(args.feature_triplet_sample_size),
        "run_tsne": bool(args.run_tsne),
        "tsne_top_features": int(args.tsne_top_features),
        "tsne_sample_size": int(args.tsne_sample_size),
        "tsne_perplexity": float(args.tsne_perplexity),
        "tsne_max_iter": int(args.tsne_max_iter),
        "input_paths": {
            "proprietary": str(resolved["prop_path"]),
            "client": str(resolved["client_path"]),
        },
        "output_paths": {
            "clustered": str(clustered_path),
            "silhouette": str(silhouette_csv_path),
            "summary": str(summary_csv_path),
            "pca_scores": str(pca_scores_path),
            "selection_summary": str(selection_summary_path),
            "selection_per_pc": str(selection_per_pc_path),
        },
        "image_dir": str(img_dir),
    }
    report_json_path.write_text(json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")

    if scenario_k_values:
        prepared_base = prepared_df.drop(columns=["Cluster"]).reset_index(drop=True)
        for k in scenario_k_values:
            labels_k = _fit_labels_for_k(cluster_matrix, k=int(k), n_init=int(args.n_init), seed=int(args.seed))
            scenario_df = prepared_base.copy()
            scenario_df["Cluster"] = labels_k

            scenario_out_dir, scenario_img_dir = _build_scenario_dirs(out_dir, img_dir, k=int(k))
            scenario_out_dir.mkdir(parents=True, exist_ok=True)
            scenario_img_dir.mkdir(parents=True, exist_ok=True)
            scenario_paths = _expected_output_paths(scenario_out_dir)

            scenario_df.to_parquet(scenario_paths["clustered"], index=False)
            scenario_summary = _summarize_clusters(scenario_df, feature_cols=feature_cols)
            scenario_summary.to_csv(scenario_paths["summary"], index=False)
            silhouette_curve.to_csv(scenario_paths["silhouette"], index=False)
            selection_summary_df.to_csv(scenario_paths["selection_summary"], index=False)
            selection_per_pc_df.to_csv(scenario_paths["selection_per_pc"], index=False)

            scenario_pca_scores = pd.DataFrame(projection_matrix, columns=pc_cols)
            scenario_pca_scores["Cluster"] = labels_k
            if "Group" in scenario_df.columns:
                scenario_pca_scores["Group"] = scenario_df["Group"].to_numpy()
            scenario_pca_scores.to_parquet(scenario_paths["pca_scores"], index=False)

            scenario_report = dict(report)
            scenario_report["k_selected"] = int(k)
            scenario_report["k_source"] = "explicit_cli_list"
            scenario_report["k_list"] = [int(v) for v in scenario_k_values]
            scenario_report["rows_after_preprocessing"] = int(len(scenario_df))
            scenario_report["output_paths"] = {
                "clustered": str(scenario_paths["clustered"]),
                "silhouette": str(scenario_paths["silhouette"]),
                "summary": str(scenario_paths["summary"]),
                "pca_scores": str(scenario_paths["pca_scores"]),
                "selection_summary": str(scenario_paths["selection_summary"]),
                "selection_per_pc": str(scenario_paths["selection_per_pc"]),
            }
            scenario_report["image_dir"] = str(scenario_img_dir)
            scenario_paths["report"].write_text(
                json.dumps(_to_jsonable(scenario_report), indent=2),
                encoding="utf-8",
            )

            if args.plots == "plotly":
                _render_result_visualizations(
                    out_dir=scenario_img_dir,
                    clustered_df=scenario_df,
                    silhouette_curve=silhouette_curve,
                    report=scenario_report,
                    summary_df=scenario_summary,
                    selection_summary_df=selection_summary_df,
                    selection_per_pc_df=selection_per_pc_df,
                    pca_scores_df=scenario_pca_scores,
                    seed=int(args.seed),
                    n_feature_triplets=int(args.n_feature_triplets),
                    feature_triplet_sample_size=int(args.feature_triplet_sample_size),
                    run_tsne=bool(args.run_tsne),
                    tsne_top_n_features=int(args.tsne_top_features),
                    tsne_sample_size=int(args.tsne_sample_size),
                    tsne_perplexity=float(args.tsne_perplexity),
                    tsne_max_iter=int(args.tsne_max_iter),
                    logger=logger,
                    highlight_k=int(k),
                )

            logger.info(
                "Scenario k=%d completed in compute mode. out_dir=%s | img_dir=%s",
                int(k),
                scenario_out_dir,
                scenario_img_dir,
            )

    if args.plots == "plotly":
        _render_result_visualizations(
            out_dir=img_dir,
            clustered_df=prepared_df,
            silhouette_curve=silhouette_curve,
            report=report,
            summary_df=summary,
            selection_summary_df=selection_summary_df,
            selection_per_pc_df=selection_per_pc_df,
            pca_scores_df=pca_scores_df,
            seed=int(args.seed),
            n_feature_triplets=int(args.n_feature_triplets),
            feature_triplet_sample_size=int(args.feature_triplet_sample_size),
            run_tsne=bool(args.run_tsne),
            tsne_top_n_features=int(args.tsne_top_features),
            tsne_sample_size=int(args.tsne_sample_size),
            tsne_perplexity=float(args.tsne_perplexity),
            tsne_max_iter=int(args.tsne_max_iter),
            logger=logger,
        )
        logger.info("Plots written to image directory.")

    logger.info("Clustering completed.")
    logger.info("Rows loaded: %d", n_loaded)
    logger.info("Rows after preprocessing: %d", len(prepared_df))
    logger.info("Candidate feature columns (%d): %s", len(feature_cols), feature_cols)
    logger.info("Clustering feature columns (%d): %s", len(selected_features), selected_features)
    logger.info("PCA components selected: %d", int(pca.n_components_))
    logger.info("Selected k (silhouette argmax): %d", best_k)
    logger.info("Clustered parquet: %s", clustered_path)
    logger.info("Silhouette curve CSV: %s", silhouette_csv_path)
    logger.info("Cluster summary CSV: %s", summary_csv_path)
    logger.info("PCA report JSON: %s", report_json_path)
    logger.info("PCA scores parquet: %s", pca_scores_path)
    logger.info("Feature selection summary CSV: %s", selection_summary_path)
    logger.info("Feature selection per-PC CSV: %s", selection_per_pc_path)
    logger.info(
        "Feature-triplet plots requested: %d (sample_size=%d)",
        int(args.n_feature_triplets),
        int(args.feature_triplet_sample_size),
    )
    logger.info(
        "t-SNE requested: %s (top_features=%d, sample_size=%d, perplexity=%.3f, max_iter=%d)",
        bool(args.run_tsne),
        int(args.tsne_top_features),
        int(args.tsne_sample_size),
        float(args.tsne_perplexity),
        int(args.tsne_max_iter),
    )
    logger.info("Explicit k-list requested: %s", scenario_k_values if scenario_k_values else None)
    logger.info("Image directory: %s", img_dir)
    logger.info("Run log written to: %s", log_path)


if __name__ == "__main__":
    main()
