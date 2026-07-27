from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, Mapping

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMBER_LABEL_MAP_PATH = _REPO_ROOT / "config_ymls" / "member_label_map.json"

FallbackMode = Literal["original", "unknown"]


def _normalize_member_key(value: object) -> str:
    """Normalize numeric member identifiers for stable lookup."""
    if value is None:
        return ""
    try:
        is_missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        is_missing = False
    if is_missing:
        return ""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0"):
        head = text[:-2]
        if head.isdigit():
            return head
    return text


def load_member_label_map(path: str | Path | None = None) -> dict[str, str]:
    """Load the raw-member-ID to paper-display-label mapping."""
    map_path = Path(path) if path is not None else DEFAULT_MEMBER_LABEL_MAP_PATH
    if not map_path.exists():
        return {}
    with map_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Member label map must be a JSON object: {map_path}")
    return {_normalize_member_key(key): str(value) for key, value in raw.items()}


def format_member_label(
    value: object,
    mapping: Mapping[str, str] | None = None,
    *,
    fallback: FallbackMode = "original",
) -> str:
    """Return the anonymized display label for one member identifier."""
    labels = mapping if mapping is not None else load_member_label_map()
    key = _normalize_member_key(value)
    if key in labels:
        return labels[key]
    if fallback == "unknown":
        return "Member ?"
    return key


def label_member_series(
    series: pd.Series,
    mapping: Mapping[str, str] | None = None,
    *,
    fallback: FallbackMode = "original",
) -> pd.Series:
    """Map a Series of raw member identifiers to anonymized display labels."""
    labels = mapping if mapping is not None else load_member_label_map()
    return series.map(lambda value: format_member_label(value, labels, fallback=fallback))
