from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from moimpact.member_labels import format_member_label, label_member_series, load_member_label_map


def test_load_member_label_map_normalizes_keys(tmp_path: Path) -> None:
    path = tmp_path / "member_label_map.json"
    path.write_text(json.dumps({"96862": "Member 30", "91149": "Member 11"}))

    mapping = load_member_label_map(path)

    assert mapping["96862"] == "Member 30"
    assert mapping["91149"] == "Member 11"


def test_format_member_label_handles_numeric_and_string_ids(tmp_path: Path) -> None:
    path = tmp_path / "member_label_map.json"
    path.write_text(json.dumps({"96862": "Member 30"}))
    mapping = load_member_label_map(path)

    assert format_member_label(96862, mapping) == "Member 30"
    assert format_member_label("96862", mapping) == "Member 30"
    assert format_member_label(96862.0, mapping) == "Member 30"


def test_format_member_label_defaults_to_non_leaking_fallback() -> None:
    mapping = {"96862": "Member 30"}

    assert format_member_label("99999", mapping) == "Member ?"
    assert format_member_label("99999", mapping, fallback="original") == "99999"
    assert format_member_label("99999", mapping, fallback="unknown") == "Member ?"


def test_label_member_series_preserves_index() -> None:
    series = pd.Series([96862, "91149"], index=["a", "b"])
    out = label_member_series(series, {"96862": "Member 30", "91149": "Member 11"})

    assert list(out.index) == ["a", "b"]
    assert list(out) == ["Member 30", "Member 11"]
