from __future__ import annotations

import json
from pathlib import Path

from scripts import check_member_label_anonymization as checker


def test_checker_redacts_detected_raw_id(tmp_path: Path, capsys) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text("Visible member 96862", encoding="utf-8")
    map_path = tmp_path / "member_label_map.json"
    map_path.write_text(json.dumps({"96862": "Member 30"}), encoding="utf-8")

    result = checker.main(["--map-path", str(map_path), "--paper-dir", str(paper_dir)])

    output = capsys.readouterr().out
    assert result == 1
    assert "[REDACTED]" in output
    assert "96862" not in output


def test_checker_requires_local_crosswalk(tmp_path: Path, capsys) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()

    result = checker.main(
        [
            "--map-path",
            str(tmp_path / "missing.json"),
            "--paper-dir",
            str(paper_dir),
        ]
    )

    assert result == 2
    assert "Confidential member-label map not found" in capsys.readouterr().out
