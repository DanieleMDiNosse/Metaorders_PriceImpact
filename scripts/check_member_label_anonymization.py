from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "config_ymls" / "member_label_map.json"
CHECK_PATHS = [ROOT / "paper" / "main.tex"]


def main() -> int:
    mapping = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    raw_ids = [re.escape(str(key)) for key in mapping]
    if not raw_ids:
        print("No member IDs configured.")
        return 0
    pattern = re.compile(r"\b(" + "|".join(raw_ids) + r")\b")
    failures: list[str] = []
    for path in CHECK_PATHS:
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            failures.append(f"{path.relative_to(ROOT)}:{line_no}: raw member ID {match.group(0)}")
    if failures:
        print("\n".join(failures))
        return 1
    print("No raw mapped member IDs found in paper-visible text checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
