#!/usr/bin/env python3
"""
Compute member-level trading statistics across ISINs and generate Plotly outputs.

This script expects a folder of per-ISIN parquet files and produces a small set of
summary tables/figures describing:
- how many unique members trade each ISIN,
- how proprietary vs client activity varies by ISIN,
- how broadly each member trades across the available ISIN universe, and
- a day-by-day activity heatmap across all ISINs (with an interactive per-ISIN
  breakdown on hover).

Inputs
------
- `DATA_DIR` (`<repo>/data/parquet` by default) must contain one or more `*.parquet` files.
  Each file is treated as one ISIN; the ISIN is inferred from the filename stem
  (e.g., `FR0012345678.parquet` -> ISIN `FR0012345678`).
- Required parquet columns (read via pandas):
  - `"ID Member"`: member identifier (any dtype; converted to string for display).
  - `"Trade Time"`: trade timestamp (must be datetime-like; the script uses `.dt`).
  - `"Trade Type Aggressive"`: used to classify proprietary vs client trades.
  - `"Direction"`: used to split buy vs sell trades in the hover bar chart
    (expected sign: buy > 0, sell < 0). If missing, the script attempts to derive
    it from `"Total Quantity Buy"` / `"Total Quantity Sell"` when available.
- Optional parquet columns:
  - `"Aggressive Member Nationality"`: if present, the member coverage plot is
    colored by member nationality (e.g., `IT` vs `FOREIGN`). Members with missing
    or inconsistent labels are shown as `UNKNOWN` / `MIXED`.

Processing details
------------------
- Trades are filtered to `TRADING_HOURS` (default 09:30-17:30) using only the
  time-of-day portion of `"Trade Time"`.
- Proprietary trades are those where `"Trade Type Aggressive" == "Dealing_on_own_account"`.
  All other values are treated as client trades.
- Member coverage is computed as: (# ISINs traded by member) / (total # ISIN files).

Outputs
-------
Creates `images/ftsemib/member_statistics/` (and subfolders) containing:
- Interactive HTML plots (`images/ftsemib/member_statistics/html/*.html`)
- Static PNG exports (`images/ftsemib/member_statistics/png/*.png`), when Plotly image
  export is available (typically via `kaleido`). If PNG export fails, the script
  logs a warning and still writes the HTML files.

Generated figures (file stems):
1) `members_per_isin` (bar)
2) `proprietary_vs_client_trades_per_isin` (stacked bar)
3) `member_isin_coverage_per_member` (bar)
4) `member_activity_heatmap` (HTML) + `member_activity_heatmap.png` (PNG)

The heatmap HTML embeds a second bar chart that updates on hover to show the
per-ISIN buy/sell breakdown for the selected (day, member) cell (buys positive,
sells negative).

Logging
-------
Appends progress and warnings to `member_statistics.log` next to this script.

Usage
-----
Run from the repository root:

    python scripts/member_statistics.py

To change the data location or trading-hours filter, edit `DATA_DIR` and
`TRADING_HOURS` in the "Config" section near the top of the file.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Ensure repository-root imports (e.g., `moimpact`) work when running
# `python scripts/member_statistics.py` from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.plot_style import (
    THEME_BG_COLOR,
    THEME_COLORWAY,
    THEME_FONT_FAMILY,
    THEME_GRID_COLOR,
    apply_plotly_style,
)
from moimpact.plotting import (
    COLOR_CLIENT,
    COLOR_PROPRIETARY,
    PlotOutputDirs,
    ensure_plot_dirs,
    make_plot_output_dirs,
    save_plotly_figure as _save_plotly_figure,
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
# Per-ISIN parquet trade tapes live under `data/parquet/` in this repo.
DATA_DIR = _REPO_ROOT / "data" / "parquet"
PLOT_DIR = _REPO_ROOT / "images" / "ftsemib" / "member_statistics"
PLOT_OUTPUT_DIRS: PlotOutputDirs = make_plot_output_dirs(PLOT_DIR, use_subdirs=True)
HTML_DIR = PLOT_OUTPUT_DIRS.html_dir
PNG_DIR = PLOT_OUTPUT_DIRS.png_dir
TRADING_HOURS = ("09:30:00", "17:30:00")
AGGRESSIVE_MEMBER_NATIONALITY_COL = "Aggressive Member Nationality"
TICK_FONT_SIZE = 12
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 15
LEGEND_FONT_SIZE = 12
ensure_plot_dirs(PLOT_OUTPUT_DIRS)

apply_plotly_style(
    tick_font_size=TICK_FONT_SIZE,
    label_font_size=LABEL_FONT_SIZE,
    title_font_size=TITLE_FONT_SIZE,
    legend_font_size=LEGEND_FONT_SIZE,
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
    Member statistics figures are exported without top titles.
    """
    fig.update_layout(title=None)
    return _save_plotly_figure(fig, *args, **kwargs)

# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------
LOG_PATH = _REPO_ROOT / "member_statistics.log"
logger = logging.getLogger(Path(__file__).stem)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _formatter = logging.Formatter("%(asctime)s - %(message)s")
    _file_handler = logging.FileHandler(LOG_PATH, mode="a")
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)

_print = print


def log_print(*args, **kwargs):
    """
    Summary
    -------
    Print a message to stdout and append it to the script log.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to `print(...)`.

    Returns
    -------
    None

    Notes
    -----
    This helper keeps progress information both visible (stdout) and persisted
    (via `member_statistics.log`) for reproducibility/debugging.
    """
    message = " ".join(str(a) for a in args)
    logger.info(message)
    _print(*args, **kwargs)


def save_plotly(fig, stem: str):
    """
    Summary
    -------
    Save a Plotly figure to both HTML and PNG formats.

    Parameters
    ----------
    fig
        Plotly figure instance to export.
    stem
        Output filename stem (no extension).

    Returns
    -------
    None

    Notes
    -----
    PNG export requires Plotly's image export backend (typically `kaleido`).
    If PNG export fails, the HTML is still written.
    """
    html_path, png_path = save_plotly_figure(
        fig,
        stem=stem,
        dirs=PLOT_OUTPUT_DIRS,
        write_html=True,
        write_png=True,
        strict_png=False,
    )
    if png_path is None:
        log_print(f"[warn] Could not save PNG for {stem} (Plotly static export unavailable).")
    else:
        log_print(f"Saved PNG to {png_path}")
    if html_path is not None:
        log_print(f"Saved HTML to {html_path}")


# ---------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------
def normalize_member_nationality(value: object) -> str | None:
    """
    Summary
    -------
    Normalize member nationality labels to a canonical uppercase tag.

    Parameters
    ----------
    value
        Raw value from the `Aggressive Member Nationality` column.

    Returns
    -------
    str | None
        Canonical nationality tag:
        - "IT" for Italian members (when recognizable)
        - "FOREIGN" for non-Italian members (when recognizable)
        - otherwise, the uppercased stripped input string
        - `None` when missing/blank

    Notes
    -----
    The repository uses `"IT"` / `"FOREIGN"` in `data/members_nationality.parquet`
    and in the enriched per-ISIN parquet tapes. This helper is conservative: it
    standardizes common variants but does not try to infer arbitrary country codes.

    Examples
    --------
    >>> normalize_member_nationality(" IT ")
    'IT'
    >>> normalize_member_nationality("foreign")
    'FOREIGN'
    >>> normalize_member_nationality(None) is None
    True
    """
    if value is None or pd.isna(value):
        return None
    label = str(value).strip()
    if not label:
        return None
    upper = label.upper()
    if upper in {"IT", "ITALY", "ITALIAN"}:
        return "IT"
    if upper in {"FOREIGN", "FOR", "STRANIERO", "ESTERO", "NON_IT", "NON-IT", "NON IT", "ST"}:
        return "FOREIGN"
    return upper


def load_parquet_paths(data_dir: Path) -> List[Path]:
    """
    Summary
    -------
    List per-ISIN parquet files under a data directory.

    Parameters
    ----------
    data_dir
        Directory containing one parquet per ISIN.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of `*.parquet` files.

    Notes
    -----
    The script infers ISIN codes from `Path.stem`.
    """
    paths = sorted(p for p in data_dir.glob("*.parquet") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return paths


def main():
    """
    Summary
    -------
    Compute member/ISIN summary statistics and write figures to disk.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    The script reads all `*.parquet` files under `DATA_DIR`, treating each file
    as one ISIN tape, and writes Plotly figures to `PLOT_DIR`.
    """
    paths = load_parquet_paths(DATA_DIR)

    total_members: set = set()
    member_to_isins: Dict[int, set] = defaultdict(set)
    member_to_nationalities: Dict[object, Counter[str]] = defaultdict(Counter)
    members_per_isin: Dict[str, int] = {}
    trade_records: List[Dict[str, int]] = []
    activity_parts: List[pd.Series] = []
    # date -> member -> isin -> {"buy": n_buy, "sell": n_sell}
    per_day_member_isin: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(lambda: defaultdict(dict))

    for path in paths:
        isin = path.stem
        try:
            df = pd.read_parquet(
                path,
                columns=[
                    "ID Member",
                    "Trade Time",
                    "Trade Type Aggressive",
                    "Direction",
                    AGGRESSIVE_MEMBER_NATIONALITY_COL,
                ],
            )
        except Exception:
            # Backwards/heterogeneous datasets: keep a fast-path that avoids a full read when possible.
            try:
                df = pd.read_parquet(
                    path,
                    columns=["ID Member", "Trade Time", "Trade Type Aggressive", "Direction"],
                )
            except Exception:
                # Last resort: full read (needed when we must derive Direction).
                df = pd.read_parquet(path)
        start, end = TRADING_HOURS
        # Filter between 9:30 and 17:30
        df = df[
            (df["Trade Time"].dt.time >= pd.to_datetime(start).time())
            & (df["Trade Time"].dt.time <= pd.to_datetime(end).time())
        ].copy()

        if "Direction" not in df.columns and {"Total Quantity Buy", "Total Quantity Sell"}.issubset(df.columns):
            buy_qty = pd.to_numeric(df["Total Quantity Buy"], errors="coerce").fillna(0)
            sell_qty = pd.to_numeric(df["Total Quantity Sell"], errors="coerce").fillna(0)
            df["Direction"] = 0
            df.loc[buy_qty > 0, "Direction"] = 1
            df.loc[sell_qty > 0, "Direction"] = -1

        required = {"ID Member", "Trade Time", "Trade Type Aggressive", "Direction"}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"Missing columns in {path}: {missing}")

        if AGGRESSIVE_MEMBER_NATIONALITY_COL in df.columns:
            # Avoid per-row Python calls: normalize only the (few) distinct labels.
            raw_nat = df[AGGRESSIVE_MEMBER_NATIONALITY_COL]
            unique_raw = pd.unique(raw_nat.dropna())
            norm_map = {val: normalize_member_nationality(val) for val in unique_raw}
            nat = raw_nat.map(norm_map)
            nat_counts = (
                pd.DataFrame({"member": df["ID Member"], "nationality": nat})
                .dropna(subset=["nationality"])
                .groupby(["member", "nationality"])
                .size()
            )
            for (member, nationality), n in nat_counts.items():
                member_to_nationalities[member][str(nationality)] += int(n)

        members = pd.unique(df["ID Member"])
        members_per_isin[isin] = len(members)
        total_members.update(members)
        for m in members:
            member_to_isins[m].add(isin)

        # Proprietary vs client trade counts
        trade_type = df["Trade Type Aggressive"].astype(str)
        prop_mask = trade_type.eq("Dealing_on_own_account")
        prop_trades = int(prop_mask.sum())
        client_trades = int((~prop_mask).sum())
        trade_records.append({
            "isin": isin,
            "proprietary_trades": prop_trades,
            "client_trades": client_trades,
            "total_trades": prop_trades + client_trades,
        })

        # Activity per (date, member)
        df["date"] = df["Trade Time"].dt.floor("D")
        grouped = df.groupby(["date", "ID Member"]).size()
        activity_parts.append(grouped.rename("trades"))

        # Per-ISIN breakdown for hover
        direction_num = pd.to_numeric(df["Direction"], errors="coerce")
        side = pd.Series(pd.NA, index=df.index, dtype="string")
        side[direction_num > 0] = "buy"
        side[direction_num < 0] = "sell"
        if side.isna().any():
            direction_str = df["Direction"].astype(str).str.strip().str.lower()
            side[side.isna() & direction_str.isin({"buy", "b"})] = "buy"
            side[side.isna() & direction_str.isin({"sell", "s"})] = "sell"
        df["side"] = side.fillna("unknown")
        unknown_side = int((df["side"] == "unknown").sum())
        if unknown_side:
            log_print(f"[warn] {isin}: {unknown_side} trades have unknown Direction; excluded from buy/sell breakdown.")

        grouped_side = df[df["side"].isin({"buy", "sell"})].groupby(["date", "ID Member", "side"]).size()
        for (date, member, trade_side), trades in grouped_side.items():
            date_key = pd.to_datetime(date).strftime("%Y-%m-%d")
            member_key = str(member)
            isin_entry = per_day_member_isin[date_key][member_key].setdefault(isin, {"buy": 0, "sell": 0})
            isin_entry[str(trade_side)] += int(trades)

    # ------------------------------------------------------------------
    # Total members
    # ------------------------------------------------------------------
    log_print(f"Total unique members across all ISINs: {len(total_members)}")
    total_isins = len(paths)
    all_isins_sorted = sorted(members_per_isin.keys())

    # ------------------------------------------------------------------
    # Members per ISIN plot
    # ------------------------------------------------------------------
    members_df = (
        pd.DataFrame(members_per_isin.items(), columns=["isin", "unique_members"])
        .sort_values("isin")
    )

    fig_members = px.bar(
        members_df,
        x="isin",
        y="unique_members",
        title="Unique members per ISIN",
        labels={"isin": "ISIN", "unique_members": "Number of unique members"},
        color_discrete_sequence=["#5B8FF9"],
    )
    fig_members.update_layout(xaxis_tickangle=90, bargap=0.2)
    save_plotly(fig_members, "members_per_isin")

    # ------------------------------------------------------------------
    # Proprietary vs client trades plot
    # ------------------------------------------------------------------
    trades_df = pd.DataFrame(trade_records).sort_values("isin")
    trades_long = trades_df.melt(
        id_vars="isin",
        value_vars=["client_trades", "proprietary_trades"],
        var_name="trade_type",
        value_name="trades",
    ).replace({"client_trades": "Client", "proprietary_trades": "Proprietary"})
    fig_trades = px.bar(
        trades_long,
        x="isin",
        y="trades",
        color="trade_type",
        title="Proprietary vs client trades per ISIN",
        labels={"isin": "ISIN", "trades": "Number of trades", "trade_type": "Trade type"},
        color_discrete_map={"Client": COLOR_CLIENT, "Proprietary": COLOR_PROPRIETARY},
    )
    fig_trades.update_layout(barmode="stack", xaxis_tickangle=90, bargap=0.2)
    save_plotly(fig_trades, "proprietary_vs_client_trades_per_isin")

    # ------------------------------------------------------------------
    # Member coverage across ISINs
    # ------------------------------------------------------------------
    coverage = pd.Series({m: len(isins) for m, isins in member_to_isins.items()}, name="isin_count") / total_isins
    coverage_df = (
        coverage.rename("coverage")
        .reset_index()
        .rename(columns={"index": "member"})
        .sort_values("coverage", ascending=False)
    )

    member_to_nationality: Dict[object, str] = {}
    mixed_members: List[str] = []
    for member in coverage_df["member"].tolist():
        counts = member_to_nationalities.get(member, Counter())
        labels = sorted(counts.keys())
        if not labels:
            member_to_nationality[member] = "UNKNOWN"
        elif len(labels) == 1:
            member_to_nationality[member] = labels[0]
        else:
            member_to_nationality[member] = "MIXED"
            mixed_members.append(str(member))

    if mixed_members:
        log_print(
            f"[warn] Found members with multiple '{AGGRESSIVE_MEMBER_NATIONALITY_COL}' labels "
            f"(showing as MIXED): {len(mixed_members)} members. Example IDs: {mixed_members[:10]}"
        )

    coverage_df["member_str"] = coverage_df["member"].astype(str)
    coverage_df["member_nationality"] = coverage_df["member"].map(member_to_nationality)

    nationality_color_map = {
        "IT": COLOR_PROPRIETARY,
        "FOREIGN": COLOR_CLIENT,
        "MIXED": "#9467bd",
        "UNKNOWN": "#7f7f7f",
    }
    fig_cov = px.bar(
        coverage_df,
        x="member_str",
        y="coverage",
        color="member_nationality",
        labels={
            "member_str": "Member ID",
            "coverage": "Fraction of ISINs traded",
            "member_nationality": "Member nationality",
        },
        category_orders={"member_str": coverage_df["member_str"].tolist()},
        color_discrete_map=nationality_color_map,
    )
    fig_cov.update_layout(xaxis_tickangle=90, xaxis_tickfont=dict(size=7), bargap=0.2)
    save_plotly(fig_cov, "member_isin_coverage_per_member")

    # ------------------------------------------------------------------
    # Member activity heatmap
    # ------------------------------------------------------------------
    activity_series = pd.concat(activity_parts)
    activity_series.index = activity_series.index.set_names(["date", "member"])
    activity = activity_series.groupby(level=["date", "member"]).sum().reset_index()
    activity_pivot = activity.pivot(index="date", columns="member", values="trades").fillna(0)
    activity_pivot = activity_pivot.sort_index()
    activity_pivot = activity_pivot[sorted(activity_pivot.columns)]
    iso_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in activity_pivot.index]
    members_sorted = [str(m) for m in activity_pivot.columns]

    heatmap = go.Figure(
        data=go.Heatmap(
            z=activity_pivot.values,
            x=members_sorted,
            y=iso_dates,
            colorscale="Magma",
            colorbar={"title": "Number of trades"},
        )
    )
    heatmap.update_layout(
        xaxis_title="Member",
        yaxis_title="Day",
        xaxis=dict(showticklabels=True, type="category"),
        yaxis=dict(showticklabels=True, type="category"),
    )

    # Save PNG for the heatmap
    _, heatmap_png = save_plotly_figure(
        heatmap,
        stem="member_activity_heatmap",
        dirs=PLOT_OUTPUT_DIRS,
        write_html=False,
        write_png=True,
        strict_png=False,
    )
    if heatmap_png is None:
        log_print("[warn] Could not save PNG for member_activity_heatmap (Plotly static export unavailable).")
    else:
        log_print(f"Saved PNG to {heatmap_png}")

    # Build interactive HTML with hover-driven per-ISIN bar chart
    bar_stub = go.Figure(
        data=[
            go.Bar(
                name="Buy",
                x=all_isins_sorted,
                y=[0] * len(all_isins_sorted),
                marker=dict(color="#91CC75"),
            ),
            go.Bar(
                name="Sell",
                x=all_isins_sorted,
                y=[0] * len(all_isins_sorted),
                marker=dict(color="#EE6666"),
            ),
        ],
        layout=go.Layout(
            xaxis=dict(title="ISIN", categoryorder="array", categoryarray=all_isins_sorted),
            yaxis=dict(title="Trades (Buy+, Sell-)", rangemode="tozero", zeroline=True, zerolinecolor="#888"),
            barmode="relative",
        ),
    )

    heatmap_json = pio.to_json(heatmap, validate=False)
    bar_json = pio.to_json(bar_stub, validate=False)
    breakdown_json = json.dumps(per_day_member_isin)
    all_isins_json = json.dumps(all_isins_sorted)

    html_content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Member activity heatmap</title>
  <script src="https://cdn.plot.ly/plotly-2.31.1.min.js"></script>
</head>
<body>
  <div id="heatmap" style="width: 100%; height: 600px;"></div>
  <div id="bar" style="width: 100%; height: 400px;"></div>
  <script>
    const heatmapFig = {heatmap_json};
    const barFig = {bar_json};
    const breakdown = {breakdown_json};
    const allIsins = {all_isins_json};

    Plotly.newPlot('heatmap', heatmapFig.data, heatmapFig.layout);
    Plotly.newPlot('bar', barFig.data, barFig.layout);

    const heatmapDiv = document.getElementById('heatmap');
    heatmapDiv.on('plotly_hover', function(evt) {{
      const pt = evt.points[0];
      const member = String(pt.x);
      const date = String(pt.y);
      const memberData = (breakdown[date] || {{}})[member];
      if (!memberData) {{
        Plotly.react('bar',
          [
            {{type: 'bar', name: 'Buy',  x: allIsins, y: allIsins.map(() => 0), marker: {{color: '#91CC75'}}}},
            {{type: 'bar', name: 'Sell', x: allIsins, y: allIsins.map(() => 0), marker: {{color: '#EE6666'}}}},
          ],
          {{
            title: 'No per-ISIN data for member ' + member + ' on ' + date,
            xaxis: {{title: 'ISIN', categoryorder: 'array', categoryarray: allIsins}},
            yaxis: {{title: 'Trades (Buy+, Sell-)', rangemode: 'tozero', zeroline: true, zerolinecolor: '#888'}},
            barmode: 'relative',
          }}
        );
        return;
      }}
      const isins = allIsins.filter(isin => memberData[isin]);
      const buys = isins.map(isin => (memberData[isin].buy || 0));
      const sells = isins.map(isin => -(memberData[isin].sell || 0));
      Plotly.react('bar', [
        {{
          type: 'bar',
          name: 'Buy',
          x: isins,
          y: buys,
          marker: {{color: '#91CC75'}},
        }},
        {{
          type: 'bar',
          name: 'Sell',
          x: isins,
          y: sells,
          marker: {{color: '#EE6666'}},
        }},
      ], {{
        title: 'Per-ISIN trades (buy/sell) for member ' + member + ' on ' + date,
        xaxis: {{title: 'ISIN', categoryorder: 'array', categoryarray: allIsins}},
        yaxis: {{title: 'Trades (Buy+, Sell-)', rangemode: 'tozero', zeroline: true, zerolinecolor: '#888'}},
        barmode: 'relative',
      }});
    }});
  </script>
</body>
</html>"""

    heatmap_html_path = HTML_DIR / "member_activity_heatmap.html"
    heatmap_html_path.write_text(html_content)
    log_print(f"Saved HTML to {heatmap_html_path}")


if __name__ == "__main__":
    main()
