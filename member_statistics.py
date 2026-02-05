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
- `DATA_DIR` (`<repo>/data` by default) must contain one or more `*.parquet` files.
  Each file is treated as one ISIN; the ISIN is inferred from the filename stem
  (e.g., `FR0012345678.parquet` -> ISIN `FR0012345678`).
- Required parquet columns (read via pandas):
  - `"ID Member"`: member identifier (any dtype; converted to string for display).
  - `"Trade Time"`: trade timestamp (must be datetime-like; the script uses `.dt`).
  - `"Trade Type Aggressive"`: used to classify proprietary vs client trades.
  - `"Direction"`: used to split buy vs sell trades in the hover bar chart
    (expected sign: buy > 0, sell < 0). If missing, the script attempts to derive
    it from `"Total Quantity Buy"` / `"Total Quantity Sell"` when available.

Processing details
------------------
- Trades are filtered to `TRADING_HOURS` (default 09:30-17:30) using only the
  time-of-day portion of `"Trade Time"`.
- Proprietary trades are those where `"Trade Type Aggressive" == "Dealing_on_own_account"`.
  All other values are treated as client trades.
- Member coverage is computed as: (# ISINs traded by member) / (total # ISIN files).

Outputs
-------
Creates `images/member_statistics/` (and subfolders) containing:
- Interactive HTML plots (`images/member_statistics/html/*.html`)
- Static PNG exports (`images/member_statistics/png/*.png`), when Plotly image
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

    python member_statistics.py

To change the data location or trading-hours filter, edit `DATA_DIR` and
`TRADING_HOURS` in the "Config" section near the top of the file.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
PLOT_DIR = Path(__file__).resolve().parent / "images" / "member_statistics"
HTML_DIR = PLOT_DIR / "html"
PNG_DIR = PLOT_DIR / "png"
TRADING_HOURS = ("09:30:00", "17:30:00")
for d in (PLOT_DIR, HTML_DIR, PNG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------
LOG_PATH = Path(__file__).with_suffix(".log")
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
    message = " ".join(str(a) for a in args)
    logger.info(message)
    _print(*args, **kwargs)


def save_plotly(fig, stem: str):
    """Save a Plotly figure to both HTML and PNG formats."""
    html_path = HTML_DIR / f"{stem}.html"
    png_path = PNG_DIR / f"{stem}.png"
    fig.write_html(html_path)
    try:
        fig.write_image(png_path)
    except Exception as exc:
        log_print(f"[warn] Could not save PNG for {stem}: {exc}")
    else:
        log_print(f"Saved PNG to {png_path}")
    log_print(f"Saved HTML to {html_path}")


# ---------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------
def load_parquet_paths(data_dir: Path) -> List[Path]:
    paths = sorted(p for p in data_dir.glob("*.parquet") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return paths


def main():
    paths = load_parquet_paths(DATA_DIR)

    total_members: set = set()
    member_to_isins: Dict[int, set] = defaultdict(set)
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
                columns=["ID Member", "Trade Time", "Trade Type Aggressive", "Direction"],
            )
        except Exception:
            # Backwards/heterogeneous datasets: fall back to full read and derive Direction if needed.
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
        color_discrete_map={"Client": "#91CC75", "Proprietary": "#5470C6"},
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

    coverage_df["member_str"] = coverage_df["member"].astype(str)
    fig_cov = px.bar(
        coverage_df,
        x="member_str",
        y="coverage",
        title="ISIN coverage per member",
        labels={"member_str": "Member ID", "coverage": "Fraction of ISINs traded"},
        color_discrete_sequence=["#EE6666"],
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
        title="Member activity heatmap (all ISINs)",
        xaxis_title="Member",
        yaxis_title="Day",
        xaxis=dict(showticklabels=False, type="category"),
        yaxis=dict(showticklabels=False, type="category"),
    )

    # Save PNG for the heatmap
    heatmap_png = PNG_DIR / "member_activity_heatmap.png"
    try:
        heatmap.write_image(heatmap_png)
        log_print(f"Saved PNG to {heatmap_png}")
    except Exception as exc:
        log_print(f"[warn] Could not save PNG for member_activity_heatmap: {exc}")

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
            title="Hover a cell to see per-ISIN trades (buy/sell)",
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
