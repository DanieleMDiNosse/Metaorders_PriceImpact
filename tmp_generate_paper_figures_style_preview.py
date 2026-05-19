#!/usr/bin/env python3
"""Regenerate paper figures for style preview without bootstrap/permutation/metaorder recompute.

This is a temporary orchestration script. It writes figures to paper/images and
routes non-figure outputs/logs to a timestamped tmp_paper_figure_style_preview_* folder.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any, Mapping

import yaml

REPO = Path(__file__).resolve().parent
DATASET = "ftsemib"
PAPER_IMAGES = REPO / "paper" / "images"
CANONICAL_OUT = REPO / "out_files" / DATASET
STYLE_CONFIG = REPO / "config_ymls" / "paper_figure_styles.yml"
PAPER_FIGURES_CONFIG = REPO / "config_ymls" / "paper_figures.yml"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PREVIEW_ROOT = REPO / f"tmp_paper_figure_style_preview_{TIMESTAMP}"
TMP_CFG_DIR = PREVIEW_ROOT / "configs"
TMP_OUT_DIR = PREVIEW_ROOT / "out_files"
LOG_DIR = PREVIEW_ROOT / "logs"
TMP_META_OUT = TMP_OUT_DIR / "metaorders_inputs"

STYLE_UPDATES = {
    "IMPACT_FIT_FIGURE_WIDTH": 1200,
    "IMPACT_FIT_FIGURE_HEIGHT": 1000,
}

PROP_PARQUET = CANONICAL_OUT / "metaorders_info_sameday_filtered_member_proprietary.parquet"
CLIENT_PARQUET = CANONICAL_OUT / "metaorders_info_sameday_filtered_member_non_proprietary.parquet"
PROP_IT_PARQUET = CANONICAL_OUT / "metaorders_info_sameday_filtered_member_proprietary_member_nationality_it.parquet"
CLIENT_IT_PARQUET = CANONICAL_OUT / "metaorders_info_sameday_filtered_member_non_proprietary_member_nationality_it.parquet"
PROP_DICT = CANONICAL_OUT / "metaorders_dict_all_member_proprietary.pkl"
CLIENT_DICT = CANONICAL_OUT / "metaorders_dict_all_member_non_proprietary.pkl"
OVERLAP_FEATURES = CANONICAL_OUT / "crowding_overlap_analysis" / "overlap_features.parquet"


def require_paths(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required existing inputs:\n" + "\n".join(missing))


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML config is not a mapping: {path}")
    return data


def write_temp_config(base: Path, updates: Mapping[str, Any], name: str) -> Path:
    cfg = load_yaml(base)
    cfg.update(dict(updates))
    path = TMP_CFG_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


def tail(path: Path, n: int = 80) -> str:
    if not path.exists():
        return "<missing log>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def command_display(cmd: list[str], env_updates: Mapping[str, str] | None = None) -> str:
    env_part = " ".join(f"{k}={v}" for k, v in sorted((env_updates or {}).items()))
    cmd_part = " ".join(cmd)
    return (env_part + " " + cmd_part).strip()


def run(name: str, cmd: list[str], *, env_updates: Mapping[str, str] | None = None) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    env = os.environ.copy()
    env.update(BASE_ENV)
    if env_updates:
        env.update(env_updates)
    display = command_display(cmd, env_updates)
    print(f"[run:{name}] {display}")
    print(f"[log:{name}] {log_path}")
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            cmd,
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        print(f"[failed:{name}] exit_code={completed.returncode}")
        print(tail(log_path))
        raise SystemExit(completed.returncode)
    print(f"[done:{name}]")


def symlink_input(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def copy_summary_compatibility() -> None:
    pairs = [
        (
            PAPER_IMAGES / "member_metaorder_summary_statistics" / "png" / "mean_daily_metaorder_volume_share.png",
            PAPER_IMAGES / "prop_vs_nonprop" / "png" / "mean_daily_metaorder_volume_share.png",
        ),
        (
            PAPER_IMAGES / "member_metaorder_summary_statistics" / "html" / "mean_daily_metaorder_volume_share.html",
            PAPER_IMAGES / "prop_vs_nonprop" / "html" / "mean_daily_metaorder_volume_share.html",
        ),
        (
            PAPER_IMAGES / "member_metaorder_summary_statistics" / "pdf" / "mean_daily_metaorder_volume_share.pdf",
            PAPER_IMAGES / "prop_vs_nonprop" / "pdf" / "mean_daily_metaorder_volume_share.pdf",
        ),
    ]
    for src, dst in pairs:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[copy] {src} -> {dst}")


BASE_ENV = {
    "PAPER_FIGURE_STYLE_MODE": "per-figure",
    "PAPER_FIGURE_STYLES_CONFIG": str(STYLE_CONFIG),
    "PLOTLY_WRITE_PDF": "true",
}

PY = sys.executable


def main() -> int:
    require_paths([
        STYLE_CONFIG,
        PAPER_FIGURES_CONFIG,
        PROP_PARQUET,
        CLIENT_PARQUET,
        PROP_IT_PARQUET,
        CLIENT_IT_PARQUET,
        PROP_DICT,
        CLIENT_DICT,
        OVERLAP_FEATURES,
    ])
    PAPER_IMAGES.mkdir(parents=True, exist_ok=True)
    TMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_META_OUT.mkdir(parents=True, exist_ok=True)

    # Symlink canonical filtered metaorder tables into a temporary OUT_DIR so
    # metaorders compute can render WLS/path figures without running metaorder or SQL recomputation.
    for src in [PROP_PARQUET, CLIENT_PARQUET, PROP_IT_PARQUET, CLIENT_IT_PARQUET]:
        symlink_input(src, TMP_META_OUT)

    manifest: dict[str, Any] = {
        "created_at": TIMESTAMP,
        "repo": str(REPO),
        "paper_images": str(PAPER_IMAGES),
        "style_config": str(STYLE_CONFIG),
        "preview_root": str(PREVIEW_ROOT),
        "no_bootstrap_permutation_recompute_policy": {
            "metaorders_compute": {
                "RUN_INTRO": False,
                "RUN_METAORDER_COMPUTATION": False,
                "RUN_SQL_FITS": False,
                "RUN_WLS": True,
                "RUN_IMPACT_PATH_PLOT": True,
            },
            "crowding_daily": {
                "BOOTSTRAP_RUNS": 0,
                "BOOTSTRAP_HEATMAP": False,
                "ACF_BOOTSTRAP_SAMPLES": 0,
                "RUN_CROWDING_VS_PART_RATE": False,
                "ATTACH_DAILY_RETURNS": False,
            },
            "crowding_eta": {
                "bootstrap_runs": 0,
                "cluster_ci": "none",
                "permutation_runs": 0,
                "run_regressions": False,
            },
            "crowding_impact": {
                "BOOTSTRAP_RUNS": 0,
                "RUN_JOINT_BIN_REGRESSION": False,
                "RUN_POOLED_LOG_REGRESSION": False,
            },
            "member_overlap": {"BOOTSTRAP_RUNS": 0, "input": str(OVERLAP_FEATURES)},
            "distributions": {"POWERLAW_FULL_BOOTSTRAP_ENABLED": False, "POWERLAW_FULL_BOOTSTRAP_RUNS": 0},
            "execution_schedule": {"RUN_EXECUTION_SCHEDULE_INFERENCE": False, "BOOTSTRAP_RUNS": 0},
            "impact_overlay": {"RUN_RETENTION_BOOTSTRAP": False, "RETENTION_BOOTSTRAP_RUNS": 0},
        },
    }
    (PREVIEW_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] {PREVIEW_ROOT / 'run_manifest.json'}")

    # 1. Member statistics.
    run(
        "member_statistics",
        [PY, "scripts/run_analysis.py", "members", "stats"],
        env_updates={"DATASET_NAME": DATASET, "IMG_OUTPUT_PATH_OVERRIDE": str(PAPER_IMAGES)},
    )

    # 2. Metaorder summary statistics, pooled and split. Tables/logs go to temp; figures to paper/images.
    summary_cfg = write_temp_config(
        REPO / "config_ymls" / "metaorder_summary_statistics.yml",
        {
            "DATASET_NAME": DATASET,
            "OUTPUT_FILE_PATH": str(TMP_OUT_DIR / "summary"),
            "IMG_OUTPUT_PATH": str(PAPER_IMAGES),
            "MEMBER_NATIONALITY": None,
            "PROPRIETARY_DICT_PATH": str(PROP_DICT),
            "CLIENT_DICT_PATH": str(CLIENT_DICT),
        },
        "metaorder_summary_statistics.yml",
    )
    for mode in ["false", "true"]:
        run(
            f"metaorder_summary_condition_{mode}",
            [PY, "scripts/run_analysis.py", "metaorders", "summary", "--condition-on-client-proprietary", mode],
            env_updates={"METAORDER_SUMMARY_STATS_CONFIG": str(summary_cfg)},
        )
    copy_summary_compatibility()

    # 3. Metaorder distributions with full-pipeline bootstrap disabled.
    dist_cfg = write_temp_config(
        REPO / "config_ymls" / "metaorder_distributions.yml",
        {
            "DATASET_NAME": DATASET,
            "OUTPUT_FILE_PATH": str(TMP_OUT_DIR / "distributions"),
            "IMG_OUTPUT_PATH": str(PAPER_IMAGES),
            "MEMBER_NATIONALITY": None,
            "PROPRIETARY_DICT_PATH": str(PROP_DICT),
            "CLIENT_DICT_PATH": str(CLIENT_DICT),
            "POWERLAW_FULL_BOOTSTRAP_ENABLED": False,
            "POWERLAW_FULL_BOOTSTRAP_RUNS": 0,
            "POWERLAW_FIT_MAX_WORKERS": 1,
        },
        "metaorder_distributions.yml",
    )
    run(
        "metaorder_distributions_no_bootstrap",
        [PY, "scripts/run_analysis.py", "metaorders", "distributions"],
        env_updates={"METAORDER_DISTRIBUTIONS_CONFIG": str(dist_cfg)},
    )

    # 4. Execution schedule, inference/bootstrap disabled.
    sched_cfg = write_temp_config(
        REPO / "config_ymls" / "metaorder_execution_schedule.yml",
        {
            "DATASET_NAME": DATASET,
            "OUTPUT_FILE_PATH": str(TMP_OUT_DIR / "execution_schedule"),
            "IMG_OUTPUT_PATH": str(PAPER_IMAGES),
            "MEMBER_NATIONALITY": None,
            "PROPRIETARY_PATH": str(PROP_PARQUET),
            "CLIENT_PATH": str(CLIENT_PARQUET),
            "RUN_EXECUTION_SCHEDULE_INFERENCE": False,
            "BOOTSTRAP_RUNS": 0,
        },
        "metaorder_execution_schedule.yml",
    )
    run(
        "execution_schedule_no_inference",
        [PY, "scripts/run_analysis.py", "execution", "schedule"],
        env_updates={"METAORDER_EXECUTION_SCHEDULE_CONFIG": str(sched_cfg)},
    )

    # 5. Proprietary vs client impact overlays, retention bootstrap disabled.
    overlay_cfg = write_temp_config(
        REPO / "config_ymls" / "plot_prop_nonprop_fits.yml",
        {
            "DATASET_NAME": DATASET,
            "OUTPUT_FILE_PATH": str(TMP_OUT_DIR / "impact_overlay"),
            "IMG_OUTPUT_PATH": str(PAPER_IMAGES),
            "PROPRIETARY_PATH": str(PROP_PARQUET),
            "CLIENT_PATH": str(CLIENT_PARQUET),
            "RUN_RETENTION_BOOTSTRAP": False,
            "RETENTION_BOOTSTRAP_RUNS": 0,
            **STYLE_UPDATES,
        },
        "plot_prop_nonprop_fits.yml",
    )
    run(
        "impact_overlay_no_retention_bootstrap",
        [PY, "scripts/run_analysis.py", "impact", "overlay"],
        env_updates={"PLOT_PROP_NONPROP_FITS_CONFIG": str(overlay_cfg)},
    )

    # 6. Impact fit/surface/path figures from existing filtered parquets only.
    for nationality in [None, "it"]:
        tag = "all" if nationality is None else nationality
        for proprietary in [True, False]:
            group = "proprietary" if proprietary else "client"
            for split_by_side in [False, True]:
                split_tag = "by_side" if split_by_side else "primary"
                compute_cfg = write_temp_config(
                    REPO / "config_ymls" / "metaorder_computation.yml",
                    {
                        "DATASET_NAME": DATASET,
                        "OUTPUT_FILE_PATH": str(TMP_META_OUT),
                        "IMG_OUTPUT_PATH": str(PAPER_IMAGES),
                        "PROPRIETARY": proprietary,
                        "MEMBER_NATIONALITY": nationality,
                        "RUN_INTRO": False,
                        "RUN_METAORDER_COMPUTATION": False,
                        "RUN_SQL_FITS": False,
                        "RUN_WLS": True,
                        "RUN_IMPACT_PATH_PLOT": True,
                        "RUN_SIGNATURE_PLOTS": False,
                        "SPLIT_BY_SIDE": split_by_side,
                        "COMPUTE_EXECUTION_SCHEDULES": False,
                        **STYLE_UPDATES,
                    },
                    f"metaorder_compute_{tag}_{group}_{split_tag}.yml",
                )
                run(
                    f"impact_{tag}_{group}_{split_tag}_existing_tables",
                    [PY, "scripts/run_analysis.py", "metaorders", "compute"],
                    env_updates={"METAORDER_COMP_CONFIG": str(compute_cfg)},
                )

    # 7. Daily/cross/all/member crowding figures: bootstrap/permutation and integrated eta disabled.
    crowd_daily_cfg = write_temp_config(
        REPO / "config_ymls" / "crowding_analysis.yml",
        {
            "DATASET_NAME": DATASET,
            "OUTPUT_FILE_PATH": str(TMP_OUT_DIR / "crowding_daily"),
            "IMG_OUTPUT_PATH": str(PAPER_IMAGES),
            "PROP_PATH": str(PROP_PARQUET),
            "CLIENT_PATH": str(CLIENT_PARQUET),
            "BOOTSTRAP_RUNS": 0,
            "BOOTSTRAP_HEATMAP": False,
            "ACF_BOOTSTRAP_SAMPLES": 0,
            "ATTACH_DAILY_RETURNS": False,
            "PLOT_IMBALANCE_VS_RETURNS": False,
            "ACF_IMBALANCE": False,
            "DISTRIBUTIONS_IMBALANCE": False,
            "RUN_CROWDING_VS_PART_RATE": False,
            "CROWDING_VS_PART_RATE_BOOTSTRAP_RUNS": 0,
            "CROWDING_VS_PART_RATE_CLUSTER_CI": "none",
            "CROWDING_VS_PART_RATE_PERMUTATION_RUNS": 0,
            "CROWDING_VS_PART_RATE_RUN_REGRESSIONS": False,
        },
        "crowding_analysis_daily_no_bootstrap.yml",
    )
    run(
        "crowding_daily_no_bootstrap_no_eta",
        [PY, "scripts/run_analysis.py", "crowding", "daily"],
        env_updates={"CROWDING_CONFIG": str(crowd_daily_cfg)},
    )

    # 8. Crowding vs eta style figures with CI/bootstrap/regression/permutation disabled.
    crowd_eta_cfg = write_temp_config(
        REPO / "config_ymls" / "crowding_analysis.yml",
        {
            "DATASET_NAME": DATASET,
            "OUTPUT_FILE_PATH": str(TMP_OUT_DIR / "crowding_eta"),
            "IMG_OUTPUT_PATH": str(PAPER_IMAGES),
            "PROP_PATH": str(PROP_PARQUET),
            "CLIENT_PATH": str(CLIENT_PARQUET),
            "CROWDING_VS_PART_RATE_BOOTSTRAP_RUNS": 0,
            "CROWDING_VS_PART_RATE_CLUSTER_CI": "none",
            "CROWDING_VS_PART_RATE_PERMUTATION_RUNS": 0,
            "CROWDING_VS_PART_RATE_RUN_REGRESSIONS": False,
        },
        "crowding_analysis_eta_no_bootstrap.yml",
    )
    run(
        "crowding_eta_no_bootstrap_no_permutation",
        [
            PY,
            "scripts/run_analysis.py",
            "crowding",
            "eta",
            "--config-path",
            str(crowd_eta_cfg),
            "--dataset-name",
            DATASET,
            "--prop-path",
            str(PROP_PARQUET),
            "--client-path",
            str(CLIENT_PARQUET),
            "--output-file-path",
            str(TMP_OUT_DIR / "crowding_eta"),
            "--img-output-path",
            str(PAPER_IMAGES),
            "--analysis-tag",
            "crowding_vs_part_rate",
            "--bootstrap-runs",
            "0",
            "--cluster-ci",
            "none",
            "--permutation-runs",
            "0",
            "--no-run-regressions",
            "--plots",
            "plotly",
        ],
    )

    # 9. Crowding-conditioned impact figures with bootstrap/regressions disabled.
    crowd_impact_cfg = write_temp_config(
        REPO / "config_ymls" / "crowding_impact_analysis.yml",
        {
            "DATASET_NAME": DATASET,
            "OUTPUT_FILE_PATH": str(TMP_OUT_DIR / "crowding_impact"),
            "IMG_OUTPUT_PATH": str(PAPER_IMAGES),
            "PROP_PATH": str(PROP_PARQUET),
            "CLIENT_PATH": str(CLIENT_PARQUET),
            "BOOTSTRAP_RUNS": 0,
            "RUN_JOINT_BIN_REGRESSION": False,
            "RUN_POOLED_LOG_REGRESSION": False,
            "SHOW_PROGRESS": False,
            **STYLE_UPDATES,
        },
        "crowding_impact_analysis_no_bootstrap.yml",
    )
    run(
        "crowding_impact_no_bootstrap_no_regression",
        [
            PY,
            "scripts/run_analysis.py",
            "crowding",
            "impact",
            "--config-path",
            str(crowd_impact_cfg),
            "--dataset-name",
            DATASET,
            "--prop-path",
            str(PROP_PARQUET),
            "--client-path",
            str(CLIENT_PARQUET),
            "--output-file-path",
            str(TMP_OUT_DIR / "crowding_impact"),
            "--img-output-path",
            str(PAPER_IMAGES),
            "--analysis-tag",
            "crowding_impact",
            "--bootstrap-runs",
            "0",
            "--no-joint-regression",
            "--no-pooled-log-regression",
            "--no-progress",
            "--write-html",
            "--write-png",
        ],
    )

    # 10. Member active-overlap figures from existing overlap_features.parquet only.
    member_overlap_cfg = write_temp_config(
        REPO / "config_ymls" / "member_active_overlap_crowding.yml",
        {
            "DATASET_NAME": DATASET,
            "OUTPUT_FILE_PATH": str(TMP_OUT_DIR / "member_active_overlap"),
            "IMG_OUTPUT_PATH": str(PAPER_IMAGES),
            "INPUT_PATH": str(OVERLAP_FEATURES),
            "BOOTSTRAP_RUNS": 0,
            "N_JOBS": 1,
            "WRITE_TARGET_PARQUET": True,
        },
        "member_active_overlap_crowding_no_bootstrap.yml",
    )
    run(
        "member_active_overlap_existing_features_no_bootstrap",
        [
            PY,
            "scripts/run_analysis.py",
            "crowding",
            "member-overlap",
            "--config-path",
            str(member_overlap_cfg),
            "--dataset-name",
            DATASET,
            "--input-path",
            str(OVERLAP_FEATURES),
            "--output-file-path",
            str(TMP_OUT_DIR / "member_active_overlap"),
            "--img-output-path",
            str(PAPER_IMAGES),
            "--analysis-tag",
            "member_active_overlap_crowding",
            "--bootstrap-runs",
            "0",
            "--n-jobs",
            "1",
        ],
    )

    print(f"[done] Paper figure style-preview regeneration completed. Root: {PREVIEW_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
