"""
YAML configuration and path-templating helpers shared across scripts.

The repository historically duplicated small utilities (loading YAML configs,
resolving repo-relative paths, formatting placeholder templates). This module
centralizes those helpers to reduce copy/paste drift while keeping behavior
conservative and explicit (unknown placeholders raise immediately).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional


def load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    """
    Summary
    -------
    Load a YAML file expected to contain a top-level mapping (dictionary).

    Parameters
    ----------
    config_path : Path
        Path to the YAML file to read.

    Returns
    -------
    cfg : dict[str, Any]
        Parsed YAML content. Returns an empty dict when the file is empty.

    Notes
    -----
    - Raises FileNotFoundError if `config_path` does not exist.
    - Raises TypeError if the YAML file does not parse into a mapping.
    - Uses `yaml.safe_load`.

    Examples
    --------
    >>> from pathlib import Path
    >>> cfg = load_yaml_mapping(Path("config_ymls/metaorder_summary_statistics.yml"))
    >>> isinstance(cfg, dict)
    True
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise ImportError("Missing dependency: pyyaml is required to read YAML configs.") from exc

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a mapping (YAML dict): {config_path}")
    return cfg


def cfg_require(cfg: Mapping[str, Any], key: str, config_path: Path) -> Any:
    """
    Summary
    -------
    Fetch a required configuration value from a mapping.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        Parsed YAML configuration mapping.
    key : str
        Required key.
    config_path : Path
        Path used only to produce informative error messages.

    Returns
    -------
    value : Any
        The value stored under `key`.

    Notes
    -----
    - Raises KeyError if `key` is missing.

    Examples
    --------
    >>> cfg_require({"A": 1}, "A", Path("cfg.yml"))
    1
    """
    if key not in cfg:
        raise KeyError(f"Missing required key '{key}' in {config_path}")
    return cfg[key]


def format_path_template(template: str, context: Mapping[str, str]) -> str:
    """
    Summary
    -------
    Format a path template using a restricted set of placeholders.

    Parameters
    ----------
    template : str
        Template string that may contain placeholders like `{DATASET_NAME}`.
    context : Mapping[str, str]
        Placeholder names (keys) and replacement values.

    Returns
    -------
    formatted : str
        The formatted string.

    Notes
    -----
    - If `template` contains placeholders not present in `context`, this function
      raises KeyError with a helpful message listing allowed placeholders. This
      avoids silent mistakes when configs drift.

    Examples
    --------
    >>> format_path_template("out_files/{DATASET_NAME}", {"DATASET_NAME": "ftsemib"})
    'out_files/ftsemib'
    """
    if "{" not in template:
        return template
    try:
        return template.format(**context)
    except KeyError as exc:
        allowed = ", ".join(sorted(context.keys()))
        raise KeyError(
            f"Unknown placeholder {exc} in path template '{template}'. "
            f"Allowed placeholders: {allowed}."
        ) from exc


def resolve_repo_path(script_dir: Path, value: str | Path) -> Path:
    """
    Summary
    -------
    Resolve a path relative to a script directory (repo root in this repository).

    Parameters
    ----------
    script_dir : Path
        Directory relative paths are resolved against (typically the directory
        containing the calling script).
    value : str | Path
        Input path (absolute or relative).

    Returns
    -------
    resolved : Path
        Absolute resolved Path.

    Notes
    -----
    - Relative paths are resolved as `(script_dir / value).resolve()`.

    Examples
    --------
    >>> from pathlib import Path
    >>> resolve_repo_path(Path("."), "config_ymls").is_absolute()
    True
    """
    path = Path(value)
    if not path.is_absolute():
        path = (script_dir / path).resolve()
    return path


def resolve_opt_repo_path(
    script_dir: Path,
    value: Optional[str | Path],
    default: Path,
) -> Path:
    """
    Summary
    -------
    Resolve an optional repo-relative path, falling back to a default.

    Parameters
    ----------
    script_dir : Path
        Directory relative paths are resolved against.
    value : Optional[str | Path]
        Optional input path. If None, `default` is used.
    default : Path
        Fallback path (may be relative or absolute).

    Returns
    -------
    resolved : Path
        Resolved absolute Path.

    Notes
    -----
    - This mirrors the common "null means default" pattern in YAML configs.

    Examples
    --------
    >>> from pathlib import Path
    >>> resolve_opt_repo_path(Path("."), None, Path("out_files")).is_absolute()
    True
    """
    if value is None:
        return resolve_repo_path(script_dir, default)
    return resolve_repo_path(script_dir, value)
