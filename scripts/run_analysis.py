#!/usr/bin/env python3
"""Run repository analysis workflows from one CLI entrypoint."""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Sequence


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class CommandSpec:
    """Describe one unified CLI command.

    Parameters
    ----------
    module:
        Import path of the workflow module that exposes a ``main`` function.
    description:
        Short human-readable description shown in help output.
    config_env:
        Optional environment variable consumed by modules that load YAML
        configuration at import time or during startup.
    pass_config_path:
        Whether to pass the normalized ``--config-path`` option through to the
        underlying workflow.

    Returns
    -------
    CommandSpec
        Immutable command metadata used by the dispatcher.

    Notes
    -----
    Some legacy workflows read config at import time, so the dispatcher must set
    the environment variable before importing the module.

    Examples
    --------
    >>> CommandSpec("moimpact.workflows.metaorders.compute", "compute metaorders").module
    'moimpact.workflows.metaorders.compute'
    """

    module: str
    description: str
    config_env: str | None = None
    pass_config_path: bool = False


COMMANDS: dict[tuple[str, str], CommandSpec] = {
    ("metaorders", "compute"): CommandSpec(
        "moimpact.workflows.metaorders.compute",
        "reconstruct metaorders and compute impact outputs",
        config_env="METAORDER_COMP_CONFIG",
    ),
    ("metaorders", "distributions"): CommandSpec(
        "moimpact.workflows.metaorders.distributions",
        "compare proprietary/client metaorder distributions",
        config_env="METAORDER_DISTRIBUTIONS_CONFIG",
    ),
    ("metaorders", "summary"): CommandSpec(
        "moimpact.workflows.metaorders.summary",
        "build metaorder summary statistics",
        config_env="METAORDER_SUMMARY_STATS_CONFIG",
    ),
    ("metaorders", "intraday-impact"): CommandSpec(
        "moimpact.workflows.metaorders.intraday_impact",
        "split impact fits by intraday session",
        config_env="METAORDER_INTRADAY_CONFIG",
    ),
    ("metaorders", "start-event"): CommandSpec(
        "moimpact.workflows.metaorders.start_event_study",
        "run start-intensity event study around high-participation anchors",
        pass_config_path=True,
    ),
    ("metaorders", "start-time"): CommandSpec(
        "moimpact.workflows.metaorders.start_time_distribution",
        "compare intraday start-time distributions",
        config_env="METAORDER_START_TIME_DISTRIBUTION_CONFIG",
        pass_config_path=True,
    ),
    ("impact", "overlay"): CommandSpec(
        "moimpact.workflows.impact.overlay",
        "plot proprietary-vs-client impact fit overlays",
        config_env="PLOT_PROP_NONPROP_FITS_CONFIG",
    ),
    ("crowding", "daily"): CommandSpec(
        "moimpact.workflows.crowding.daily",
        "run canonical daily/cross/all-others crowding analysis",
        config_env="CROWDING_CONFIG",
    ),
    ("crowding", "eta"): CommandSpec(
        "moimpact.workflows.crowding.eta",
        "run crowding versus participation-rate analysis",
        pass_config_path=True,
    ),
    ("crowding", "impact"): CommandSpec(
        "moimpact.workflows.crowding.impact",
        "run crowding-conditioned impact analysis",
        config_env="CROWDING_IMPACT_CONFIG",
        pass_config_path=True,
    ),
    ("crowding", "intraday"): CommandSpec(
        "moimpact.workflows.crowding.intraday",
        "profile crowding by intraday start bin",
        config_env="CROWDING_INTRADAY_PROFILE_CONFIG",
        pass_config_path=True,
    ),
    ("crowding", "overlap"): CommandSpec(
        "moimpact.workflows.crowding.overlap",
        "compute active-overlap crowding features",
        config_env="CROWDING_OVERLAP_ANALYSIS_CONFIG",
        pass_config_path=True,
    ),
    ("crowding", "member-overlap"): CommandSpec(
        "moimpact.workflows.crowding.member_overlap",
        "analyze member-level active-overlap crowding",
        config_env="MEMBER_ACTIVE_OVERLAP_CROWDING_CONFIG",
        pass_config_path=True,
    ),
    ("execution", "schedule"): CommandSpec(
        "moimpact.workflows.execution.schedule",
        "compare proprietary/client execution schedules",
        config_env="METAORDER_EXECUTION_SCHEDULE_CONFIG",
    ),
    ("execution", "typology"): CommandSpec(
        "moimpact.workflows.execution.typology",
        "cluster execution typologies",
        pass_config_path=True,
    ),
    ("execution", "cluster"): CommandSpec(
        "moimpact.workflows.execution.cluster",
        "run PCA/k-means clustering on metaorder features",
        pass_config_path=True,
    ),
    ("members", "stats"): CommandSpec(
        "moimpact.workflows.members.stats",
        "build member and ISIN descriptive plots",
    ),
    ("paper", "figures"): CommandSpec(
        "moimpact.workflows.paper.figures",
        "generate figures referenced by paper/main.tex",
        config_env="PAPER_FIGURES_CONFIG",
    ),
    ("paper", "style-preview"): CommandSpec(
        "moimpact.workflows.paper.style_preview",
        "create a no-data temporary paper build with fake styled figures",
        pass_config_path=True,
    ),
}


ALIASES: dict[tuple[str, str], tuple[str, str]] = {
    ("metaorders", "intraday"): ("metaorders", "intraday-impact"),
    ("event", "study"): ("metaorders", "start-event"),
    ("crowding", "part-rate"): ("crowding", "eta"),
    ("paper", "figs"): ("paper", "figures"),
    ("paper", "preview"): ("paper", "style-preview"),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run a workflow selected by a two-part command.

    Parameters
    ----------
    argv:
        Optional argument list excluding the program name. When ``None``,
        ``sys.argv[1:]`` is used.

    Returns
    -------
    int
        Process-style exit code returned by the selected workflow, or ``0`` for
        help/list output.

    Notes
    -----
    The dispatcher accepts a uniform ``--config PATH`` alias. It sets the
    workflow-specific environment variable before import when needed, because
    some workflows resolve YAML defaults at module import time.

    Examples
    --------
    >>> main(["--list"])
    0
    """

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0
    if args[0] == "--list":
        _print_command_list()
        return 0
    if len(args) == 1 and args[0] in _groups():
        _print_group_help(args[0])
        return 0
    if len(args) == 2 and args[0] in _groups() and args[1] in {"-h", "--help"}:
        _print_group_help(args[0])
        return 0
    if len(args) < 2:
        _die(f"Expected a two-part command, got: {' '.join(args)}")

    key = (args[0], args[1])
    key = ALIASES.get(key, key)
    spec = COMMANDS.get(key)
    if spec is None:
        _die(f"Unknown command: {args[0]} {args[1]}")

    forwarded_args, config_path = _extract_config_alias(args[2:])
    if config_path is not None:
        if spec.config_env is not None:
            os.environ[spec.config_env] = config_path
        if spec.pass_config_path:
            forwarded_args = ["--config-path", config_path, *forwarded_args]
    return _run_module_main(spec.module, forwarded_args)


def _run_module_main(module_name: str, argv: list[str]) -> int:
    """Import a workflow module and call its ``main`` function.

    Parameters
    ----------
    module_name:
        Fully qualified module name.
    argv:
        Arguments to forward to the workflow, excluding program name.

    Returns
    -------
    int
        Workflow exit code. ``None`` returns are normalized to ``0``.

    Notes
    -----
    Older workflows expose ``main()`` and parse ``sys.argv`` internally, while
    newer workflows expose ``main(argv)``. This adapter supports both forms.

    Examples
    --------
    The function is intended for CLI dispatch and is not executed in doctests.
    """

    module = importlib.import_module(module_name)
    module_main = getattr(module, "main", None)
    if not callable(module_main):
        raise AttributeError(f"Workflow module has no callable main(): {module_name}")

    result: object
    if _main_accepts_argv(module_main):
        result = module_main(argv)
    else:
        old_argv = sys.argv[:]
        sys.argv = [f"{module_name.replace('.', '/')}.py", *argv]
        try:
            result = module_main()
        finally:
            sys.argv = old_argv
    return int(result) if result is not None else 0


def _main_accepts_argv(func: Callable[..., object]) -> bool:
    """Return whether a workflow ``main`` accepts an explicit argv argument.

    Parameters
    ----------
    func:
        Callable object to inspect.

    Returns
    -------
    bool
        ``True`` when at least one positional parameter is available.

    Notes
    -----
    Bound and unbound Python functions both expose enough signature information
    for this simple adapter.

    Examples
    --------
    >>> def f(argv=None): return 0
    >>> _main_accepts_argv(f)
    True
    """

    signature = inspect.signature(func)
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            return True
    return False


def _extract_config_alias(argv: Sequence[str]) -> tuple[list[str], str | None]:
    """Extract ``--config`` or ``--config-path`` from forwarded arguments.

    Parameters
    ----------
    argv:
        Raw forwarded argument list.

    Returns
    -------
    tuple[list[str], str | None]
        Arguments with the config option removed, plus the config path when
        provided.

    Notes
    -----
    Only one config path is allowed. This keeps command invocation
    deterministic and avoids precedence ambiguity.

    Examples
    --------
    >>> _extract_config_alias(["--config", "a.yml", "--dry-run"])
    (['--dry-run'], 'a.yml')
    """

    cleaned: list[str] = []
    config_path: str | None = None
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in {"--config", "--config-path"}:
            if config_path is not None:
                _die("Pass only one of --config or --config-path.")
            if i + 1 >= len(argv):
                _die(f"{token} requires a path value.")
            config_path = argv[i + 1]
            i += 2
            continue
        if token.startswith("--config="):
            if config_path is not None:
                _die("Pass only one of --config or --config-path.")
            config_path = token.split("=", 1)[1]
            i += 1
            continue
        if token.startswith("--config-path="):
            if config_path is not None:
                _die("Pass only one of --config or --config-path.")
            config_path = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned, config_path


def _groups() -> set[str]:
    """Return available top-level command groups.

    Parameters
    ----------
    None.

    Returns
    -------
    set[str]
        Top-level command group names.

    Notes
    -----
    Alias-only groups are intentionally omitted from the canonical list.

    Examples
    --------
    >>> "metaorders" in _groups()
    True
    """

    return {group for group, _ in COMMANDS}


def _print_help() -> None:
    """Print top-level usage information.

    Parameters
    ----------
    None.

    Returns
    -------
    None
        Writes help text to stdout.

    Notes
    -----
    Workflow-specific options are owned by each workflow and can be inspected by
    appending ``--help`` after the two-part command.

    Examples
    --------
    >>> _print_help()  # doctest: +ELLIPSIS
    Usage:...
    """

    print("Usage: python scripts/run_analysis.py <group> <command> [workflow options]")
    print()
    print("Use --config PATH as a uniform alias for workflow YAML config paths.")
    print("Use --list to show all commands.")
    print()
    print("Groups:")
    for group in sorted(_groups()):
        print(f"  {group}")


def _print_group_help(group: str) -> None:
    """Print command help for one top-level group.

    Parameters
    ----------
    group:
        Top-level command group.

    Returns
    -------
    None
        Writes help text to stdout.

    Notes
    -----
    Unknown groups are handled by the caller before this function is used.

    Examples
    --------
    >>> _print_group_help("members")  # doctest: +ELLIPSIS
    Commands for members:
    ...
    """

    print(f"Commands for {group}:")
    for (command_group, command), spec in sorted(COMMANDS.items()):
        if command_group == group:
            print(f"  {command:<16} {spec.description}")


def _print_command_list() -> None:
    """Print all canonical commands.

    Parameters
    ----------
    None.

    Returns
    -------
    None
        Writes command names and descriptions to stdout.

    Notes
    -----
    Aliases are not printed to keep the output stable.

    Examples
    --------
    >>> _print_command_list()  # doctest: +ELLIPSIS
    crowding daily...
    """

    for (group, command), spec in sorted(COMMANDS.items()):
        print(f"{group} {command:<16} {spec.description}")


def _die(message: str) -> None:
    """Raise a CLI usage error.

    Parameters
    ----------
    message:
        Error message to show to the user.

    Returns
    -------
    None
        This function always raises ``SystemExit``.

    Notes
    -----
    ``SystemExit`` is used so shell callers receive a normal CLI failure code.

    Examples
    --------
    This function is not invoked in doctests because it exits.
    """

    print(f"error: {message}", file=sys.stderr)
    print("Run `python scripts/run_analysis.py --help` for usage.", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
