"""Command-line interface: ``python -m ocsr_bench <command>``.

Commands (the cluster workflow runs them in this order, see ``slurm/``):

  selfcheck   chemistry sanity check (needs only RDKit)
  preflight   check packages, GPU, remote files and tool code
  prefetch    download datasets, checkpoints and tool code (run on a CPU node, not on a GPU)
  overview    summarise every dataset and save a sample figure
  run         run the benchmark, then write figures and the requirements audit
  report      rebuild figures and the audit from a finished run's files
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

from .config import ConfigError, get_specs, load_config
from .utils import resolve_dir

logger = logging.getLogger("ocsr_bench")
EXPECTED_ERRORS = ("ConfigError", "DatasetError", "ToolSetupError", "PipelineError")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), force=True, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocsr_bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selfcheck", help="chemistry sanity check")
    for name in ("preflight", "prefetch", "overview", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/default.yaml", help="YAML configuration (default: %(default)s)")
        p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config value, e.g. run.max_samples=20 (repeatable)")
        p.add_argument("--only-datasets", nargs="+", metavar="NAME", help="enable only these datasets")
        p.add_argument("--only-tools", nargs="+", metavar="NAME", help="enable only these tools")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _run_dir(cfg) -> Path:
    return resolve_dir(cfg["run"]["output_dir"]) / cfg["run"]["name"]


def cmd_selfcheck(args, cfg) -> int:
    from .selfcheck import run_selfcheck
    run_selfcheck()
    print("Chemistry sanity checks passed.")
    return 0


def cmd_preflight(args, cfg) -> int:
    from .preflight import run_preflight
    return 1 if run_preflight(cfg) else 0


def cmd_prefetch(args, cfg) -> int:
    """Download everything on a CPU node so the GPU job only does inference."""
    from .data import load_dataset_samples
    from .tools import build_tool
    run, data_dir = cfg["run"], resolve_dir(cfg["run"]["data_dir"])
    for spec in get_specs(cfg, "datasets"):
        loaded = load_dataset_samples(spec, data_dir, run["max_samples"], run["seed"])
        logger.info("Dataset %-14s ready: %d samples", spec.name, len(loaded.samples))
    for spec in get_specs(cfg, "tools"):
        logger.info("Preparing tool %s ...", spec.name)
        build_tool(spec, run, data_dir).prefetch()
        logger.info("Tool %-18s ready", spec.name)
    print("Prefetch complete: datasets, checkpoints and tool code are cached.")
    return 0


def cmd_overview(args, cfg) -> int:
    from .data import dataset_overview
    from .figures import plot_dataset_samples
    out = _run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    table = dataset_overview(cfg)
    table.to_csv(out / "dataset_overview.csv", index=False)
    print(table.to_string(index=False))
    try:
        plot_dataset_samples(cfg, out / "dataset_samples.png")
    except Exception as exc:
        logger.warning("Could not draw the dataset sample figure: %s: %s", type(exc).__name__, exc)
    return 0


def cmd_run(args, cfg) -> int:
    from .audit import requirements_audit
    from .figures import make_figures
    from .pipeline import run_benchmark

    def on_sigterm(signum, frame):  # lets `finally` blocks stop workers; progress is saved after every batch
        logger.warning("Received SIGTERM: stopping. Progress is saved; run the same command again to resume.")
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, on_sigterm)
    result = run_benchmark(cfg)
    make_figures(result, resolve_dir(cfg["run"]["data_dir"]))
    ok = requirements_audit(result, cfg)
    print(f"\nResults: {result.out_dir}")
    print(result.comparison[result.comparison["category"] == "all"].to_string(index=False))
    return 0  # the audit is informational: it is printed and saved but never fails the job


def cmd_report(args, cfg) -> int:
    from .audit import requirements_audit
    from .figures import make_figures
    from .results import load_result
    result = load_result(_run_dir(cfg))
    print("Figures:", [str(p) for p in make_figures(result, resolve_dir(cfg["run"]["data_dir"]))])
    requirements_audit(result, cfg)
    return 0


COMMANDS = {"selfcheck": cmd_selfcheck, "preflight": cmd_preflight, "prefetch": cmd_prefetch,
            "overview": cmd_overview, "run": cmd_run, "report": cmd_report}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    cfg = None
    try:
        if args.command != "selfcheck":
            cfg = load_config(args.config, args.overrides, args.only_datasets, args.only_tools)
        return COMMANDS[args.command](args, cfg)
    except Exception as exc:
        if type(exc).__name__ in EXPECTED_ERRORS:  # a clear, actionable message instead of a stack trace
            logger.error("%s: %s", type(exc).__name__, exc)
            return 2
        raise
