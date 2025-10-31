#!/usr/bin/env python3
"""
Kaggle-friendly entry point for running the IC-FLD/FLD hyperparameter sweep
without Optuna/MLflow.

Example (Notebook cell):

    !python scripts/run_kaggle_optuna.py \\
        --data-root /kaggle/input/icfld-data \\
        --trials 10

Expected dataset layout inside ``--data-root``::

    <root>/
        physionet/
        mimic/
        activity/
        ushcn/

The script will create lightweight symlinks into ``./data`` so the training
code can discover the datasets via the default relative paths.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
DEFAULT_REQUIREMENTS = ROOT / "requirements.txt"

DEFAULT_DATASETS: Tuple[Tuple[str, int], ...] = (
    ("physionet", 24),
    ("mimic", 24),
    ("activity", 3000),
    ("ushcn", 24),
)

DEFAULT_FUNCTIONS: Tuple[str, ...] = ("C", "L", "Q", "S")

DEFAULT_SEARCH_PLANS: Tuple[Tuple[str, str, int], ...] = tuple(
    (fn, dataset, history)
    for dataset, history in DEFAULT_DATASETS
    for fn in DEFAULT_FUNCTIONS
)


def parse_plan_tokens(tokens: Sequence[str]) -> List[Tuple[str, str, int]]:
    """
    Parse CLI tokens of the form ``FN:DATASET:HISTORY`` into a list of tuples.
    Falls back to DEFAULT_SEARCH_PLANS if no tokens are provided.
    """
    if not tokens:
        return list(DEFAULT_SEARCH_PLANS)

    plans = []
    for token in tokens:
        try:
            fn, dataset, history = token.split(":")
            plans.append((fn.upper(), dataset.lower(), int(history)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid plan '{token}'. Use FUNCTION:dataset:history (e.g. Q:physionet:24)"
            ) from exc

    return plans


def ensure_data_layout(data_root: Path) -> None:
    """
    Ensure ``./data/<dataset>`` exists (symlink or directory) for known datasets.
    """
    DATA_DIR.mkdir(exist_ok=True)
    expected = ["physionet", "mimic", "activity", "ushcn"]

    for name in expected:
        target = DATA_DIR / name
        if target.exists():
            continue
        candidate = data_root / name
        if not candidate.exists():
            print(f"[data] Missing '{candidate}'. Skipping symlink for '{name}'.")
            continue
        try:
            os.symlink(candidate, target)
            print(f"[data] Linked {candidate} -> {target}")
        except FileExistsError:
            pass


def install_requirements(requirements: Path) -> None:
    """
    Install python packages inside the Kaggle session.
    """
    if not requirements.exists():
        print(f"[pip] Requirements file '{requirements}' not found; skipping install.")
        return

    print(f"[pip] Installing dependencies from {requirements} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


def run_studies(
    plans: Iterable[Tuple[str, str, int]],
    trials: int,
    study_prefix: str | None,
    trainer: str,
    extra_cli_args: Sequence[str] | None = None,
    seed: int | None = None,
) -> None:
    """
    Execute lightweight grid searches for each plan.
    """
    import optuna_icfld
    default_prefix = "fld" if trainer == "fld" else "icfld"
    prefix = study_prefix or default_prefix

    for function, dataset, history in plans:
        study_name = f"{prefix}_{function}_{dataset}"
        print(f"\n=== Study: {study_name} (history={history}, max_trials={trials}) ===")

        if trainer == "fld":
            result = optuna_icfld.run_fld_search(
                function=function,
                dataset=dataset,
                history=history,
                max_trials=trials,
                seed=seed,
                extra_args=extra_cli_args,
            )
        else:
            result = optuna_icfld.run_icfld_search(
                function=function,
                dataset=dataset,
                history=history,
                max_trials=trials,
                seed=seed,
                extra_args=extra_cli_args,
            )

        optuna_icfld._print_summary(result)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the IC-FLD/FLD hyperparameter sweep on Kaggle.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/kaggle/input"),
        help="Directory that stores the raw datasets (default: /kaggle/input).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Maximum trial evaluations per (function, dataset) plan.",
    )
    parser.add_argument(
        "--plan",
        nargs="*",
        default=None,
        help="Override search plans; each token is FUNCTION:dataset:history (e.g. Q:physionet:24).",
    )
    parser.add_argument(
        "--study-prefix",
        type=str,
        default=None,
        help="Optional prefix for report labels (default: icfld).",
    )
    parser.add_argument(
        "--trainer",
        choices=["icfld", "fld"],
        default="icfld",
        help="Select which trainer to optimize (icfld = IC-FLD w/ invertible cores, fld = original FLD).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for shuffling trial order.",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip pip install for requirements.txt (use when you built a custom image).",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help=f"Path to requirements file (default: {DEFAULT_REQUIREMENTS}).",
    )
    parser.add_argument(
        "--fld-report",
        action="store_true",
        help="Enable FLD TSDM reporting (--fldReport). Default runs tPatchGNN-style metrics only.",
    )
    parser.add_argument(
        "--fld-tasks",
        type=str,
        default="75-3,75-25,50-50",
        help="Comma-separated tasks to evaluate when --fld-report is set.",
    )
    parser.add_argument(
        "--fld-scale",
        type=str,
        default="zscore",
        help="Scaling mode for FLD reporter when enabled (zscore|minmax|none|auto).",
    )
    parser.add_argument(
        "--fld-stats-from",
        type=str,
        default="obs+targets",
        help="Source data for FLD stats when --fld-report is enabled.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs passed to train_FLD_ICC.py (default: script-defined).",
    )

    args = parser.parse_args(argv)

    plans = parse_plan_tokens(args.plan)

    if not args.no_install:
        install_requirements(args.requirements)
    else:
        print("[pip] Skipping dependency installation (--no-install).")

    if args.data_root:
        ensure_data_layout(args.data_root)

    extra_cli_args: List[str] = []
    if args.fld_report:
        extra_cli_args.extend(
            [
                "--fldReport",
                "--fldTasks",
                args.fld_tasks,
                "--fldScale",
                args.fld_scale,
                "--fldStatsFrom",
                args.fld_stats_from,
            ]
        )
        print(f"[fld] FLD reporter enabled with tasks={args.fld_tasks}, scale={args.fld_scale}.")
    else:
        print("[fld] FLD reporter disabled; using base metrics (matches tPatchGNN).")

    if args.epochs is not None:
        extra_cli_args.extend(["--epochs", str(args.epochs)])
        print(f"[train] Overriding epochs to {args.epochs}.")

    run_studies(
        plans,
        trials=args.trials,
        study_prefix=args.study_prefix,
        trainer=args.trainer,
        extra_cli_args=extra_cli_args,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
