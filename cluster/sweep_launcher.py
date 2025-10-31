#!/usr/bin/env python3
"""Python orchestrator for dataset-specific sweep submit scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = REPO_ROOT / "cluster"
DATA_DIR = REPO_ROOT / "data"

DATASET_CHOICES = ("physionet", "mimic", "activity", "ushcn")
TRAINER_CHOICES = ("icfld", "fld", "both", "ushcn", "ushcn_fld")
DEFAULT_TRIALS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch dataset-specific sweeps via sbatch.")
    parser.add_argument("dataset", choices=DATASET_CHOICES, help="Dataset to sweep.")
    parser.add_argument(
        "--trainer",
        choices=TRAINER_CHOICES,
        default="icfld",
        help="Sweep trainer variant (icfld/fld/both). USHCN-specific values are ushcn/ushcn_fld.",
    )
    parser.add_argument("--trials", type=int, default=None, help="Override Optuna trial count.")
    parser.add_argument("--mlflow-dir", type=str, default=None, help="Write results to a custom MLflow folder.")
    return parser.parse_args()


def ensure_dataset_folder(dataset: str) -> None:
    path = DATA_DIR / dataset
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset folder '{dataset}' is missing under data/. (expected: {path})\n"
            "Sync the dataset before submitting the sweep."
        )


def resolve_script(dataset: str) -> Path:
    script = CLUSTER_DIR / f"submit_{dataset}_sweep.sh"
    if not script.exists():
        raise FileNotFoundError(f"Required submit script not found: {script}")
    return script


def build_command(script: Path, trials: int | None, mlflow_dir: str | None) -> List[str]:
    cmd = ["sbatch", str(script)]
    if trials is not None:
        cmd.append(str(trials))
    if mlflow_dir:
        cmd.append(mlflow_dir)
    return cmd


def validate_trainer(dataset: str, trainer: str) -> None:
    if dataset != "ushcn" and trainer in {"ushcn", "ushcn_fld"}:
        raise ValueError("USHCN-specific trainer modes are only valid when dataset=ushcn.")


def main() -> int:
    args = parse_args()
    validate_trainer(args.dataset, args.trainer)
    ensure_dataset_folder(args.dataset)

    script = resolve_script(args.dataset)
    trials = args.trials if args.trials is not None else DEFAULT_TRIALS
    cmd = build_command(script, trials, args.mlflow_dir)

    env = os.environ.copy()
    env["TRAINER_KIND"] = args.trainer

    print(f"Submitting sweep: {' '.join(cmd)} (TRAINER_KIND={args.trainer})")
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        print(f"sbatch exited with code {result.returncode}", file=sys.stderr)
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
