#!/usr/bin/env python3
"""
Quick sanity-check training run for Kaggle.

Runs IC-FLD for a tiny 2-epoch schedule so you can verify the environment,
dataset mounts, and logging before launching longer Optuna sweeps.

Example (Notebook cell):

    !python scripts/run_kaggle_sanity.py \\
        --data-root /kaggle/input/icfld-data \\
        --dataset physionet
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_kaggle_optuna import (  # noqa: E402
    ROOT,
    DEFAULT_REQUIREMENTS,
    ensure_data_layout,
    install_requirements,
)

TRAIN_SCRIPT = ROOT / "FLD_ICC" / "train_FLD_ICC.py"

DEFAULTS = {
    "physionet": {"observation_time": 24},
    "mimic": {"observation_time": 24},
    "activity": {"observation_time": 3000},
    "ushcn": {"observation_time": 24},
}


def build_command(args: argparse.Namespace) -> list[str]:
    cfg = DEFAULTS.get(args.dataset, {})
    observation_time = args.observation_time or cfg.get("observation_time", 24)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "-d",
        args.dataset,
        "-ot",
        str(observation_time),
        "-bs",
        str(args.batch_size),
        "-fn",
        args.function,
        "-ed",
        str(args.embedding_dim),
        "-ld",
        str(args.latent_dim),
        "-nh",
        str(args.num_heads),
        "--depth",
        str(args.depth),
        "--epochs",
        str(args.epochs),
        "--early-stop",
        str(args.early_stop),
        "--seed",
        str(args.seed),
    ]

    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
    if args.wd is not None:
        cmd.extend(["--wd", str(args.wd)])
    if args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])
    if args.no_tensorboard:
        # nothing to add
        pass
    else:
        cmd.append("--tbon")
    if args.fld_report:
        cmd.extend(
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
    if args.extra_args:
        cmd.extend(args.extra_args)

    return cmd


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a quick 2-epoch IC-FLD sanity check on Kaggle.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/kaggle/input"),
        help="Directory containing dataset folders (default: /kaggle/input).",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip pip install for requirements.txt.",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help=f"Requirements file to install (default: {DEFAULT_REQUIREMENTS}).",
    )
    parser.add_argument(
        "--dataset",
        choices=["physionet", "mimic", "activity", "ushcn"],
        default="physionet",
        help="Dataset name to train on.",
    )
    parser.add_argument(
        "--function",
        choices=["C", "L", "Q", "S"],
        default="L",
        help="Basis function to use.",
    )
    parser.add_argument(
        "--observation-time",
        type=int,
        default=None,
        help="Override observation window (default depends on dataset).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for the quick run.",
    )
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--early-stop", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help='Value for "--gpu" CLI flag (set to "0" on Kaggle if you enabled GPU).',
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Do not enable TensorBoard logging.",
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
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to train_FLD_ICC.py (prefix with '--').",
    )

    args = parser.parse_args(argv)

    if not args.no_install:
        install_requirements(args.requirements)
    else:
        print("[pip] Skipping dependency installation (--no-install).")

    if args.data_root:
        ensure_data_layout(args.data_root)

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found at {TRAIN_SCRIPT}")

    if args.fld_report:
        print(f"[fld] FLD reporter enabled with tasks={args.fld_tasks}, scale={args.fld_scale}.")
    else:
        print("[fld] FLD reporter disabled; using base metrics (matches tPatchGNN).")

    cmd = build_command(args)
    print("\n[run] " + " ".join(cmd))
    env = os.environ.copy()
    if args.gpu is None:
        # Force CPU if user did not request GPU.
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
    subprocess.check_call(cmd, env=env)


if __name__ == "__main__":
    main()
