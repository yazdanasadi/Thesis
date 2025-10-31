#!/usr/bin/env python3
"""
Grid-search based hyperparameter sweep dedicated to the USHCN dataset.

This replaces the previous Optuna/MLflow-powered harness with a deterministic
enumeration of hand-picked parameter grids tailored to USHCN runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_trainer(relative: str) -> Path:
    parts = Path(relative)
    candidates = [
        SCRIPT_DIR / parts,
        SCRIPT_DIR.parent / parts,
        (SCRIPT_DIR.parents[2] / parts) if len(SCRIPT_DIR.parents) > 2 else None,
        Path.cwd() / parts,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate trainer script '{relative}'.")


ICFLD_USHCN = _resolve_trainer("FLD_ICC/train_FLD_ICC_ushcn.py")
FLD_USHCN = _resolve_trainer("FLD/train_FLD.py")
PROJECT_ROOT = ICFLD_USHCN.parents[1]

# Hyperparameter grids tuned for USHCN runs.
ICFLD_LATENT_DIMS = [32, 64, 128, 256]
ICFLD_NUM_HEADS = [2, 4, 8]
ICFLD_DEPTH = [1, 2, 4]
ICFLD_EMBED_DIM = [16, 32, 64]  # explicit control instead of heads*embed
ICFLD_BATCH = [32, 48, 64]
ICFLD_EPOCHS = [200, 400]
ICFLD_EARLY_STOP = [20, 30, 40]
ICFLD_LR = [5e-4, 1e-3, 2e-3]
ICFLD_WD = [0.0, 1e-4, 1e-3]

FLD_EMBED_DIM = [32, 64, 128]
FLD_NUM_HEADS = [1, 2, 4]
FLD_DEPTH = [1, 2, 3]
FLD_BATCH = [32, 64]
FLD_EPOCHS = [100, 200, 300]
FLD_EARLY_STOP = [10, 20, 30]
FLD_LR = [5e-4, 1e-3, 2e-3, 5e-3]
FLD_WD = [0.0, 1e-4, 1e-3]

FUNCTIONS = ("C", "L", "Q", "S")

VAL_MSE_RE = re.compile(r"val_mse(?:_best)?:\s*([0-9.+-eE]+)")
VAL_RMSE_RE = re.compile(r"val_rmse(?:_best)?:\s*([0-9.+-eE]+)")
VAL_MAE_RE = re.compile(r"val_mae(?:_best)?:\s*([0-9.+-eE]+)")


def _run_trainer(cmd: Sequence[str], cwd: Path) -> list[str]:
    with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmp:
        result = subprocess.run(
            cmd,
            stdout=tmp,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd),
        )
        tmp_path = Path(tmp.name)

    if result.returncode != 0:
        raise RuntimeError(f"Trainer failed (exit {result.returncode}): {result.stderr}")

    try:
        return tmp_path.read_text().splitlines()
    finally:
        tmp_path.unlink(missing_ok=True)


def _parse_metrics(lines: Iterable[str]) -> dict:
    best_val = float("inf")
    best_rmse = None
    best_mae = None

    for raw in lines:
        line = raw.strip()
        mse_match = VAL_MSE_RE.search(line)
        if not mse_match:
            continue
        try:
            val = float(mse_match.group(1))
        except ValueError:
            continue

        rmse_val = None
        mae_val = None
        rmse_match = VAL_RMSE_RE.search(line)
        mae_match = VAL_MAE_RE.search(line)
        if rmse_match:
            try:
                rmse_val = float(rmse_match.group(1))
            except ValueError:
                rmse_val = None
        if mae_match:
            try:
                mae_val = float(mae_match.group(1))
            except ValueError:
                mae_val = None

        if val < best_val:
            best_val = val
            best_rmse = rmse_val
            best_mae = mae_val

    if best_val < float("inf"):
        return {
            "val_mse_best": best_val,
            "val_rmse_best": best_rmse,
            "val_mae_best": best_mae,
        }
    return {
        "val_mse_best": float("inf"),
        "val_rmse_best": None,
        "val_mae_best": None,
    }


def _prepare_trials(grid: list[dict], max_trials: int, seed: int | None) -> list[dict]:
    items = list(grid)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(items)
    if max_trials > 0:
        items = items[:max_trials]
    return items


def _print_summary(label: str, function: str, results: dict) -> None:
    print(f"\nResult summary for {label} [{function}]")
    print(f"  Trials evaluated: {results['evaluated']}/{results['total']}")
    print(f"  Failures: {results['failures']}")
    if results["best_params"] and results["best_metrics"]:
        best_val = results["best_metrics"].get("val_mse_best")
        print(f"  Best val_mse: {best_val}")
        print(f"  Best params: {json.dumps(results['best_params'], sort_keys=True)}")
    else:
        print("  No successful trials.")


def _run_grid(
    label: str,
    function: str,
    params_grid: list[dict],
    runner,
    max_trials: int,
    seed: int | None,
    extra_args: Sequence[str] | None,
) -> dict:
    trials = _prepare_trials(params_grid, max_trials, seed)
    total = len(trials)
    evaluated = 0
    failures = 0
    best_val = float("inf")
    best_params = None
    best_metrics = None

    print(f"\n[label={label}] function={function} -> evaluating up to {total} combinations")
    for idx, params in enumerate(trials, start=1):
        print(f"  [{idx}/{total}] params={params}")
        try:
            metrics = runner(function, params, extra_args)
        except Exception as exc:  # pragma: no cover - defensive logging
            failures += 1
            print(f"    ! trial failed: {exc}")
            continue

        val = metrics.get("val_mse_best")
        if val is None or not isinstance(val, (float, int)) or not math.isfinite(float(val)):
            failures += 1
            print("    ! no finite val_mse_best; skipping")
            continue

        evaluated += 1
        val = float(val)
        if val < best_val:
            best_val = val
            best_params = params.copy()
            best_metrics = metrics
            print(f"    ✓ new best val_mse={val:.6f}")
        else:
            print(f"    val_mse={val:.6f}")

    return {
        "evaluated": evaluated,
        "failures": failures,
        "total": total,
        "best_params": best_params,
        "best_metrics": best_metrics,
    }


def _icfld_grid() -> list[dict]:
    grid = []
    counter = 0
    for latent, heads, depth, embed, batch, epochs, early, lr, wd in itertools.product(
        ICFLD_LATENT_DIMS,
        ICFLD_NUM_HEADS,
        ICFLD_DEPTH,
        ICFLD_EMBED_DIM,
        ICFLD_BATCH,
        ICFLD_EPOCHS,
        ICFLD_EARLY_STOP,
        ICFLD_LR,
        ICFLD_WD,
    ):
        grid.append(
            {
                "latent_dim": latent,
                "num_heads": heads,
                "depth": depth,
                "embed_dim": embed,
                "batch_size": batch,
                "epochs": epochs,
                "early_stop": early,
                "lr": lr,
                "wd": wd,
                "seed": counter,
            }
        )
        counter += 1
    return grid


def _fld_grid() -> list[dict]:
    grid = []
    counter = 0
    for embed, heads, depth, batch, epochs, early, lr, wd in itertools.product(
        FLD_EMBED_DIM,
        FLD_NUM_HEADS,
        FLD_DEPTH,
        FLD_BATCH,
        FLD_EPOCHS,
        FLD_EARLY_STOP,
        FLD_LR,
        FLD_WD,
    ):
        grid.append(
            {
                "embed_dim": embed,
                "num_heads": heads,
                "depth": depth,
                "batch_size": batch,
                "epochs": epochs,
                "early_stop": early,
                "lr": lr,
                "wd": wd,
                "seed": counter,
            }
        )
        counter += 1
    return grid


def _run_icfld(function: str, params: dict, extra_args: Sequence[str] | None) -> dict:
    cmd = [
        "python",
        str(ICFLD_USHCN),
        "-d",
        "ushcn",
        "-fn",
        function,
        "-ld",
        str(params["latent_dim"]),
        "-nh",
        str(params["num_heads"]),
        "-ed",
        str(params["embed_dim"]),
        "-dp",
        str(params["depth"]),
        "-bs",
        str(params["batch_size"]),
        "-e",
        str(params["epochs"]),
        "-es",
        str(params["early_stop"]),
        "-lr",
        str(params["lr"]),
        "-wd",
        str(params["wd"]),
        "-s",
        str(params.get("seed", 0)),
    ]
    if extra_args:
        cmd.extend(extra_args)
    lines = _run_trainer(cmd, cwd=ICFLD_USHCN.parent)

    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    return _parse_metrics(lines)


def _run_fld(function: str, params: dict, extra_args: Sequence[str] | None) -> dict:
    cmd = [
        "python",
        str(FLD_USHCN),
        "-d",
        "ushcn",
        "-fn",
        function,
        "-nh",
        str(params["num_heads"]),
        "-ed",
        str(params["embed_dim"]),
        "-dp",
        str(params["depth"]),
        "-bs",
        str(params["batch_size"]),
        "-e",
        str(params["epochs"]),
        "-es",
        str(params["early_stop"]),
        "-lr",
        str(params["lr"]),
        "-wd",
        str(params["wd"]),
        "-s",
        str(params.get("seed", 0)),
    ]
    if extra_args:
        cmd.extend(extra_args)
    lines = _run_trainer(cmd, cwd=FLD_USHCN.parent)

    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    return _parse_metrics(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="USHCN-only grid search (IC-FLD + FLD variants).")
    parser.add_argument(
        "--trainer",
        choices=("icfld", "fld"),
        default="icfld",
        help="Select which USHCN trainer to optimize.",
    )
    parser.add_argument(
        "--functions",
        nargs="*",
        default=FUNCTIONS,
        help="Subset of basis functions to evaluate.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Maximum number of combinations to evaluate per function.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for shuffling the parameter grid (set to None for deterministic order).",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        dest="extra_args",
        default=None,
        help="Additional CLI arguments forwarded to the trainer (repeatable).",
    )

    args = parser.parse_args()
    extra_args = args.extra_args or []

    print("==============================================")
    print("USHCN Hyperparameter Sweep")
    print("==============================================")
    print(f"Trainer          : {args.trainer}")
    print(f"Trials per func  : {args.trials}")
    print(f"Shuffle seed     : {args.seed}")
    print(f"Functions        : {', '.join(args.functions)}")
    if extra_args:
        print(f"Extra trainer CLI: {extra_args}")
    print("==============================================")

    if args.trainer == "icfld":
        grid = _icfld_grid()
        runner = _run_icfld
        label = "IC-FLD (USHCN)"
    else:
        grid = _fld_grid()
        runner = _run_fld
        label = "FLD (USHCN)"

    for function in args.functions:
        results = _run_grid(
            label=label,
            function=function,
            params_grid=grid,
            runner=runner,
            max_trials=args.trials,
            seed=args.seed,
            extra_args=extra_args,
        )
        _print_summary(label, function, results)

    print("\nAll USHCN sweeps completed.")


if __name__ == "__main__":
    main()
