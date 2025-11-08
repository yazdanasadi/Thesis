"""Grid-based hyperparameter sweep for the classic FLD trainer."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

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


FLD_TRAIN_SCRIPT = _resolve_trainer("FLD/train_FLD.py")

# Hyperparameter grids (matching the Optuna categorical choices)
HEAD_VALUES = [4, 8]
DEPTH_VALUES = [2, 4]
EMB_PER_HEAD_VALUES = [2, 4, 8]
FLD_EPOCH_VALUES = [100, 200, 300]
FLD_EARLY_STOP_VALUES = [10, 20, 30]
FLD_BATCH_SIZE_VALUES = [16, 32, 64]
FUNCTIONS = ["C", "L", "Q", "S"]

VAL_MSE_RE = re.compile(r"val_mse(?:_best)?:\s*([0-9.+-eE]+)")
VAL_RMSE_RE = re.compile(r"val_rmse(?:_best)?:\s*([0-9.+-eE]+)")
VAL_MAE_RE = re.compile(r"val_mae(?:_best)?:\s*([0-9.+-eE]+)")


def _normalize_dataset_name(name: str) -> str:
    # Accept values like physionet, "physionet", 'physionet', and mixed case.
    return name.strip().strip("'\"").lower()


def _filter_plans_by_env(plans: Iterable[tuple[str, str, int]]) -> List[tuple[str, str, int]]:
    allowed_spec = os.environ.get("SWEEP_DATASETS")
    if not allowed_spec:
        return list(plans)

    allowed: set[str] = set()
    for raw in allowed_spec.split(","):
        normalized = _normalize_dataset_name(raw)
        if normalized:
            allowed.add(normalized)

    filtered: List[tuple[str, str, int]] = []
    for plan in plans:
        dataset_name = _normalize_dataset_name(plan[1])
        if dataset_name in allowed:
            filtered.append(plan)

    if not filtered:
        schedule_names = sorted({_normalize_dataset_name(plan[1]) for plan in plans})
        raise ValueError(
            f"No search plans remain after applying SWEEP_DATASETS={allowed_spec!r}. "
            f"Valid dataset names: {', '.join(schedule_names)}"
        )
    return filtered


def _run_subprocess(cmd: Sequence[str], cwd: Path) -> List[str]:
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
        raise RuntimeError(f"Training failed (exit {result.returncode}): {result.stderr}")

    try:
        return tmp_path.read_text().splitlines()
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_metrics_from_logs(lines: Iterable[str]) -> dict:
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


def run_fld_training(function: str, params: dict, extra_args: Iterable[str] | None = None) -> dict:
    lr_val = params.get("lr", 1e-4)
    wd_val = params.get("wd", 1e-3)
    cmd = [
        "python",
        str(FLD_TRAIN_SCRIPT),
        "--function",
        function,
        "--dataset",
        params.get("dataset", "physionet"),
        "--observation-time",
        str(params.get("history", 24)),
        "--epochs",
        str(params.get("epochs", 300)),
        "--early-stop",
        str(params.get("early_stop", 30)),
        "--batch-size",
        str(params.get("batch_size", 32)),
        "--embedding-dim",
        str(params["embed_per_head"]),
        "--num-heads",
        str(params["num_heads"]),
        "--depth",
        str(params["depth"]),
        "--learn-rate",
        str(lr_val),
        "--weight-decay",
        str(wd_val),
        "--seed",
        str(params.get("seed", 0)),
    ]

    if extra_args:
        cmd.extend(extra_args)

    lines = _run_subprocess(cmd, cwd=FLD_TRAIN_SCRIPT.parent)

    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break

    metrics = _extract_metrics_from_logs(lines)
    if not math.isfinite(metrics.get("val_mse_best", float("inf"))):
        metrics["val_mse_best"] = float("inf")
    return metrics


def _fld_param_grid(dataset: str, history: int, seed_base: int) -> List[dict]:
    grid: List[dict] = []
    counter = 0
    for num_heads, depth, embed, epochs, early, batch in itertools.product(
        HEAD_VALUES,
        DEPTH_VALUES,
        EMB_PER_HEAD_VALUES,
        FLD_EPOCH_VALUES,
        FLD_EARLY_STOP_VALUES,
        FLD_BATCH_SIZE_VALUES,
    ):
        params = {
            "num_heads": num_heads,
            "depth": depth,
            "embed_per_head": embed,
            "epochs": epochs,
            "early_stop": early,
            "batch_size": batch,
            "dataset": dataset,
            "history": history,
        }
        params["seed"] = seed_base + counter
        counter += 1
        grid.append(params)
    return grid


def _prepare_trials(
    grid: List[dict],
    max_trials: int | None,
    shuffle_seed: int | None,
) -> List[dict]:
    items = list(grid)
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(items)
    if max_trials is not None and max_trials > 0:
        items = items[: max_trials]
    return items


def _summarize_params(params: dict) -> str:
    core = {
        "num_heads": params.get("num_heads"),
        "embed": params.get("embed_per_head"),
        "depth": params.get("depth"),
        "epochs": params.get("epochs"),
        "batch": params.get("batch_size"),
        "early_stop": params.get("early_stop"),
    }
    filtered = {k: v for k, v in core.items() if v is not None}
    parts = [f"{k}={v}" for k, v in filtered.items()]
    return ", ".join(parts)


def _run_search(
    label: str,
    function: str,
    dataset: str,
    history: int,
    grid: List[dict],
    run_fn: Callable[[str, dict], dict],
    max_trials: int,
    shuffle_seed: int | None,
    extra_args: Iterable[str] | None = None,
) -> dict:
    trials = _prepare_trials(grid, max_trials, shuffle_seed)
    total = len(trials)
    best_val = float("inf")
    best_params: dict | None = None
    best_metrics: dict | None = None
    evaluated = 0
    failures = 0

    print(f"  -> Evaluating {total} combinations for {label} (function={function}, dataset={dataset}, history={history})")
    for idx, params in enumerate(trials, start=1):
        summary = _summarize_params(params)
        print(f"     [{idx}/{total}] {summary}")
        try:
            metrics = run_fn(function, params, extra_args=extra_args)
        except Exception as exc:  # pragma: no cover - defensive
            failures += 1
            print(f"        ! Trial failed: {exc}")
            continue

        val = metrics.get("val_mse_best")
        if val is None or not isinstance(val, (float, int)) or not math.isfinite(float(val)):
            failures += 1
            print("        ! No finite val_mse_best; skipping.")
            continue

        evaluated += 1
        val = float(val)
        if val < best_val:
            best_val = val
            best_params = params.copy()
            best_metrics = metrics
            print(f"        ✓ New best val_mse={val:.6f}")
        else:
            print(f"        val_mse={val:.6f}")

    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "evaluated": evaluated,
        "failures": failures,
        "total_trials": total,
        "label": label,
        "function": function,
        "dataset": dataset,
        "history": history,
    }


def run_fld_search(
    function: str,
    dataset: str,
    history: int,
    max_trials: int,
    *,
    seed: int | None = None,
    extra_args: Iterable[str] | None = None,
) -> dict:
    seed_base = (seed or 0) * 1000
    grid = _fld_param_grid(dataset, history, seed_base=seed_base)
    return _run_search(
        label="FLD",
        function=function,
        dataset=dataset,
        history=history,
        grid=grid,
        run_fn=run_fld_training,
        max_trials=max_trials,
        shuffle_seed=seed,
        extra_args=extra_args,
    )


def _print_summary(result: dict) -> None:
    label = result["label"]
    function = result["function"]
    dataset = result["dataset"]
    total = result["total_trials"]
    evaluated = result["evaluated"]
    failures = result["failures"]
    best_params = result["best_params"]
    best_metrics = result["best_metrics"]

    print(f"\nResult summary for {label} [{function} on {dataset}]")
    print(f"  Trials evaluated: {evaluated}/{total}")
    print(f"  Failures: {failures}")
    if best_params and best_metrics:
        print(f"  Best val_mse: {best_metrics.get('val_mse_best')}")
        print(f"  Best params: {json.dumps(best_params, sort_keys=True)}")
    else:
        print("  No successful trials.")
    print("")

DEFAULT_SEARCH_PLAN: List[Tuple[str, str, int]] = [
    ("C", "activity", 3000),
    ("L", "activity", 3000),
    ("S", "activity", 3000),
    ("Q", "activity", 3000),
    ("L", "mimic", 24),
    ("Q", "physionet", 24),
    ("S", "ushcn", 24),
]

DATASET_DEFAULT_HISTORY = {
    "activity": 3000,
    "mimic": 24,
    "physionet": 24,
    "ushcn": 24,
}


def _build_full_grid_plan() -> List[Tuple[str, str, int]]:
    plan: List[Tuple[str, str, int]] = []
    for dataset, history in DATASET_DEFAULT_HISTORY.items():
        for function in FUNCTIONS:
            plan.append((function, dataset, history))
    return plan


def _filter_plan(plan: Iterable[Tuple[str, str, int]]) -> List[Tuple[str, str, int]]:
    return _filter_plans_by_env(plan)


def run_sweeps(
    trials: int,
    search_plan: Iterable[Tuple[str, str, int]],
    seed: int | None = None,
) -> None:
    plan_list = _filter_plan(search_plan)
    total_trials = len(plan_list) * trials

    print(f"Starting FLD hyperparameter search: {len(plan_list)} configs, <= {trials} trials each")
    print(f"Upper bound on total trials: {total_trials}")
    if seed is not None:
        print(f"Shuffling trial order with seed={seed}")
    print()

    for idx, (function, dataset, history) in enumerate(plan_list, start=1):
        print(f"[{idx}/{len(plan_list)}] FLD sweep for function={function}, dataset={dataset}, history={history}")
        result = run_fld_search(
            function=function,
            dataset=dataset,
            history=history,
            max_trials=trials,
            seed=seed,
        )
        _print_summary(result)

    print("All FLD hyperparameter searches completed!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLD hyperparameter search without Optuna/MLflow.")
    parser.add_argument("--trials", type=int, default=20, help="Maximum trial evaluations per dataset/function pair")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed to shuffle trial order (shared across datasets).",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Sweep all dataset/function combinations (default plan uses four representative configs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.full_grid:
        search_plan: Iterable[Tuple[str, str, int]] = _build_full_grid_plan()
    else:
        search_plan = DEFAULT_SEARCH_PLAN

    run_sweeps(
        trials=args.trials,
        search_plan=search_plan,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
