"""Grid-based hyperparameter sweep for the classic FLD trainer."""
from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple

import optuna_icfld as base

DEFAULT_SEARCH_PLAN: List[Tuple[str, str, int]] = [
    ("C", "activity", 3000),
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
        for function in base.FUNCTIONS:
            plan.append((function, dataset, history))
    return plan


def _filter_plan(plan: Iterable[Tuple[str, str, int]]) -> List[Tuple[str, str, int]]:
    return base._filter_plans_by_env(plan)


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
        result = base.run_fld_search(
            function=function,
            dataset=dataset,
            history=history,
            max_trials=trials,
            seed=seed,
        )
        base._print_summary(result)

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
