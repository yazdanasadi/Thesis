#!/usr/bin/env python3
"""
Aggregate best hyperparameters from MLflow folders and export to JSON.

Expected directory layout (per dataset):
    mlflows/
        activity/mlruns/
        mimic/mlruns/
        physionet/mlruns/
        ushcn/mlruns/

Usage:
    python scripts/export_best_hparams.py \
        --mlflows-root mlflows \
        --output best_hparams.json
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Any

import mlflow
from mlflow.tracking import MlflowClient


def _parse_best_params(raw: str | None) -> Dict[str, Any] | str | None:
    if raw is None:
        return None
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return parsed
        return parsed
    except Exception:
        return raw


def _infer_function(run_name: str | None) -> str | None:
    if not run_name:
        return None
    parts = run_name.split("_")
    if len(parts) >= 2:
        return parts[-2].upper()
    return run_name


def _infer_trainer(run_name: str | None) -> str | None:
    if not run_name:
        return None
    prefix = run_name.split("_")[0].lower()
    if "fld" in prefix and not prefix.startswith("ic"):
        return "fld"
    return "icfld"


def collect_best_runs(tracking_dir: Path) -> Dict[str, Dict[str, Any]]:
    tracking_uri = f"file:{tracking_dir.resolve()}"
    client = MlflowClient(tracking_uri=tracking_uri)

    dataset_summary: Dict[str, Dict[str, Any]] = {}

    experiments = client.list_experiments()
    for exp in experiments:
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string="attributes.status = 'FINISHED' AND tags.mlflow.parentRunId = ''",
            max_results=500,
        )
        for run in runs:
            run_name = run.info.run_name or run.data.tags.get("mlflow.runName")
            function = _infer_function(run_name) or exp.name
            trainer = _infer_trainer(run_name)

            best_params = _parse_best_params(run.data.params.get("best_params"))
            best_metric = run.data.metrics.get("best_val_mse")

            record = {
                "trainer": trainer,
                "experiment_id": run.info.experiment_id,
                "run_id": run.info.run_id,
                "run_name": run_name,
                "best_params": best_params,
                "best_val_mse": best_metric,
            }

            if function not in dataset_summary:
                dataset_summary[function] = record
            else:
                existing = dataset_summary[function]
                existing_metric = existing.get("best_val_mse")
                if existing_metric is None or (best_metric is not None and best_metric < existing_metric):
                    dataset_summary[function] = record

    return dataset_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export best hyperparameters per dataset/function.")
    parser.add_argument(
        "--mlflows-root",
        type=Path,
        default=Path("mlflows"),
        help="Root directory containing dataset subfolders with mlruns/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("best_hparams.json"),
        help="Path to output JSON file.",
    )

    args = parser.parse_args()
    mlflows_root: Path = args.mlflows_root
    output_path: Path = args.output

    if not mlflows_root.exists():
        raise FileNotFoundError(f"mlflows root '{mlflows_root}' does not exist.")

    summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset_dir in sorted(mlflows_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        tracking_dir = dataset_dir / "mlruns"
        if not tracking_dir.exists():
            continue

        dataset_name = dataset_dir.name
        summary[dataset_name] = collect_best_runs(tracking_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[export] Wrote best hyperparameters to {output_path}")


if __name__ == "__main__":
    main()
