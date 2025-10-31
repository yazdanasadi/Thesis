#!/usr/bin/env bash
###############################################################################
# FLD Hyperparameter Sweep (Grid Search Wrapper)
#
# Delegates to scripts/optuna_icfld.py with --trainer=fld so the new deterministic
# grid search handles classic FLD experiments without Optuna/MLflow.
# Usage:
#   ./scripts/run_fld_hyperparam_sweep.sh [TRIALS_PER_CONFIG] [SEED]
# The optional second argument controls shuffle order for reproducibility.
#
# To restrict datasets, set SWEEP_DATASETS (e.g., "physionet,mimic") before
# invoking this script.
###############################################################################

set -euo pipefail

TRIALS="${1:-20}"
SEED="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "FLD Hyperparameter Sweep"
echo "=============================================="
echo "Trials per config : $TRIALS"
echo "Shuffle seed      : $SEED"
if [[ -n "${SWEEP_DATASETS:-}" ]]; then
  echo "Dataset filter    : $SWEEP_DATASETS"
else
  echo "Dataset filter    : (all defaults)"
fi
echo "=============================================="
echo ""

cd "$PROJECT_ROOT"

python "$PROJECT_ROOT/scripts/optuna_icfld.py" \
  --trainer fld \
  --trials "$TRIALS" \
  --seed "$SEED"

echo ""
echo "=============================================="
echo "Sweep complete! Review the console output above for best runs."
echo "=============================================="
