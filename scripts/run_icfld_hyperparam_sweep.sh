#!/usr/bin/env bash
###############################################################################
# IC-FLD Hyperparameter Sweep (Grid Search Wrapper)
#
# Delegates to scripts/optuna_icfld.py, which now implements a deterministic
# grid search without Optuna/MLflow. Pass the maximum number of evaluated
# combinations per dataset/function as the first argument (default: 20). Optionally
# provide a shuffle seed as the second argument to randomize trial order while
# keeping runs reproducible.
#   ./scripts/run_icfld_hyperparam_sweep.sh [TRIALS_PER_CONFIG] [SEED]
#
# Filtering:
#   export SWEEP_DATASETS="activity,mimic"
# will limit the sweep to those datasets before invoking this script.
###############################################################################

set -euo pipefail

TRIALS="${1:-20}"
SEED="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "IC-FLD Hyperparameter Sweep"
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
  --trainer icfld \
  --trials "$TRIALS" \
  --seed "$SEED"

echo ""
echo "=============================================="
echo "Sweep complete! Review the console output above for best runs."
echo "=============================================="
