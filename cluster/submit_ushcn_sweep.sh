#!/bin/bash
#SBATCH --job-name=ushcn_sweep
#SBATCH --output=logs/ushcn_sweep_%j.out
#SBATCH --error=logs/ushcn_sweep_%j.err
#SBATCH --mail-user=asadi@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00

TRAINER_KIND="${TRAINER_KIND:-icfld}"

TRIALS=${1:-20}
MLFLOW_DIR=${2:-}

set -euo pipefail

eval "$(conda shell.bash hook)"
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64}
conda activate thesis

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export SWEEP_DATASETS="ushcn"

declare -a CMD
PY_SCRIPT=""
case "${TRAINER_KIND}" in
  icfld)
    PY_SCRIPT="scripts/optuna_icfld.py"
    CMD=(python -u "${PY_SCRIPT}" --trainer icfld --trials "${TRIALS}")
    ;;
  fld)
    PY_SCRIPT="scripts/optuna_fld.py"
    CMD=(python -u "${PY_SCRIPT}" --full-grid --trials "${TRIALS}")
    ;;
  both)
    PY_SCRIPT="scripts/optuna_icfld.py"
    CMD=(python -u "${PY_SCRIPT}" --trainer both --trials "${TRIALS}")
    ;;
  ushcn)
    PY_SCRIPT="scripts/optuna_ushcn.py"
    CMD=(python -u "${PY_SCRIPT}" --trainer icfld --trials "${TRIALS}")
    ;;
  ushcn_fld)
    PY_SCRIPT="scripts/optuna_ushcn.py"
    CMD=(python -u "${PY_SCRIPT}" --trainer fld --trials "${TRIALS}")
    ;;
  *)
    echo "TRAINER_KIND='${TRAINER_KIND}' is invalid. Use icfld, fld, both, ushcn, or ushcn_fld." >&2
    exit 1
    ;;
esac

if [[ -n "${MLFLOW_DIR}" && "${PY_SCRIPT}" != "scripts/optuna_ushcn.py" ]]; then
  CMD+=(--mlflow-dir "${MLFLOW_DIR}")
fi

echo "Running sweep for USHCN with TRAINER_KIND=${TRAINER_KIND} (trials=${TRIALS})"
echo "Command: ${CMD[*]}"
srun "${CMD[@]}"
