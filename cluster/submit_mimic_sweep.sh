#!/bin/bash
#SBATCH --job-name=mimic_sweep
#SBATCH --output=logs/mimic_sweep_%j.out
#SBATCH --error=logs/mimic_sweep_%j.err
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
export SWEEP_DATASETS="mimic"

declare -a CMD
case "${TRAINER_KIND}" in
  icfld)
    CMD=(python -u scripts/optuna_icfld.py --trainer icfld --trials "${TRIALS}")
    ;;
  fld)
    CMD=(python -u scripts/optuna_fld.py --full-grid --trials "${TRIALS}")
    ;;
  both)
    CMD=(python -u scripts/optuna_icfld.py --trainer both --trials "${TRIALS}")
    ;;
  *)
    echo "TRAINER_KIND='${TRAINER_KIND}' is invalid. Use icfld, fld, or both." >&2
    exit 1
    ;;
esac

if [[ -n "${MLFLOW_DIR}" ]]; then
  CMD+=(--mlflow-dir "${MLFLOW_DIR}")
fi

echo "Running sweep for mimic with TRAINER_KIND=${TRAINER_KIND} (trials=${TRIALS})"
echo "Command: ${CMD[*]}"
srun "${CMD[@]}"
