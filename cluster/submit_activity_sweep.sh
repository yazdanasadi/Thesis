#!/bin/bash
#SBATCH --job-name=activity_sweep
#SBATCH --output=logs/activity_sweep_%j.out
#SBATCH --error=logs/activity_sweep_%j.err
#SBATCH --mail-user=asadi@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00

TRAINER_KIND="${TRAINER_KIND:-icfld}"

TRIALS=${1:-20}
REPO_DIR="${REPO_DIR:-/home/asadi/MasterThesis-1}"

set -euo pipefail

eval "$(conda shell.bash hook)"
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64}
conda activate thesis

cd "$REPO_DIR"
mkdir -p logs
export SWEEP_DATASETS="activity"

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

echo "Running sweep for activity with TRAINER_KIND=${TRAINER_KIND} (trials=${TRIALS})"
echo "Command: ${CMD[*]}"
echo "Working directory: $PWD"
srun "${CMD[@]}"
