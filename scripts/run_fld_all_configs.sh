#!/usr/bin/env bash
###############################################################################
# Run FLD Baseline on All Dataset/Function Combinations
#
# This script runs FLD training with fixed hyperparameters across all
# datasets (physionet, mimic, activity, ushcn) and all basis functions (C, L, Q, S).
#
# Usage:
#   ./scripts/run_fld_all_configs.sh
#
# Total runs: 4 datasets × 4 functions = 16 training runs
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Separate directories for full training runs
TB_DIR="${PROJECT_ROOT}/runs_fld_full"
MODELS_DIR="${PROJECT_ROOT}/FLD/saved_models_full"

mkdir -p "$TB_DIR"
mkdir -p "$MODELS_DIR"

echo "=============================================="
echo "FLD Baseline Full Training - All Configurations"
echo "=============================================="
echo "TensorBoard logs:  $TB_DIR"
echo "Model checkpoints: $MODELS_DIR"
echo "=============================================="
echo ""

# Fixed hyperparameters (reasonable defaults from search space)
# NOTE: FLD's latent_dim is hardcoded to 20 in train_FLD.py, not configurable via CLI
NUM_HEADS=4
EMBED_PER_HEAD=4
DEPTH=2
LR="1e-4"
WD="1e-3"
EPOCHS=2
EARLY_STOP=30
BATCH_SIZE=32

# Dataset configurations: dataset, observation_time
DATASETS=(
    "physionet 24"
    "mimic 24"
    "activity 3000"
    "ushcn 24"
)

# Basis functions
FUNCTIONS=("C" "L" "Q" "S")

cd "$PROJECT_ROOT"

# Counter for progress tracking
TOTAL_CONFIGS=$((${#DATASETS[@]} * ${#FUNCTIONS[@]}))
CURRENT=0

echo "Hyperparameters:"
echo "  Latent dim:     20 (hardcoded in FLD)"
echo "  Num heads:     $NUM_HEADS"
echo "  Embed per head: $EMBED_PER_HEAD"
echo "  Depth:         $DEPTH"
echo "  Learning rate: $LR"
echo "  Weight decay:  $WD"
echo "  Epochs:        $EPOCHS (early stop: $EARLY_STOP)"
echo "  Batch size:    $BATCH_SIZE"
echo ""

for dataset_config in "${DATASETS[@]}"; do
    read -r DATASET OBS_TIME <<< "$dataset_config"

    for FUNCTION in "${FUNCTIONS[@]}"; do
        CURRENT=$((CURRENT + 1))
        RUN_NAME="fld_${DATASET}_${FUNCTION}"
        LOGDIR="${TB_DIR}/${RUN_NAME}"

        echo ""
        echo "=============================================="
        echo "[$CURRENT/$TOTAL_CONFIGS] Running: $RUN_NAME"
        echo "=============================================="
        echo "  Dataset:    $DATASET"
        echo "  Function:   $FUNCTION"
        echo "  Obs time:   $OBS_TIME"
        echo "  Log dir:    $LOGDIR"
        echo "----------------------------------------------"

        # Build and run command
        # NOTE: FLD has latent_dim hardcoded to 20 in train_FLD.py (line 92)
        python FLD/train_FLD.py \
            --dataset "$DATASET" \
            --observation-time "$OBS_TIME" \
            --batch-size "$BATCH_SIZE" \
            --function "$FUNCTION" \
            --num-heads "$NUM_HEADS" \
            --embedding-dim "$EMBED_PER_HEAD" \
            --depth "$DEPTH" \
            --epochs "$EPOCHS" \
            --early-stop "$EARLY_STOP" \
            --learn-rate "$LR" \
            --weight-decay "$WD" \
            --seed 42

        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Completed: $RUN_NAME"
        else
            echo "✗ Failed: $RUN_NAME (exit code: $EXIT_CODE)"
        fi
    done
done

echo ""
echo "=============================================="
echo "FLD Baseline Full Training Complete!"
echo "=============================================="
echo "Total configurations: $TOTAL_CONFIGS"
echo "TensorBoard logs: $TB_DIR"
echo "Model checkpoints: $MODELS_DIR"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir $TB_DIR"
echo "=============================================="
