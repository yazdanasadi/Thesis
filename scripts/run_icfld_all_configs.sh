#!/usr/bin/env bash
###############################################################################
# Run IC-FLD on All Dataset/Function Combinations
#
# This script runs IC-FLD training with fixed hyperparameters across all
# datasets (physionet, mimic, activity, ushcn) and all basis functions (C, L, Q, S).
#
# Usage:
#   ./scripts/run_icfld_all_configs.sh
#
# Total runs: 4 datasets × 4 functions = 16 training runs
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Separate directories for full training runs
TB_DIR="${PROJECT_ROOT}/runs_icfld_full"
MODELS_DIR="${PROJECT_ROOT}/FLD_ICC/saved_models_full"

mkdir -p "$TB_DIR"
mkdir -p "$MODELS_DIR"

echo "=============================================="
echo "IC-FLD Full Training - All Configurations"
echo "=============================================="
echo "TensorBoard logs:  $TB_DIR"
echo "Model checkpoints: $MODELS_DIR"
echo "=============================================="
echo ""

# Fixed hyperparameters (reasonable defaults from search space)
LATENT_DIM=128
NUM_HEADS=4
EMBED_PER_HEAD=4
EMBEDDING_DIM=$((NUM_HEADS * EMBED_PER_HEAD))  # 16
DEPTH=2
LR="1e-4"
WD="1e-3"
EPOCHS=2
EARLY_STOP=15
BATCH_SIZE=32

# Dataset configurations: dataset, observation_time, use_ushcn_trainer
DATASETS=(
    "physionet 24 0"
    "mimic 24 0"
    "activity 3000 0"
    "ushcn 24 1"
)

# Basis functions
FUNCTIONS=("C" "L" "Q" "S")

cd "$PROJECT_ROOT"

# Counter for progress tracking
TOTAL_CONFIGS=$((${#DATASETS[@]} * ${#FUNCTIONS[@]}))
CURRENT=0

echo "Hyperparameters:"
echo "  Latent dim:    $LATENT_DIM"
echo "  Num heads:     $NUM_HEADS"
echo "  Embed per head: $EMBED_PER_HEAD"
echo "  Embedding dim: $EMBEDDING_DIM"
echo "  Depth:         $DEPTH"
echo "  Learning rate: $LR"
echo "  Weight decay:  $WD"
echo "  Epochs:        $EPOCHS (early stop: $EARLY_STOP)"
echo "  Batch size:    $BATCH_SIZE"
echo ""

for dataset_config in "${DATASETS[@]}"; do
    read -r DATASET OBS_TIME USE_USHCN <<< "$dataset_config"

    # Select appropriate trainer
    if [ "$USE_USHCN" -eq 1 ]; then
        TRAINER="FLD_ICC/train_FLD_ICC_ushcn.py"
        TRAINER_NAME="train_FLD_ICC_ushcn.py"
    else
        TRAINER="FLD_ICC/train_FLD_ICC.py"
        TRAINER_NAME="train_FLD_ICC.py"
    fi

    for FUNCTION in "${FUNCTIONS[@]}"; do
        CURRENT=$((CURRENT + 1))
        RUN_NAME="icfld_${DATASET}_${FUNCTION}"
        LOGDIR="${TB_DIR}/${RUN_NAME}"

        echo ""
        echo "=============================================="
        echo "[$CURRENT/$TOTAL_CONFIGS] Running: $RUN_NAME"
        echo "=============================================="
        echo "  Dataset:    $DATASET"
        echo "  Function:   $FUNCTION"
        echo "  Trainer:    $TRAINER_NAME"
        echo "  Obs time:   $OBS_TIME"
        echo "  Log dir:    $LOGDIR"
        echo "----------------------------------------------"

        # Build and run command
        python "$TRAINER" \
            --dataset "$DATASET" \
            --observation-time "$OBS_TIME" \
            --batch-size "$BATCH_SIZE" \
            --function "$FUNCTION" \
            --latent-dim "$LATENT_DIM" \
            --num-heads "$NUM_HEADS" \
            --embedding-dim "$EMBEDDING_DIM" \
            --depth "$DEPTH" \
            --epochs "$EPOCHS" \
            --early-stop "$EARLY_STOP" \
            --lr "$LR" \
            --wd "$WD" \
            --tbon \
            --logdir "$LOGDIR" \
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
echo "IC-FLD Full Training Complete!"
echo "=============================================="
echo "Total configurations: $TOTAL_CONFIGS"
echo "TensorBoard logs: $TB_DIR"
echo "Model checkpoints: $MODELS_DIR"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir $TB_DIR"
echo "=============================================="
