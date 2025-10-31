#!/bin/bash
# Benchmark FLD vs IC-FLD on PhysioNet Dataset
# Runs all basis functions (C, L, Q, S) for 1000 epochs
# Logs metrics and timing information

set -e

# Configuration (can be overridden via environment variables)
EPOCHS=${EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-32}
OBSERVATION_TIME=${OBSERVATION_TIME:-24}
EARLY_STOP=${EARLY_STOP:-15}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-3}
SEED=${SEED:-42}
GPU=${GPU:-0}

DATASET="physionet"
FUNCTIONS=("C" "L" "Q" "S")
LOG_DIR="benchmark_results_physionet_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}FLD vs IC-FLD Benchmark - PhysioNet${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo "  Dataset:          $DATASET"
echo "  Epochs:           $EPOCHS"
echo "  Batch Size:       $BATCH_SIZE"
echo "  Observation Time: $OBSERVATION_TIME hours"
echo "  Early Stop:       $EARLY_STOP epochs"
echo "  Learning Rate:    $LEARNING_RATE"
echo "  Weight Decay:     $WEIGHT_DECAY"
echo "  Seed:             $SEED"
echo "  GPU:              $GPU"
echo "  Log Directory:    $LOG_DIR"
echo ""
echo -e "${YELLOW}Functions to test: ${FUNCTIONS[*]}${NC}"
echo ""

# Progress tracking
TOTAL_RUNS=$((${#FUNCTIONS[@]} * 2))
CURRENT_RUN=0

# Arrays to store results
declare -A FLD_TIMES FLD_METRICS
declare -A ICFLD_TIMES ICFLD_METRICS

#==============================================================================
# Function: Run Training and Extract Metrics
#==============================================================================
run_training() {
    local MODEL_TYPE=$1
    local FUNCTION=$2
    local TRAINER_SCRIPT=$3
    shift 3
    local EXTRA_ARGS=("$@")

    ((CURRENT_RUN++))
    local PROGRESS=$(awk "BEGIN {printf \"%.1f\", ($CURRENT_RUN / $TOTAL_RUNS) * 100}")

    echo ""
    echo -e "${GREEN}[$CURRENT_RUN/$TOTAL_RUNS - $PROGRESS%] Running $MODEL_TYPE with function '$FUNCTION'...${NC}"
    echo "----------------------------------------"

    local LOG_FILE="$LOG_DIR/${MODEL_TYPE}_${FUNCTION}.log"
    local METRICS_FILE="$LOG_DIR/${MODEL_TYPE}_${FUNCTION}_metrics.json"

    # Build argument list (FLD uses short forms, IC-FLD uses long forms)
    if [[ "$TRAINER_SCRIPT" == *"FLD/train_FLD.py" ]]; then
        # FLD-specific argument format
        local ARGS=(
            "$TRAINER_SCRIPT"
            -d "$DATASET"
            -ot "$OBSERVATION_TIME"
            -bs "$BATCH_SIZE"
            -e "$EPOCHS"
            -es "$EARLY_STOP"
            -fn "$FUNCTION"
            -lr "$LEARNING_RATE"
            -wd "$WEIGHT_DECAY"
            -s "$SEED"
            --gpu "$GPU"
            --tbon
            --logdir "runs/${MODEL_TYPE}_${FUNCTION}_physionet"
        )
    else
        # IC-FLD argument format
        local ARGS=(
            "$TRAINER_SCRIPT"
            -d "$DATASET"
            -ot "$OBSERVATION_TIME"
            -bs "$BATCH_SIZE"
            --epochs "$EPOCHS"
            --early-stop "$EARLY_STOP"
            -fn "$FUNCTION"
            --lr "$LEARNING_RATE"
            --wd "$WEIGHT_DECAY"
            --seed "$SEED"
            --gpu "$GPU"
            --tbon
            --logdir "runs/${MODEL_TYPE}_${FUNCTION}_physionet"
        )
    fi

    # Add model-specific arguments
    ARGS+=("${EXTRA_ARGS[@]}")

    # Start timer
    local START_TIME=$(date +%s)
    echo -e "${NC}Start time: $(date '+%Y-%m-%d %H:%M:%S')${NC}"

    # Run training and capture output (use -u for unbuffered output)
    python -u "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
    local EXIT_CODE=${PIPESTATUS[0]}

    # Stop timer
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))

    echo -e "${NC}End time:   $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${CYAN}Duration:   $DURATION seconds ($((DURATION / 60)) minutes)${NC}"

    # Extract metrics from last JSON line
    local JSON_LINE=$(grep -E '^\{.*\}$' "$LOG_FILE" | tail -1)

    if [ -n "$JSON_LINE" ]; then
        # Parse metrics using python
        python3 << EOF > "$METRICS_FILE"
import json
import sys
from datetime import datetime

metrics = $JSON_LINE
metrics['total_time_seconds'] = $DURATION
metrics['start_time'] = '$(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r $START_TIME '+%Y-%m-%d %H:%M:%S')'
metrics['end_time'] = '$(date -d @$END_TIME '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r $END_TIME '+%Y-%m-%d %H:%M:%S')'
metrics['function'] = '$FUNCTION'
metrics['model_type'] = '$MODEL_TYPE'

print(json.dumps(metrics, indent=2))
EOF

        # Display key metrics
        echo ""
        echo -e "${YELLOW}Key Metrics:${NC}"
        python3 << EOF
import json
with open('$METRICS_FILE', 'r') as f:
    m = json.load(f)
print(f"  Best Epoch:       {m.get('best_epoch', 'N/A')}")
print(f"  Val MSE (best):   {m.get('val_mse_best', 0):.6f}")
print(f"  Val RMSE (best):  {m.get('val_rmse_best', 0):.6f}")
print(f"  Val MAE (best):   {m.get('val_mae_best', 0):.6f}")
if m.get('test_mse_best'):
    print(f"  Test MSE (best):  {m['test_mse_best']:.6f}")
    print(f"  Test RMSE (best): {m['test_rmse_best']:.6f}")
    print(f"  Test MAE (best):  {m['test_mae_best']:.6f}")
print(f"  Total Time:       $DURATION seconds")
EOF

        # Store results
        if [ "$MODEL_TYPE" == "FLD" ]; then
            FLD_TIMES[$FUNCTION]=$DURATION
            FLD_METRICS[$FUNCTION]=$METRICS_FILE
        else
            ICFLD_TIMES[$FUNCTION]=$DURATION
            ICFLD_METRICS[$FUNCTION]=$METRICS_FILE
        fi

        return 0
    else
        echo -e "${YELLOW}Warning: Could not extract metrics from output${NC}"
        return 1
    fi
}

#==============================================================================
# Run FLD Benchmarks
#==============================================================================
echo ""
echo -e "${MAGENTA}========================================${NC}"
echo -e "${MAGENTA}Part 1: FLD Baseline${NC}"
echo -e "${MAGENTA}========================================${NC}"

for FUNCTION in "${FUNCTIONS[@]}"; do
    run_training "FLD" "$FUNCTION" "FLD/train_FLD.py" \
        -ed 4 -nh 4 -dp 2 || true
    sleep 2
done

#==============================================================================
# Run IC-FLD Benchmarks
#==============================================================================
echo ""
echo -e "${MAGENTA}========================================${NC}"
echo -e "${MAGENTA}Part 2: IC-FLD (Novel Model)${NC}"
echo -e "${MAGENTA}========================================${NC}"

for FUNCTION in "${FUNCTIONS[@]}"; do
    run_training "ICFLD" "$FUNCTION" "FLD_ICC/train_FLD_ICC.py" \
        -ld 128 -ed 64 -nh 4 --depth 2 || true
    sleep 2
done

#==============================================================================
# Generate Summary Report
#==============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}BENCHMARK COMPLETE - SUMMARY REPORT${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Summary table - Timing
echo -e "${YELLOW}Timing Summary (PhysioNet, $EPOCHS epochs):${NC}"
echo ""
printf "%-10s %-15s %-15s %-20s\n" "Function" "FLD (seconds)" "ICFLD (seconds)" "Speedup (FLD/ICFLD)"
printf "%-10s %-15s %-15s %-20s\n" "--------" "--------------" "----------------" "-------------------"

TOTAL_FLD_TIME=0
TOTAL_ICFLD_TIME=0

for FUNCTION in "${FUNCTIONS[@]}"; do
    FLD_TIME=${FLD_TIMES[$FUNCTION]:-"FAILED"}
    ICFLD_TIME=${ICFLD_TIMES[$FUNCTION]:-"FAILED"}

    if [[ $FLD_TIME =~ ^[0-9]+$ ]] && [[ $ICFLD_TIME =~ ^[0-9]+$ ]]; then
        SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $FLD_TIME / $ICFLD_TIME}")
        SPEEDUP_STR="${SPEEDUP}x"
        TOTAL_FLD_TIME=$((TOTAL_FLD_TIME + FLD_TIME))
        TOTAL_ICFLD_TIME=$((TOTAL_ICFLD_TIME + ICFLD_TIME))
    else
        SPEEDUP_STR="N/A"
    fi

    printf "%-10s %-15s %-15s %-20s\n" "$FUNCTION" "$FLD_TIME" "$ICFLD_TIME" "$SPEEDUP_STR"
done

printf "%-10s %-15s %-15s %-20s\n" "--------" "--------------" "----------------" "-------------------"
if [ $TOTAL_ICFLD_TIME -gt 0 ]; then
    TOTAL_SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $TOTAL_FLD_TIME / $TOTAL_ICFLD_TIME}")
else
    TOTAL_SPEEDUP="N/A"
fi
printf "%-10s %-15s %-15s %-20s\n" "TOTAL" "$TOTAL_FLD_TIME" "$TOTAL_ICFLD_TIME" "${TOTAL_SPEEDUP}x"

# Summary table - Performance
echo ""
echo -e "${YELLOW}Performance Summary:${NC}"
echo ""
printf "%-10s %-8s %-15s %-15s %-15s %-15s\n" "Function" "Model" "Val MSE" "Test MSE" "Val MAE" "Test MAE"
printf "%-10s %-8s %-15s %-15s %-15s %-15s\n" "--------" "-----" "-------" "--------" "-------" "--------"

for FUNCTION in "${FUNCTIONS[@]}"; do
    for MODEL_TYPE in FLD ICFLD; do
        if [ "$MODEL_TYPE" == "FLD" ]; then
            METRICS_FILE=${FLD_METRICS[$FUNCTION]}
        else
            METRICS_FILE=${ICFLD_METRICS[$FUNCTION]}
        fi

        if [ -f "$METRICS_FILE" ]; then
            python3 << EOF
import json
with open('$METRICS_FILE', 'r') as f:
    m = json.load(f)
val_mse = m.get('val_mse_best', 0)
test_mse = m.get('test_mse_best', None)
val_mae = m.get('val_mae_best', 0)
test_mae = m.get('test_mae_best', None)
print(f"{'$FUNCTION':<10} {'$MODEL_TYPE':<8} {val_mse:<15.6f} {test_mse if test_mse else 'N/A':<15} {val_mae:<15.6f} {test_mae if test_mae else 'N/A':<15}")
EOF
        else
            printf "%-10s %-8s %-15s %-15s %-15s %-15s\n" "$FUNCTION" "$MODEL_TYPE" "FAILED" "FAILED" "FAILED" "FAILED"
        fi
    done
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}Files Generated:${NC}"
echo ""
echo "Logs directory: $LOG_DIR"
echo ""
ls -lh "$LOG_DIR"

echo ""
echo -e "${GREEN}========================================${NC}"

# Save summary to CSV
CSV_FILE="$LOG_DIR/summary.csv"
echo "Model,Function,BestEpoch,ValMSE,ValRMSE,ValMAE,TestMSE,TestRMSE,TestMAE,TotalTimeSeconds,TotalTimeMinutes" > "$CSV_FILE"

for FUNCTION in "${FUNCTIONS[@]}"; do
    for MODEL_TYPE in FLD ICFLD; do
        if [ "$MODEL_TYPE" == "FLD" ]; then
            METRICS_FILE=${FLD_METRICS[$FUNCTION]}
        else
            METRICS_FILE=${ICFLD_METRICS[$FUNCTION]}
        fi

        if [ -f "$METRICS_FILE" ]; then
            python3 << EOF >> "$CSV_FILE"
import json
with open('$METRICS_FILE', 'r') as f:
    m = json.load(f)
time_sec = m.get('total_time_seconds', 0)
time_min = round(time_sec / 60, 2)
print(f"$MODEL_TYPE,$FUNCTION,{m.get('best_epoch', 0)},{m.get('val_mse_best', 0):.6f},{m.get('val_rmse_best', 0):.6f},{m.get('val_mae_best', 0):.6f},{m.get('test_mse_best', 0) if m.get('test_mse_best') else 'N/A'},{m.get('test_rmse_best', 0) if m.get('test_rmse_best') else 'N/A'},{m.get('test_mae_best', 0) if m.get('test_mae_best') else 'N/A'},{time_sec},{time_min}")
EOF
        fi
    done
done

echo -e "${CYAN}CSV summary saved to: $CSV_FILE${NC}"
echo ""
echo -e "${GREEN}Benchmark completed successfully!${NC}"
TOTAL_MINUTES=$(awk "BEGIN {printf \"%.2f\", ($TOTAL_FLD_TIME + $TOTAL_ICFLD_TIME) / 60}")
echo -e "${CYAN}Total execution time: $TOTAL_MINUTES minutes${NC}"
