#!/bin/bash
# Benchmark IC-FLD training with and without optimizations
# Usage: ./scripts/benchmark_optimizations.sh

set -e

DATASET="physionet"
BS=32
EPOCHS=5
FUNCTION="L"
LD=128
ED=64
NH=4
DEPTH=2

echo "========================================"
echo "IC-FLD Optimization Benchmark"
echo "========================================"
echo "Dataset: $DATASET"
echo "Batch size: $BS"
echo "Epochs: $EPOCHS (short run for benchmarking)"
echo ""

# Baseline
echo "[1/2] Running BASELINE (train_FLD_ICC.py)..."
echo "----------------------------------------"
START_BASELINE=$(date +%s)
python FLD_ICC/train_FLD_ICC.py \
  -d $DATASET \
  -bs $BS \
  --epochs $EPOCHS \
  -fn $FUNCTION \
  -ld $LD \
  -ed $ED \
  -nh $NH \
  --depth $DEPTH \
  --lr 1e-4 \
  --wd 1e-3 \
  --seed 42 \
  2>&1 | tee benchmark_baseline.log
END_BASELINE=$(date +%s)
BASELINE_TIME=$((END_BASELINE - START_BASELINE))

echo ""
echo "[2/2] Running OPTIMIZED (train_FLD_ICC_optimized.py)..."
echo "----------------------------------------"
START_OPT=$(date +%s)
python FLD_ICC/train_FLD_ICC_optimized.py \
  -d $DATASET \
  -bs $BS \
  --epochs $EPOCHS \
  -fn $FUNCTION \
  -ld $LD \
  -ed $ED \
  -nh $NH \
  --depth $DEPTH \
  --lr 1e-4 \
  --wd 1e-3 \
  --seed 42 \
  --grad-accum-steps 2 \
  2>&1 | tee benchmark_optimized.log
END_OPT=$(date +%s)
OPT_TIME=$((END_OPT - START_OPT))

echo ""
echo "========================================"
echo "Benchmark Results"
echo "========================================"
echo "Baseline time:   ${BASELINE_TIME}s"
echo "Optimized time:  ${OPT_TIME}s"

if [ $BASELINE_TIME -gt 0 ]; then
    SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $BASELINE_TIME / $OPT_TIME}")
    IMPROVEMENT=$(awk "BEGIN {printf \"%.1f\", 100 * (1 - $OPT_TIME / $BASELINE_TIME)}")
    echo "Speedup:         ${SPEEDUP}x"
    echo "Improvement:     ${IMPROVEMENT}%"
else
    echo "Speedup:         N/A"
fi

echo ""
echo "Logs saved to:"
echo "  - benchmark_baseline.log"
echo "  - benchmark_optimized.log"
echo ""
echo "To compare metrics, run:"
echo "  grep 'best_epoch\\|val_mse_best\\|test_mse_best' benchmark_*.log"
