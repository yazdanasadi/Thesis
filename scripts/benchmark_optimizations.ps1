# Benchmark IC-FLD training with and without optimizations
# Usage: .\scripts\benchmark_optimizations.ps1

param(
    [string]$Dataset = "physionet",
    [int]$BatchSize = 32,
    [int]$Epochs = 5,
    [string]$Function = "L",
    [int]$LatentDim = 128,
    [int]$EmbedDim = 64,
    [int]$NumHeads = 4,
    [int]$Depth = 2
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IC-FLD Optimization Benchmark" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Dataset: $Dataset"
Write-Host "Batch size: $BatchSize"
Write-Host "Epochs: $Epochs (short run for benchmarking)"
Write-Host ""

# Baseline
Write-Host "[1/2] Running BASELINE (train_FLD_ICC.py)..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
$startBaseline = Get-Date
python FLD_ICC/train_FLD_ICC.py `
  -d $Dataset `
  -bs $BatchSize `
  --epochs $Epochs `
  -fn $Function `
  -ld $LatentDim `
  -ed $EmbedDim `
  -nh $NumHeads `
  --depth $Depth `
  --lr 1e-4 `
  --wd 1e-3 `
  --seed 42 `
  2>&1 | Tee-Object -FilePath benchmark_baseline.log
$endBaseline = Get-Date
$baselineTime = ($endBaseline - $startBaseline).TotalSeconds

Write-Host ""
Write-Host "[2/2] Running OPTIMIZED (train_FLD_ICC_optimized.py)..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
$startOpt = Get-Date
python FLD_ICC/train_FLD_ICC_optimized.py `
  -d $Dataset `
  -bs $BatchSize `
  --epochs $Epochs `
  -fn $Function `
  -ld $LatentDim `
  -ed $EmbedDim `
  -nh $NumHeads `
  --depth $Depth `
  --lr 1e-4 `
  --wd 1e-3 `
  --seed 42 `
  --grad-accum-steps 2 `
  2>&1 | Tee-Object -FilePath benchmark_optimized.log
$endOpt = Get-Date
$optTime = ($endOpt - $startOpt).TotalSeconds

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Benchmark Results" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Baseline time:   $([math]::Round($baselineTime, 2))s"
Write-Host "Optimized time:  $([math]::Round($optTime, 2))s"

if ($baselineTime -gt 0) {
    $speedup = [math]::Round($baselineTime / $optTime, 2)
    $improvement = [math]::Round(100 * (1 - $optTime / $baselineTime), 1)
    Write-Host "Speedup:         ${speedup}x" -ForegroundColor Green
    Write-Host "Improvement:     ${improvement}%" -ForegroundColor Green
} else {
    Write-Host "Speedup:         N/A"
}

Write-Host ""
Write-Host "Logs saved to:"
Write-Host "  - benchmark_baseline.log"
Write-Host "  - benchmark_optimized.log"
Write-Host ""
Write-Host "To compare metrics, run:"
Write-Host "  Select-String -Path benchmark_*.log -Pattern 'best_epoch|val_mse_best|test_mse_best'"
