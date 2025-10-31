###############################################################################
# Run FLD Baseline on All Dataset/Function Combinations (PowerShell)
#
# This script runs FLD training with fixed hyperparameters across all
# datasets (physionet, mimic, activity, ushcn) and all basis functions (C, L, Q, S).
#
# Usage:
#   .\scripts\run_fld_all_configs.ps1
#   pwsh .\scripts\run_fld_all_configs.ps1
#
# Total runs: 4 datasets × 4 functions = 16 training runs
###############################################################################

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Separate directories for full training runs
$TBDir = Join-Path $ProjectRoot "runs_fld_full"
$ModelsDir = Join-Path $ProjectRoot "FLD\saved_models_full"

New-Item -ItemType Directory -Force -Path $TBDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

Write-Host "=============================================="
Write-Host "FLD Baseline Full Training - All Configurations"
Write-Host "=============================================="
Write-Host "TensorBoard logs:  $TBDir"
Write-Host "Model checkpoints: $ModelsDir"
Write-Host "=============================================="
Write-Host ""

# Fixed hyperparameters (reasonable defaults from search space)
# NOTE: FLD latent_dim hardcoded to 20, not configurable
# $LatentDim = 128
$NumHeads = 4
$EmbedPerHead = 4
$Depth = 2
$LR = "1e-4"
$WD = "1e-3"
$Epochs = 2
$EarlyStop = 30
$BatchSize = 32

# Dataset configurations: @{Dataset, ObsTime}
$Datasets = @(
    @{Dataset="physionet"; ObsTime=24},
    @{Dataset="mimic"; ObsTime=24},
    @{Dataset="activity"; ObsTime=3000},
    @{Dataset="ushcn"; ObsTime=24}
)

# Basis functions
$Functions = @("C", "L", "Q", "S")

Set-Location $ProjectRoot

# Counter for progress tracking
$TotalConfigs = $Datasets.Count * $Functions.Count
$Current = 0

Write-Host "Hyperparameters:"
Write-Host "  Latent dim:     20 (hardcoded in FLD)"
Write-Host "  Num heads:      $NumHeads"
Write-Host "  Embed per head: $EmbedPerHead"
Write-Host "  Depth:          $Depth"
Write-Host "  Learning rate:  $LR"
Write-Host "  Weight decay:   $WD"
Write-Host "  Epochs:         $Epochs (early stop: $EarlyStop)"
Write-Host "  Batch size:     $BatchSize"
Write-Host ""

foreach ($DatasetConfig in $Datasets) {
    $Dataset = $DatasetConfig.Dataset
    $ObsTime = $DatasetConfig.ObsTime

    foreach ($Function in $Functions) {
        $Current++
        $RunName = "fld_${Dataset}_${Function}"
        $LogDir = Join-Path $TBDir $RunName

        Write-Host ""
        Write-Host "=============================================="
        Write-Host "[$Current/$TotalConfigs] Running: $RunName"
        Write-Host "=============================================="
        Write-Host "  Dataset:    $Dataset"
        Write-Host "  Function:   $Function"
        Write-Host "  Obs time:   $ObsTime"
        Write-Host "  Log dir:    $LogDir"
        Write-Host "----------------------------------------------"

        # Build command arguments
        $Arguments = @(
            "FLD\train_FLD.py",
            "--dataset", $Dataset,
            "--observation-time", $ObsTime,
            "--batch-size", $BatchSize,
            "--function", $Function,
            "--num-heads", $NumHeads,
            "--embedding-dim", $EmbedPerHead,
            "--depth", $Depth,
            "--epochs", $Epochs,
            "--early-stop", $EarlyStop,
            "--learn-rate", $LR,
            "--weight-decay", $WD,
            "--seed", "42"
        )

        # Run training
        $Process = Start-Process -FilePath "python" -ArgumentList $Arguments -NoNewWindow -Wait -PassThru

        if ($Process.ExitCode -eq 0) {
            Write-Host "✓ Completed: $RunName" -ForegroundColor Green
        } else {
            Write-Host "✗ Failed: $RunName (exit code: $($Process.ExitCode))" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "=============================================="
Write-Host "FLD Baseline Full Training Complete!"
Write-Host "=============================================="
Write-Host "Total configurations: $TotalConfigs"
Write-Host "TensorBoard logs: $TBDir"
Write-Host "Model checkpoints: $ModelsDir"
Write-Host ""
Write-Host "To view TensorBoard logs:"
Write-Host "  tensorboard --logdir $TBDir"
Write-Host "=============================================="
