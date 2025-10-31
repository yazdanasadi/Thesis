###############################################################################
# Run IC-FLD on All Dataset/Function Combinations (PowerShell)
#
# This script runs IC-FLD training with fixed hyperparameters across all
# datasets (physionet, mimic, activity, ushcn) and all basis functions (C, L, Q, S).
#
# Usage:
#   .\scripts\run_icfld_all_configs.ps1
#   pwsh .\scripts\run_icfld_all_configs.ps1
#
# Total runs: 4 datasets × 4 functions = 16 training runs
###############################################################################

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Separate directories for full training runs
$TBDir = Join-Path $ProjectRoot "runs_icfld_full"
$ModelsDir = Join-Path $ProjectRoot "FLD_ICC\saved_models_full"

New-Item -ItemType Directory -Force -Path $TBDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

Write-Host "=============================================="
Write-Host "IC-FLD Full Training - All Configurations"
Write-Host "=============================================="
Write-Host "TensorBoard logs:  $TBDir"
Write-Host "Model checkpoints: $ModelsDir"
Write-Host "=============================================="
Write-Host ""

# Fixed hyperparameters (reasonable defaults from search space)
$LatentDim = 128
$NumHeads = 4
$EmbedPerHead = 4
$EmbeddingDim = $NumHeads * $EmbedPerHead  # 16
$Depth = 2
$LR = "1e-4"
$WD = "1e-3"
$Epochs = 2
$EarlyStop = 15
$BatchSize = 32

# Dataset configurations: @{Dataset, ObsTime, UseUshcn}
$Datasets = @(
    @{Dataset="physionet"; ObsTime=24; UseUshcn=$false},
    @{Dataset="mimic"; ObsTime=24; UseUshcn=$false},
    @{Dataset="activity"; ObsTime=3000; UseUshcn=$false},
    @{Dataset="ushcn"; ObsTime=24; UseUshcn=$true}
)

# Basis functions
$Functions = @("C", "L", "Q", "S")

Set-Location $ProjectRoot

# Counter for progress tracking
$TotalConfigs = $Datasets.Count * $Functions.Count
$Current = 0

Write-Host "Hyperparameters:"
Write-Host "  Latent dim:     $LatentDim"
Write-Host "  Num heads:      $NumHeads"
Write-Host "  Embed per head: $EmbedPerHead"
Write-Host "  Embedding dim:  $EmbeddingDim"
Write-Host "  Depth:          $Depth"
Write-Host "  Learning rate:  $LR"
Write-Host "  Weight decay:   $WD"
Write-Host "  Epochs:         $Epochs (early stop: $EarlyStop)"
Write-Host "  Batch size:     $BatchSize"
Write-Host ""

foreach ($DatasetConfig in $Datasets) {
    $Dataset = $DatasetConfig.Dataset
    $ObsTime = $DatasetConfig.ObsTime
    $UseUshcn = $DatasetConfig.UseUshcn

    # Select appropriate trainer
    if ($UseUshcn) {
        $Trainer = "FLD_ICC\train_FLD_ICC_ushcn.py"
        $TrainerName = "train_FLD_ICC_ushcn.py"
    } else {
        $Trainer = "FLD_ICC\train_FLD_ICC.py"
        $TrainerName = "train_FLD_ICC.py"
    }

    foreach ($Function in $Functions) {
        $Current++
        $RunName = "icfld_${Dataset}_${Function}"
        $LogDir = Join-Path $TBDir $RunName

        Write-Host ""
        Write-Host "=============================================="
        Write-Host "[$Current/$TotalConfigs] Running: $RunName"
        Write-Host "=============================================="
        Write-Host "  Dataset:    $Dataset"
        Write-Host "  Function:   $Function"
        Write-Host "  Trainer:    $TrainerName"
        Write-Host "  Obs time:   $ObsTime"
        Write-Host "  Log dir:    $LogDir"
        Write-Host "----------------------------------------------"

        # Build command arguments
        $Arguments = @(
            $Trainer,
            "--dataset", $Dataset,
            "--observation-time", $ObsTime,
            "--batch-size", $BatchSize,
            "--function", $Function,
            "--latent-dim", $LatentDim,
            "--num-heads", $NumHeads,
            "--embedding-dim", $EmbeddingDim,
            "--depth", $Depth,
            "--epochs", $Epochs,
            "--early-stop", $EarlyStop,
            "--lr", $LR,
            "--wd", $WD,
            "--tbon",
            "--logdir", $LogDir,
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
Write-Host "IC-FLD Full Training Complete!"
Write-Host "=============================================="
Write-Host "Total configurations: $TotalConfigs"
Write-Host "TensorBoard logs: $TBDir"
Write-Host "Model checkpoints: $ModelsDir"
Write-Host ""
Write-Host "To view TensorBoard logs:"
Write-Host "  tensorboard --logdir $TBDir"
Write-Host "=============================================="
