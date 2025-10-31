# Benchmark FLD vs IC-FLD on PhysioNet Dataset
# Runs all basis functions (C, L, Q, S) for 1000 epochs
# Logs metrics and timing information

param(
    [int]$Epochs = 1000,
    [int]$BatchSize = 32,
    [int]$ObservationTime = 24,
    [int]$EarlyStop = 15,
    [float]$LearningRate = 1e-4,
    [float]$WeightDecay = 1e-3,
    [int]$Seed = 42,
    [string]$GPU = "0"
)

# Configuration
$Dataset = "physionet"
$Functions = @("C", "L", "Q", "S")
$LogDir = "benchmark_results_physionet_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Results storage
$Results = @{
    FLD = @{}
    ICFLD = @{}
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FLD vs IC-FLD Benchmark - PhysioNet" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Dataset:          $Dataset"
Write-Host "  Epochs:           $Epochs"
Write-Host "  Batch Size:       $BatchSize"
Write-Host "  Observation Time: $ObservationTime hours"
Write-Host "  Early Stop:       $EarlyStop epochs"
Write-Host "  Learning Rate:    $LearningRate"
Write-Host "  Weight Decay:     $WeightDecay"
Write-Host "  Seed:             $Seed"
Write-Host "  GPU:              $GPU"
Write-Host "  Log Directory:    $LogDir"
Write-Host ""
Write-Host "Functions to test: $($Functions -join ', ')" -ForegroundColor Yellow
Write-Host ""

# Progress tracking
$TotalRuns = $Functions.Count * 2  # FLD + IC-FLD for each function
$CurrentRun = 0

#==============================================================================
# Function: Run Training and Extract Metrics
#==============================================================================
function Run-Training {
    param(
        [string]$ModelType,      # "FLD" or "ICFLD"
        [string]$Function,       # "C", "L", "Q", "S"
        [string]$TrainerScript,  # Path to training script
        [hashtable]$ExtraArgs = @{}
    )

    $Script:CurrentRun++
    $Progress = [math]::Round(($Script:CurrentRun / $TotalRuns) * 100, 1)

    Write-Host ""
    Write-Host "[$Script:CurrentRun/$TotalRuns - $Progress%] Running $ModelType with function '$Function'..." -ForegroundColor Green
    Write-Host "----------------------------------------"

    $LogFile = Join-Path $LogDir "${ModelType}_${Function}.log"
    $MetricsFile = Join-Path $LogDir "${ModelType}_${Function}_metrics.json"

    # Build argument list (FLD uses short forms, IC-FLD uses long forms)
    if ($TrainerScript -like "*FLD/train_FLD.py") {
        # FLD-specific argument format
        $Args = @(
            $TrainerScript,
            "-d", $Dataset,
            "-ot", $ObservationTime,
            "-bs", $BatchSize,
            "-e", $Epochs,
            "-es", $EarlyStop,
            "-fn", $Function,
            "-lr", $LearningRate,
            "-wd", $WeightDecay,
            "-s", $Seed,
            "--gpu", $GPU,
            "--tbon",
            "--logdir", "runs/${ModelType}_${Function}_physionet"
        )
    } else {
        # IC-FLD argument format
        $Args = @(
            $TrainerScript,
            "-d", $Dataset,
            "-ot", $ObservationTime,
            "-bs", $BatchSize,
            "--epochs", $Epochs,
            "--early-stop", $EarlyStop,
            "-fn", $Function,
            "--lr", $LearningRate,
            "--wd", $WeightDecay,
            "--seed", $Seed,
            "--gpu", $GPU,
            "--tbon",
            "--logdir", "runs/${ModelType}_${Function}_physionet"
        )
    }

    # Add model-specific arguments
    foreach ($key in $ExtraArgs.Keys) {
        $Args += $key
        if ($ExtraArgs[$key] -ne $null) {
            $Args += $ExtraArgs[$key]
        }
    }

    # Start timer
    $StartTime = Get-Date
    Write-Host "Start time: $($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray

    # Run training and capture output (use -u for unbuffered output)
    try {
        $PythonArgs = @("-u") + $Args  # Add unbuffered flag
        $Output = & python @PythonArgs 2>&1 | Tee-Object -FilePath $LogFile
        $ExitCode = $LASTEXITCODE

        # Stop timer
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        $DurationSeconds = [math]::Round($Duration.TotalSeconds, 2)

        Write-Host "End time:   $($EndTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
        Write-Host "Duration:   $DurationSeconds seconds ($([math]::Round($Duration.TotalMinutes, 2)) minutes)" -ForegroundColor Cyan

        # Extract metrics from last line (JSON output)
        $JsonLine = $Output | Where-Object { $_ -match '^\{.*\}$' } | Select-Object -Last 1

        if ($JsonLine) {
            $Metrics = $JsonLine | ConvertFrom-Json

            # Add timing information
            $Metrics | Add-Member -NotePropertyName "total_time_seconds" -NotePropertyValue $DurationSeconds
            $Metrics | Add-Member -NotePropertyName "start_time" -NotePropertyValue $StartTime.ToString('yyyy-MM-dd HH:mm:ss')
            $Metrics | Add-Member -NotePropertyName "end_time" -NotePropertyValue $EndTime.ToString('yyyy-MM-dd HH:mm:ss')
            $Metrics | Add-Member -NotePropertyName "function" -NotePropertyValue $Function
            $Metrics | Add-Member -NotePropertyName "model_type" -NotePropertyValue $ModelType

            # Save metrics to JSON
            $Metrics | ConvertTo-Json -Depth 10 | Out-File -FilePath $MetricsFile

            # Display key metrics
            Write-Host ""
            Write-Host "Key Metrics:" -ForegroundColor Yellow
            Write-Host "  Best Epoch:       $($Metrics.best_epoch)"
            Write-Host "  Val MSE (best):   $([math]::Round($Metrics.val_mse_best, 6))"
            Write-Host "  Val RMSE (best):  $([math]::Round($Metrics.val_rmse_best, 6))"
            Write-Host "  Val MAE (best):   $([math]::Round($Metrics.val_mae_best, 6))"
            if ($Metrics.test_mse_best) {
                Write-Host "  Test MSE (best):  $([math]::Round($Metrics.test_mse_best, 6))"
                Write-Host "  Test RMSE (best): $([math]::Round($Metrics.test_rmse_best, 6))"
                Write-Host "  Test MAE (best):  $([math]::Round($Metrics.test_mae_best, 6))"
            }
            Write-Host "  Total Time:       $DurationSeconds seconds" -ForegroundColor Cyan

            return @{
                Success = $true
                Metrics = $Metrics
                Duration = $DurationSeconds
                LogFile = $LogFile
                MetricsFile = $MetricsFile
            }
        } else {
            Write-Host "Warning: Could not extract metrics from output" -ForegroundColor Yellow
            return @{
                Success = $false
                Duration = $DurationSeconds
                LogFile = $LogFile
                Error = "No JSON metrics found in output"
            }
        }

    } catch {
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        $DurationSeconds = [math]::Round($Duration.TotalSeconds, 2)

        Write-Host "Error: $_" -ForegroundColor Red
        return @{
            Success = $false
            Duration = $DurationSeconds
            LogFile = $LogFile
            Error = $_.Exception.Message
        }
    }
}

#==============================================================================
# Run FLD Benchmarks
#==============================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "Part 1: FLD Baseline" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta

foreach ($Function in $Functions) {
    $Result = Run-Training `
        -ModelType "FLD" `
        -Function $Function `
        -TrainerScript "FLD/train_FLD.py" `
        -ExtraArgs @{
            "-ed" = "4"       # Embedding dim per head
            "-nh" = "4"       # Number of heads
            "-dp" = "2"       # Decoder depth (note: FLD hardcodes latent_dim=20)
        }

    $Results.FLD[$Function] = $Result

    # Brief pause to ensure clean separation
    Start-Sleep -Seconds 2
}

#==============================================================================
# Run IC-FLD Benchmarks
#==============================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "Part 2: IC-FLD (Novel Model)" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta

foreach ($Function in $Functions) {
    $Result = Run-Training `
        -ModelType "ICFLD" `
        -Function $Function `
        -TrainerScript "FLD_ICC/train_FLD_ICC.py" `
        -ExtraArgs @{
            "-ld" = "128"     # IC-FLD latent dimension
            "-ed" = "64"      # Embedding dimension (total)
            "-nh" = "4"       # Number of heads
            "--depth" = "2"   # Decoder depth
        }

    $Results.ICFLD[$Function] = $Result

    # Brief pause
    Start-Sleep -Seconds 2
}

#==============================================================================
# Generate Summary Report
#==============================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "BENCHMARK COMPLETE - SUMMARY REPORT" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Summary table
Write-Host "Timing Summary (PhysioNet, $Epochs epochs):" -ForegroundColor Yellow
Write-Host ""
Write-Host ("{0,-10} {1,-15} {2,-15} {3,-20}" -f "Function", "FLD (seconds)", "ICFLD (seconds)", "Speedup (FLD/ICFLD)")
Write-Host ("{0,-10} {1,-15} {2,-15} {3,-20}" -f "--------", "--------------", "----------------", "-------------------")

$TotalFLDTime = 0
$TotalICFLDTime = 0

foreach ($Function in $Functions) {
    $FLDTime = if ($Results.FLD[$Function].Success) { $Results.FLD[$Function].Duration } else { "FAILED" }
    $ICFLDTime = if ($Results.ICFLD[$Function].Success) { $Results.ICFLD[$Function].Duration } else { "FAILED" }

    if ($FLDTime -is [double] -and $ICFLDTime -is [double]) {
        $Speedup = [math]::Round($FLDTime / $ICFLDTime, 2)
        $SpeedupStr = "${Speedup}x"
        $TotalFLDTime += $FLDTime
        $TotalICFLDTime += $ICFLDTime
    } else {
        $SpeedupStr = "N/A"
    }

    Write-Host ("{0,-10} {1,-15} {2,-15} {3,-20}" -f $Function, $FLDTime, $ICFLDTime, $SpeedupStr)
}

Write-Host ("{0,-10} {1,-15} {2,-15} {3,-20}" -f "--------", "--------------", "----------------", "-------------------")
Write-Host ("{0,-10} {1,-15} {2,-15} {3,-20}" -f "TOTAL",
    [math]::Round($TotalFLDTime, 2),
    [math]::Round($TotalICFLDTime, 2),
    "$(if ($TotalICFLDTime -gt 0) { [math]::Round($TotalFLDTime / $TotalICFLDTime, 2) } else { 'N/A' })x")

Write-Host ""
Write-Host "Performance Summary:" -ForegroundColor Yellow
Write-Host ""
Write-Host ("{0,-10} {1,-8} {2,-15} {3,-15} {4,-15} {5,-15}" -f "Function", "Model", "Val MSE", "Test MSE", "Val MAE", "Test MAE")
Write-Host ("{0,-10} {1,-8} {2,-15} {3,-15} {4,-15} {5,-15}" -f "--------", "-----", "-------", "--------", "-------", "--------")

foreach ($Function in $Functions) {
    foreach ($ModelType in @("FLD", "ICFLD")) {
        $Result = $Results[$ModelType][$Function]
        if ($Result.Success) {
            $M = $Result.Metrics
            Write-Host ("{0,-10} {1,-8} {2,-15} {3,-15} {4,-15} {5,-15}" -f
                $Function,
                $ModelType,
                [math]::Round($M.val_mse_best, 6),
                $(if ($M.test_mse_best) { [math]::Round($M.test_mse_best, 6) } else { "N/A" }),
                [math]::Round($M.val_mae_best, 6),
                $(if ($M.test_mae_best) { [math]::Round($M.test_mae_best, 6) } else { "N/A" })
            )
        } else {
            Write-Host ("{0,-10} {1,-8} {2,-15} {3,-15} {4,-15} {5,-15}" -f
                $Function, $ModelType, "FAILED", "FAILED", "FAILED", "FAILED")
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Files Generated:" -ForegroundColor Yellow
Write-Host ""
Write-Host "Logs directory: $LogDir"
Write-Host ""
Get-ChildItem -Path $LogDir | ForEach-Object {
    Write-Host "  $($_.Name)"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green

# Save summary to JSON
$SummaryFile = Join-Path $LogDir "summary.json"
$Results | ConvertTo-Json -Depth 10 | Out-File -FilePath $SummaryFile
Write-Host "Summary saved to: $SummaryFile" -ForegroundColor Cyan

# Save summary to CSV
$CSVFile = Join-Path $LogDir "summary.csv"
$CSVData = @()

foreach ($Function in $Functions) {
    foreach ($ModelType in @("FLD", "ICFLD")) {
        $Result = $Results[$ModelType][$Function]
        if ($Result.Success) {
            $M = $Result.Metrics
            $CSVData += [PSCustomObject]@{
                Model = $ModelType
                Function = $Function
                BestEpoch = $M.best_epoch
                ValMSE = $M.val_mse_best
                ValRMSE = $M.val_rmse_best
                ValMAE = $M.val_mae_best
                TestMSE = $M.test_mse_best
                TestRMSE = $M.test_rmse_best
                TestMAE = $M.test_mae_best
                TotalTimeSeconds = $Result.Duration
                TotalTimeMinutes = [math]::Round($Result.Duration / 60, 2)
                StartTime = $M.start_time
                EndTime = $M.end_time
            }
        }
    }
}

$CSVData | Export-Csv -Path $CSVFile -NoTypeInformation
Write-Host "CSV summary saved to: $CSVFile" -ForegroundColor Cyan

Write-Host ""
Write-Host "Benchmark completed successfully!" -ForegroundColor Green
Write-Host "Total execution time: $([math]::Round(($TotalFLDTime + $TotalICFLDTime) / 60, 2)) minutes" -ForegroundColor Cyan
