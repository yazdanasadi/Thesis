Param(
    [string]$BestParamsPath = "best_hparams.json",
    [ValidateSet("activity","mimic","physionet","ushcn")]
    [string[]]$Datasets = @("activity","mimic","physionet","ushcn")
)

if (-not (Test-Path $BestParamsPath)) {
    Write-Error "Best params file '$BestParamsPath' not found."
    exit 1
}

$json = Get-Content $BestParamsPath -Raw | ConvertFrom-Json

foreach ($dataset in $Datasets) {
    if (-not $json.$dataset) {
        Write-Warning "No best params found for dataset '$dataset'. Skipping."
        continue
    }

    foreach ($functionName in $json.$dataset.PSObject.Properties.Name) {
        $entry = $json.$dataset.$functionName
        if (-not $entry.best_params) {
            Write-Warning "Missing best_params for $dataset / $functionName. Skipping."
            continue
        }

        $trainer = $entry.trainer
        $params = $entry.best_params

        # Determine trainer script and fixed args
        if ($dataset -eq "ushcn" -and $trainer -eq "icfld") {
            $scriptPath = "FLD_ICC/train_FLD_ICC_ushcn.py"
            $scriptExe  = Split-Path $scriptPath -Leaf
            $cmd = @(
                "python", $scriptExe,
                "-d", $dataset,
                "-fn", $functionName
            )
            if ($params.latent_dim) { $cmd += @("--latent-dim", $params.latent_dim) }
            if ($params.embed_dim)  { $cmd += @("--embedding-dim", $params.embed_dim) }
        }
        elseif ($trainer -eq "fld") {
            $scriptPath = "FLD/train_FLD.py"
            $scriptExe  = Split-Path $scriptPath -Leaf
            $cmd = @(
                "python", $scriptExe,
                "-d", $dataset,
                "-fn", $functionName
            )
            if ($params.embed_dim) { $cmd += @("-ed", $params.embed_dim) }
        } else {
            $scriptPath = "FLD_ICC/train_FLD_ICC.py"
            $scriptExe  = Split-Path $scriptPath -Leaf
            $cmd = @(
                "python", $scriptExe,
                "-d", $dataset,
                "-fn", $functionName
            )
            if ($params.latent_dim) { $cmd += @("--latent-dim", $params.latent_dim) }
            if ($params.embed_dim)  { $cmd += @("--embedding-dim", $params.embed_dim) }
        }

        if ($params.num_heads)     { $cmd += @("--num-heads", $params.num_heads) }
        if ($params.depth)         { $cmd += @("--depth", $params.depth) }
        if ($params.batch_size)    { $cmd += @("--batch-size", $params.batch_size) }
        if ($params.epochs)        { $cmd += @("--epochs", $params.epochs) }
        if ($params.early_stop)    { $cmd += @("--early-stop", $params.early_stop) }
        if ($trainer -eq "fld") {
            if ($params.lr)            { $cmd += @("-lr", $params.lr) }
            elseif ($params."learn_rate") { $cmd += @("-lr", $params."learn_rate") }
            if ($params.wd)            { $cmd += @("-wd", $params.wd) }
            elseif ($params."weight_decay") { $cmd += @("-wd", $params."weight_decay") }
        }
        else {
            if ($params.lr)            { $cmd += @("--lr", $params.lr) }
            if ($params.learn_rate)    { $cmd += @("--lr", $params.learn_rate) }
            if ($params.wd)            { $cmd += @("--wd", $params.wd) }
            if ($params.weight_decay)  { $cmd += @("--wd", $params.weight_decay) }
        }
        if ($params.history)       { $cmd += @("--observation-time", $params.history) }

        $logdir = "runs/{0}_{1}_{2}" -f $dataset, $functionName, $trainer
        $cmd += @("--tbon", "--logdir", $logdir)

        Write-Host "Launching $dataset / $functionName with trainer $trainer"
        Write-Host $cmd -join " "
        $scriptDir = Split-Path -Parent $scriptPath
        Push-Location $scriptDir
        try {
            & $cmd
        }
        finally {
            Pop-Location
        }
    }
}
