# Quick test script to verify FLD training works
# Tests FLD with Linear basis for 5 epochs

Write-Host "Testing FLD training script..." -ForegroundColor Cyan
Write-Host ""

$StartTime = Get-Date

python -u FLD/train_FLD.py `
  -d physionet `
  -ot 24 `
  -bs 32 `
  -e 5 `
  -es 3 `
  -fn L `
  -ed 4 `
  -nh 4 `
  -dp 2 `
  -lr 1e-4 `
  -wd 1e-3 `
  -s 42 `
  --gpu 0

$EndTime = Get-Date
$Duration = ($EndTime - $StartTime).TotalSeconds

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Test completed!" -ForegroundColor Green
Write-Host "Duration: $Duration seconds" -ForegroundColor Cyan
Write-Host ""

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: FLD training ran successfully!" -ForegroundColor Green
} else {
    Write-Host "FAILED: FLD training exited with code $LASTEXITCODE" -ForegroundColor Red
}
