param([int]$WaitForProcess=0)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($WaitForProcess -gt 0) {
    Write-Output "Waiting for replay process $WaitForProcess"
    Wait-Process -Id $WaitForProcess -ErrorAction SilentlyContinue
}
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
function Run-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
Run-Python scripts/build_extension_report.py
Run-Python scripts/visualize_instances.py artifacts/ycbv/extension/demo/ycbv_test_000048_1074
Run-Python -m pytest -q
Run-Python scripts/validate_extension.py

