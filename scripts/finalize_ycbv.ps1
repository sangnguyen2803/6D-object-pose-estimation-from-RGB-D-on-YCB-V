param([int]$ExpectedFrames=900, [int]$ExperimentFrames=24, [int]$WaitForProcess=0)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($WaitForProcess -gt 0) {
    Write-Output "Waiting for benchmark process $WaitForProcess"
    Wait-Process -Id $WaitForProcess -ErrorAction SilentlyContinue
}
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
function Run-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
$summary = Get-Content -LiteralPath artifacts/ycbv/benchmark_full/run_summary.json -Raw | ConvertFrom-Json
if ($summary.Count -ne $ExpectedFrames) { throw 'Benchmark is incomplete; refusing final delivery' }
if (-not (Test-Path artifacts/ycbv/benchmark_full/bop_predictions.csv)) { throw 'Benchmark CSV missing' }
Run-Python scripts/run_experiments.py --limit $ExperimentFrames
Run-Python scripts/run.py infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/demo_rgbd
Run-Python scripts/visualize_instances.py artifacts/ycbv/demo_rgbd/ycbv_test_000048_1074
Run-Python scripts/build_report.py
Run-Python -m pytest -q
Run-Python scripts/validate_ycbv.py --expected-frames $ExpectedFrames

