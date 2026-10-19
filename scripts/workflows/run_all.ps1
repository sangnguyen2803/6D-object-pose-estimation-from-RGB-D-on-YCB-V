param([int]$PoseFrames=900, [int]$ExperimentFrames=24)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
function Run-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
Run-Python scripts/data/prepare_ycbv.py
Run-Python main.py index
Run-Python main.py audit
Run-Python -m pytest -q
Run-Python main.py evaluate
Run-Python main.py benchmark --limit $PoseFrames --output artifacts/ycbv/benchmark_full
Run-Python main.py infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/demo_rgbd
Run-Python scripts/evaluation/run_experiments.py --limit $ExperimentFrames
Run-Python scripts/inference/visualize_instances.py artifacts/ycbv/demo_rgbd/ycbv_test_000048_1074
Run-Python scripts/reports/build_report.py
Run-Python scripts/validation/validate_ycbv.py --expected-frames $PoseFrames
