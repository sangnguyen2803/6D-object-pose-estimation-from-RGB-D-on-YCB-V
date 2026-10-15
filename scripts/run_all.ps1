param([int]$PoseFrames=900, [int]$ExperimentFrames=24)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
function Run-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
Run-Python scripts/prepare_ycbv.py
Run-Python scripts/run.py index
Run-Python scripts/run.py audit
Run-Python -m pytest -q
Run-Python scripts/run.py evaluate
Run-Python scripts/run.py benchmark --limit $PoseFrames --output artifacts/ycbv/benchmark_full
Run-Python scripts/run.py infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/demo_rgbd
Run-Python scripts/run_experiments.py --limit $ExperimentFrames
Run-Python scripts/visualize_instances.py artifacts/ycbv/demo_rgbd/ycbv_test_000048_1074
Run-Python scripts/build_report.py
Run-Python scripts/validate_ycbv.py --expected-frames $PoseFrames

