param([int]$WaitForProcess=0)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($WaitForProcess -gt 0) { Wait-Process -Id $WaitForProcess -ErrorAction SilentlyContinue }
$python=Join-Path (Get-Location) '.venv/Scripts/python.exe'
function Run-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
Run-Python -m pytest -q
Run-Python scripts/validate_resident_renderer.py
Run-Python scripts/benchmark_architecture.py --limit 24 --repeats 2 --variants baseline batch_only cache_only optimized --output artifacts/ycbv/architecture/component_ablation --report reports/ycbv/architecture/component_ablation.json
Run-Python scripts/benchmark_architecture.py --limit 24 --repeats 2 --variants resident_only --output artifacts/ycbv/architecture/resident_ablation --report reports/ycbv/architecture/resident_ablation.json
Run-Python scripts/benchmark_architecture.py --limit 96 --repeats 2 --variants baseline optimized_fast optimized --output artifacts/ycbv/architecture/benchmark96 --report reports/ycbv/architecture/benchmark96.json
Run-Python scripts/run.py --config configs/optimized.yaml infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/architecture/demo
Run-Python scripts/visualize_instances.py artifacts/ycbv/architecture/demo/ycbv_test_000048_1074
Run-Python scripts/build_architecture_report.py
Run-Python scripts/validate_architecture.py
