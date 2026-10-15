param([int]$WaitForProcess=0)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($WaitForProcess -gt 0) { Wait-Process -Id $WaitForProcess -ErrorAction SilentlyContinue }
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
function Run-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
Run-Python scripts/run.py --config configs/research.yaml index
Run-Python scripts/run.py --config configs/research.yaml audit --output reports/ycbv/research/data_audit.json
Run-Python -m pytest -q
Run-Python scripts/cache_point_training.py
Run-Python scripts/train_point_network.py
Run-Python scripts/point_cache_baselines.py
Run-Python scripts/train_fusion.py
Run-Python scripts/evaluate_research.py
Run-Python scripts/update_hybrid_docs.py
Run-Python scripts/run.py --config configs/hybrid.yaml infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/research/cli_demo
Run-Python scripts/validate_hybrid_external.py
Run-Python scripts/visualize_instances.py artifacts/ycbv/research/demo/ycbv_test_000048_1074
Run-Python scripts/build_research_report.py
Run-Python scripts/validate_research.py

