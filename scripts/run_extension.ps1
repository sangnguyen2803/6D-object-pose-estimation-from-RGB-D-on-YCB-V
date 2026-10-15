$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
function Run-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
Run-Python -m pytest -q
Run-Python scripts/validate_refinement_synthetic.py
Run-Python scripts/validate_refinement_synthetic.py --seed 271 --output reports/ycbv/extension/synthetic_holdout.json
Run-Python scripts/replay_refinement.py
Run-Python scripts/run.py --config configs/enhanced.yaml infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/extension/demo
Run-Python scripts/visualize_instances.py artifacts/ycbv/extension/demo/ycbv_test_000048_1074
Run-Python scripts/build_extension_report.py
Run-Python scripts/validate_extension.py

