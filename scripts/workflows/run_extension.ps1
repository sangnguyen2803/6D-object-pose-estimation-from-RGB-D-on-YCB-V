$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
function Run-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
Run-Python -m pytest -q
Run-Python scripts/validation/validate_refinement_synthetic.py
Run-Python scripts/validation/validate_refinement_synthetic.py --seed 271 --output reports/ycbv/extension/synthetic_holdout.json
Run-Python scripts/inference/replay_refinement.py
Run-Python main.py --config configs/enhanced.yaml infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/extension/demo
Run-Python scripts/inference/visualize_instances.py artifacts/ycbv/extension/demo/ycbv_test_000048_1074
Run-Python scripts/reports/build_extension_report.py
Run-Python scripts/validation/validate_extension.py
