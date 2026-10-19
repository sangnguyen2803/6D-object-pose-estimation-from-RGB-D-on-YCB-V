param([switch]$CpuOnly)
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$projectCache = Join-Path $projectRoot 'artifacts\cache\pip'
& $projectPython -m pip install --upgrade pip --cache-dir $projectCache
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
if ($CpuOnly) {
    & $projectPython -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu --cache-dir $projectCache
} else {
    & $projectPython -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --cache-dir $projectCache
}
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $projectPython -m pip install -e '.[test,geometry,viewer]' --cache-dir $projectCache
if ($LASTEXITCODE -ne 0) { throw 'Project dependencies installation failed' }
& $projectPython -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
