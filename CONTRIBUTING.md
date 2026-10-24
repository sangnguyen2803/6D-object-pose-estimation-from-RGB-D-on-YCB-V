# Contributing

Run commands from the repository root. On Windows, use
`.\.venv\Scripts\python.exe` in place of `python` if the environment is not active.
See the [README](README.md) for bootstrap instructions. For an existing Python
environment, install the package and test extra:

```powershell
python -m pip install -e '.[test]'
python main.py --help
python -m pytest -q
```

Source lives under `src/clutterpick`; place changes in the relevant feature
folder. Standalone scripts are grouped by task under `scripts`. Add regression
tests under the corresponding `tests` folder when fixing observable behavior.
Prefer small generated fixtures to downloading datasets or checkpoints for tests.

For a geometry change, start with `python -m pytest -q tests/geometry`. For input
and CLI changes, use `python -m pytest -q tests/test_cli.py tests/test_utils.py
tests/inference/test_external_inputs.py` as one command. Run the full suite
before submitting changes that affect shared utilities or the inference pipeline.

Keep geometry in meters internally; BOP translation export converts to
millimeters. Preserve the separation between inference and reference evaluation.
Explicit oracle experiments must identify their use of ground-truth masks.
Select thresholds and checkpoints using validation data, and retain separate
train, validation, holdout and test scenes where the protocol defines them.

Use separate output directories for experiments. Keep datasets, checkpoints,
local caches and generated clouds out of commits; see `.gitignore`.
Do not replace measured reports with unmeasured estimates. When reporting a
benchmark, include its cohort, configuration, checkpoint and metric scope.

A pull request should explain the concrete behavior change and relevant checks.
If a change needs GPU, OpenGL or checkpoint validation beyond the tests, state
what was run and any remaining limitation.
