# CLAUDE.md

Project rules that every Claude session (and subagent) in this repo reads automatically.

## Project
Research code for "Numerical Uncertainty in Linear Registration" (MCA on FLIRT / ANTs / SPM).
It's being refactored into the `lnrgst_mca` package; see `REFACTOR_PLAN.md` for the phases.

## Setup & checks
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[test]" black==24.4.2 flake8
.venv/bin/python -m pytest -q          # must pass before any commit
.venv/bin/black --config pyproject.toml src tests
.venv/bin/flake8 --config .flake8 src tests
```

## Rules
- Git: never add `Co-Authored-By: Claude` or any Claude/Anthropic mention to commits.
  Don't open PRs and don't push unless told to. Integration happens on the `refactor` branch.
- Results must not change. Refactors keep numerical outputs and invocation JSONs identical;
  if a change is unavoidable, stop and explain why.
- Stay inside the files your brief lists. If you need to touch another file, stop and report it.
- Data lives on the cluster (`/home/niusham/...`), not on this machine. Don't try to run the
  full pipeline; use the tests in `tests/`.
- Don't edit notebooks in `src/*.ipynb` unless the brief says to.

## Layout
- `src/lnrgst_mca/`: package (load_utils, metrics_utils, invocation_utils, plot_utils, constants)
- `src/*.py`: analysis scripts (within/between software, QC plots)
- `src/config.py`: experiment configuration (paths, failed-QC lists)
- `bash_scripts/`: SLURM jobs, `descriptors/`: Boutiques JSON, `docker_resources/`: containers
- `tests/`: pytest suite; `tests/conftest.py` builds affine fixtures for each tool format
