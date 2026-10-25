# Brief: Phase 2, packaging & tooling

**Branch:** `refactor-phase2` (from `refactor`)

## Goal
Make `pyproject.toml` the single source of packaging and tool config, and add CI.

## Tasks
1. Move everything in `setup.py` into a PEP 621 `[project]` table in `pyproject.toml`
   (keep the name `lnrgst_mca` and the `src/` layout). Pin minimum versions for the deps.
   Keep `fslpy`. Put `pytest` under `[project.optional-dependencies] test`. Delete `setup.py`.
2. Keep black's config. Add ruff only for the rules flake8 already checks (don't turn on new rules).
3. Add `.github/workflows/ci.yml`: Python 3.10, `pip install -e ".[test]"`, then pytest, black --check, flake8.

## Files you may touch
`pyproject.toml`, `setup.py` (delete), `environment.yml`, `.flake8`, `.pre-commit-config.yaml`, `.github/**`

## Out of scope (other agents own these)
- Removing `from .constants import *` touches `invocation_utils.py` and `plot_utils.py`. Leave it.
- Moving notebooks.

## Done when
- `pip install -e ".[test]"` works in a fresh venv
- pytest (49+ tests), black and flake8 pass
- committed on the branch, not pushed
