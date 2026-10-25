# Brief: Phase 4, invocation generation

**Branch:** `refactor-phase4` (from `refactor`)

## Goal
Collapse the six registration classes in `src/lnrgst_mca/invocation_utils.py`
(`{FLIRT,ANTS,SPM}_{IEEE,MCA}_registration`) into one builder per tool plus a single
MCA-repetition helper.

## Contract (must not change)
The invocation JSON written for every subject, and where it's written:
`<inv>/<tool>/<template>/anat-<dof>dofs/{ieee | mca/<1..n>}/<sub>_<ses>_invocation.json`.
Before refactoring, generate invocations for a fake BIDS tree (see `tests/test_invocation_utils.py`)
and snapshot them. Afterwards they must be byte-identical.

## Tasks
1. Compute output and invocation folders with a pure function, e.g. `registration_dir(root, tool, template, dof, mode, run=None)`.
   Don't change `self.output_dir` inside loops.
2. Add one MCA wrapper/helper that repeats any registration step `n_mca` times.
3. Keep the old class names as thin aliases for now, because `different_metrics_invo.py` and
   the notebooks import them.
4. Add a `--cost_function` option to `parse_args` (default `corratio`) and pass it to FLIRT.

## Files you may touch
`src/lnrgst_mca/invocation_utils.py`, `src/different_metrics_invo.py`, `tests/test_invocation_utils.py`

## Done when
- snapshot comparison is identical
- pytest, black and flake8 pass
- committed on the branch, not pushed
