# Refactoring Plan: research scripts → `lnrgst_mca` package

Goal: turn the repo into an installable, tested package with a small CLI, keeping
the paper's results reproducible at every step.

Guiding rule: **freeze outputs first, then refactor.** Each phase should finish
with the regression check (Phase 0) still passing.

---

## Current state (what the audit found)

### Bugs / broken code to fix first
| Where | Problem |
|---|---|
| `src/within_software_analysis.py:217-232` | Calls `metrics_utils.FD_all_subjects`, which no longer exists (renamed to `framewise_displacment_all_subjects_vectorized`). The script crashes. |
| `src/config.py` → `configurations_cost["corratio"]` | `path_PD`/`path_HC` point at the **`normcorr`** directory, so the CR results are really NCC results. |
| `invocation_utils.py` `__main__` | `handle_preprocessing(...)` is called without `dry_run`, and `prepare_preprocess_directories` runs `mkdir` even in `--dry-run`. |
| `invocation_utils.handle_reference` | Returns a `Path` in one branch and a `str` in the other. |
| `ANTS_MCA_registration` / `SPM_MCA_registration.create_invocations` | Mutate `self.output_dir` in a loop with no `try/finally` (FLIRT already does it the safe way), so an exception leaves the object in a broken state. |
| `metrics_utils.py`, `within_software_analysis.py` | Import `fsl.transform` (fslpy), which is missing from `setup.py` and `environment.yml`. |
| `.gitignore` | `test*` would ignore a future `tests/` dir, and `*.json` hides `descriptors/` (they're tracked only because they were force-added). |
| `config.get_configurations` `__main__` | Calls `get_configurations("FSL", "mni152")`, which isn't a valid key. |

### Structural problems
- **Config is code.** `src/config.py` holds 338 lines of nested dicts: hard-coded
  cluster paths (`/home/niusham/projects/...`), failed-QC subject lists repeated
  verbatim (the SPM/Asym list appears twice), and plotting bins and colors mixed with
  data paths. It also does I/O on import paths relative to `cwd`.
- **Two "constants" modules.** `lnrgst_mca/constants.py` and the top of
  `load_utils.py` both define `IEEE`/`MCA`/`PATTERN` with different meanings.
  `REF` is computed from `Path.cwd().parent.parent` at import time.
- **Star imports** (`from .constants import *`), so flake8 has to ignore F403/F405.
- **Registration class explosion.** There are 6 classes (`{FLIRT,ANTS,SPM}_{IEEE,MCA}`) where
  the MCA variants differ only by "loop over `n_mca` subdirs". Docstrings are
  copy-pasted (SPM's docstring says "ANTS").
- **Analysis scripts are 400-600 line `__main__` blocks.** `within_software_analysis.py`
  repeats every operation 4× (PD/HC × fine/failed) by hand, re-implements the
  outlier detection already in `metrics_utils`, and has inline imports.
  `between_softwares_and_versions.py` hard-codes a cluster path in `__main__`.
- **Large commented-out blocks** (abandoned ANOVA/Friedman/GMM experiments).
- **Plotting is coupled to analysis**: `plt.figure()`/`savefig` are called inside
  loops, with global state and no returned `Figure` objects.
- **Bash scripts are copy-paste variants.** There are 20 SLURM scripts that differ by
  software/cohort/template/array, and they contain typos and duplicates
  (`new_pipline_*` vs `new_pipeline_*`), hard-coded image paths and
  personal venvs.
- **Packaging:** `setup.py` plus a `pyproject.toml` that only configures black.
  There's no version pinning, no tests, and no CI. Notebooks live inside `src/`.
- **README** describes a `docker/ scripts/ analysis/ qc/ figures/` layout that
  doesn't exist.

---

## Target layout

```text
.
├── pyproject.toml              # single source: build, deps, black, ruff, pytest
├── environment.yml
├── configs/
│   ├── paths.example.yaml      # data roots (user copies to paths.yaml, git-ignored)
│   ├── experiments.yaml        # software × template × cost-function matrix
│   ├── qc_failures.yaml        # failed-QC subject IDs, one list per (software, template, cost)
│   └── plotting.yaml           # palettes, bin sizes, x-limits
├── src/lnrgst_mca/
│   ├── __init__.py
│   ├── constants.py            # BIDS entities, enums (Software, Template, Mode)
│   ├── config.py               # load + validate YAML → dataclasses
│   ├── io/
│   │   ├── bids.py             # find_scans, subject maps (from invocation_utils)
│   │   ├── matrices.py         # load_file, get_paths, get_matrices(_tensor)
│   │   └── subjects.py         # subject-list read/write (subject_selector, create_subject_list)
│   ├── invocations/
│   │   ├── base.py             # Step ABC + write/dry-run logic
│   │   ├── preprocessing.py    # RobustFOV, BET
│   │   └── registration.py     # FLIRT/ANTs/SPM builders + MCA repetition wrapper
│   ├── metrics/
│   │   ├── decomposition.py    # decompose_tensor
│   │   ├── displacement.py     # FD, MAD
│   │   ├── stats.py            # normality, Mann-Whitney, t-tests
│   │   └── outliers.py         # KDE / IsolationForest / OCSVM / percentile, recall
│   ├── analysis/
│   │   ├── within_software.py  # pure functions returning results objects
│   │   └── between_software.py
│   ├── plotting/
│   │   ├── distributions.py    # plotter, hist_plotter, swarm_QC (return Figures)
│   │   └── qc.py               # slice_plotter, GIFs, QC_plotter
│   └── cli.py                  # `lnrgst` entry point (typer/argparse subcommands)
├── scripts/
│   ├── slurm/run_bosh.sbatch   # ONE parametrized SLURM template
│   └── submit.sh               # loops the experiment matrix → sbatch
├── matlab/                     # run_affreg.m, octave_stability.m
├── docker/                     # current docker_resources/
├── descriptors/                # boutiques JSON (un-ignore in .gitignore)
├── notebooks/                  # moved out of src/; toy_nb/ → notebooks/scratch or deleted
├── data/subject_lists/         # HC/PD_selected_subjects.txt
└── tests/
    ├── fixtures/               # tiny .mat / .txt affines for each tool format
    └── test_*.py
```

---

## Phases

### Phase 0 — Safety net (do this first)
1. Fix `.gitignore` (drop `test*` and `*.json`, then add explicit ignores such as `invocations*/`).
2. Add `tests/` with pytest and small fixtures:
   - one FLIRT text affine, one ANTs `0GenericAffine.mat`, one SPM `.mat`
   - an identity matrix and a known rotation/translation matrix for FD.
3. **Golden-output regression**: run the current pipeline on a small subject subset
   (or on the cluster outputs) and snapshot `reports/*.txt`, FD arrays and invocation
   JSONs into `tests/golden/`. Every later phase is checked against these files.
4. Unit tests for the pure functions that already exist: `load_file`, `_ensure_4x4`,
   `decompose_tensor`, `framewise_displacment_all_subjects_vectorized`,
   `mean_absolute_difference`, `get_paths`, and the invocation dict builders.

### Phase 1 — Fix known bugs (small, separate commits)
- Replace `FD_all_subjects` calls with the vectorized FD function.
- Point the `corratio` config at the `corratio` directory, then **re-check the
  paper's CR numbers**.
- Thread `dry_run` through `handle_preprocessing` and skip `mkdir` in dry-run.
- Make `handle_reference` always return a `Path`.
- Add `fslpy` to the dependencies.

### Phase 2 — Packaging & tooling
- Move everything into `pyproject.toml` (PEP 621), delete `setup.py`, pin
  minimum versions, and add the `[project.scripts] lnrgst = "lnrgst_mca.cli:main"` entry point.
- Switch flake8 to **ruff** (lint + isort). Remove the `F403/F405` ignores by
  eliminating star imports. Drop the line length from 150 to ~110.
- Add a GitHub Actions workflow that runs ruff, black --check and pytest.
- Move notebooks to `notebooks/`, run `nbstripout` in pre-commit, and delete or archive `toy_nb/`.

### Phase 3 — Configuration out of code
- Split `src/config.py` into the YAML files under `configs/`. Failed-QC lists
  become data, not Python, and are defined once and referenced by key.
- Add `lnrgst_mca/config.py` with dataclasses (`ExperimentConfig`, `PathsConfig`)
  and a loader that builds paths from
  `root / cohort / outputs / [cost] / software / template / anat-{dof}dofs`
  instead of listing every combination by hand.
- Read data roots from `configs/paths.yaml` or env vars (`LNRGST_DATA_ROOT`).
  Absolute `/home/niusham/...` paths should no longer appear anywhere in the code.
- Remove import-time side effects (`REF = Path.cwd()...`).

### Phase 4 — Invocation generation
- Collapse the six registration classes into **three tool builders plus one
  MCA wrapper**:
  ```python
  class RegistrationStep(Step):          # builds one invocation dict
      tool: Software
      def build(self, scan, out_dir) -> dict: ...

  def expand_mca(step, n_mca) -> Iterable[(out_dir, inv_dir)]  # ieee/ or mca/1..n
  ```
  Output dirs are computed by a pure function (`layout.registration_dir(...)`)
  rather than mutated on `self`.
- Use `dataclass` for scan records instead of loosely typed dicts.
- Make `different_metrics_invo.py` a CLI option (`--cost-function`, `--scan-parser short`).

### Phase 5 — Analysis & metrics
- Split `metrics_utils.py` into `metrics/{decomposition,displacement,stats,outliers}.py`.
- Rewrite `within_software_analysis.py` around a **group abstraction** so you no
  longer hand-unroll every PD/HC × passed/failed combination:
  ```python
  groups = split_by_qc(load_cohorts(cfg), cfg.failed)   # {("PD","fine"): tensor, ...}
  fd = {g: compute_fd(t) for g, t in groups.items()}
  results = WithinSoftwareResult(fd=..., mad=..., tests=..., outliers=...)
  results.save(out_dir); plots.render(results, out_dir)
  ```
  Use the existing `metrics_utils.detect_outliers_*` helpers instead of the
  inline copies. Delete the commented-out experiments (they remain in git history).
- In `between_softwares_and_versions.py`, move parameters out of `__main__` into
  config/CLI flags and reuse the same result loaders.
- Choose one result format, e.g. a tidy `results.parquet/csv` with columns
  `subject, cohort, qc, software, template, cost, run, fd`. This replaces the many
  `*.txt` arrays that the between-software analysis has to re-parse by filename pattern.

### Phase 6 — Plotting
- Plot functions take data plus an `Axes` (or create a figure) and **return the
  `Figure`**. Saving is done separately by the caller.
- Move palettes, bins and limits into `configs/plotting.yaml`, keyed by
  enum values instead of raw strings.
- Fix the mutable default arguments (`palette=["orange","blue"]`) and typos (`ylable`).

### Phase 7 — CLI and HPC scripts
- Add a single CLI:
  ```bash
  lnrgst invocations --config configs/experiments.yaml --cohort pd [--dry-run]
  lnrgst analyze within --software flirt --template asym [--cost normmi] [--verrou]
  lnrgst analyze between
  lnrgst qc-plots ...
  lnrgst subjects select ...
  ```
  Replace `-v/--verrou type=bool` (any non-empty string evaluates to True) with a real flag.
- Replace the 20 bash scripts with **one** SLURM template parametrized by
  `SOFTWARE`, `MODE` (ieee/mca), `COHORT`, `TEMPLATE` and `IMAGE`, plus a submit
  script that iterates the experiment matrix. Container image paths come from config.
- Move the `.m` files to `matlab/`.

### Phase 8 — Docs
- Rewrite the README "Repository Structure" and add a "Reproduce the paper" section
  (data → invocations → bosh/SLURM → analysis → figures).
- Add docstrings and type hints on the public API, and fix the many typos
  (`framewise_displacment`, `scan_filed_dict`, `pipline`, `largest_indces`...).
  Keep deprecated aliases for one release if notebooks still use the old names.

---

## Suggested order & PR breakdown
1. PR 1: Phase 0 + Phase 1 (tests + bug fixes). This has the highest value and the lowest risk.
2. PR 2: Phase 2 (packaging/tooling/CI).
3. PR 3: Phase 3 (config to YAML).
4. PR 4: Phase 4 (invocations).
5. PR 5: Phases 5-6 (analysis + plotting). This is the largest PR, so verify it against the golden outputs.
6. PR 6: Phases 7-8 (CLI, SLURM, docs).

## Open questions
- `load_file` ignores the `fixed` center stored in ANTs `*0GenericAffine.mat` files. The true
  translation is `t + c - A·c`, so ANTs translations (and therefore FD) differ from the
  real affine whenever `A` varies across MCA runs. Is that acceptable for the FD-SD metric,
  or should the center be applied? This is pinned by `test_ants_ignores_fixed_center`.
- Is the `corratio → normcorr` path intentional (e.g. a re-run reused the folder), or a bug?
  The answer affects the paper's CR results.
- Is the cluster layout (`pipline/{pd,hc}/outputs/...`) fixed? Changing the directory
  name (`pipline` → `pipeline`) would break existing outputs unless the old name is
  supported through config.
- Are the Verrou experiments still part of the paper, or can they be archived?
- Which notebooks produce the paper figures (`finalized_plots.ipynb`? `code_paper.ipynb`?).
  Those should be converted into scripts or the CLI; the rest can be archived.
