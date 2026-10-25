# Brief: Phase 7a, SLURM scripts

**Branch:** `refactor-phase7a` (from `refactor`)

## Goal
Replace the ~20 copy-paste scripts in `bash_scripts/` with one parametrized SLURM template
and a submit script.

## Tasks
1. Read every script in `bash_scripts/` and make a table (in your final report) of what varies:
   software, mode (ieee/mca), cohort, template, container image, descriptor, time, memory, array.
2. Write `scripts/slurm/run_bosh.sbatch`. It takes `SOFTWARE MODE COHORT TEMPLATE` (plus optional
   overrides) and builds the invocation glob and `bosh exec launch` command used today.
3. Write `scripts/slurm/images.env`, which maps software×mode to a container image path and descriptor,
   using the exact paths found in the current scripts.
4. Write `scripts/submit.sh`, which loops over a chosen set of combinations and calls `sbatch`
   with the right `--time`, `--mem-per-cpu` and `--array`.
5. Delete the replaced scripts only when the table shows the new template covers them.
   Keep `copy_from_server.sh` and `within_software_analysis.sh` as they are.

## Files you may touch
`bash_scripts/**`, `scripts/**`

## Done when
- `bash -n` passes on every new script
- a `--dry-run` flag in `submit.sh` prints the exact `sbatch` and `bosh` commands, and for at least
  FLIRT/ANTs/SPM × ieee/mca these match what the old scripts ran
- committed on the branch, not pushed
