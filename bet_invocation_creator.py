import pathlib
import json
import argparse
from typing import Dict, List
from invocation_creator import make_bet_invocations, create_subject_map, read_subjects, make_flirt_invocations, make_robustov_invocations

#solve the copy problem later

BASELINE_SESSION = 'ses-BL'
# BASELINE_SESSION = 'ses-*'
ANATOMICAL = 'anat'
ACCUSITION = 'acq-sag3D'
MODALITY= 'T1w'
RUN = 'run-01'
SUFFIX = '.nii.gz'
MOSUF = MODALITY + SUFFIX
ORIGINAL = 'ieee'
MCA = 'mca'
MAT = '.mat'

PATTERN = pathlib.Path('') / 'sub-*' / BASELINE_SESSION / ANATOMICAL / f'sub-*_{BASELINE_SESSION}_{ACCUSITION}_{RUN}_{MOSUF}'
REF = pathlib.Path.cwd() / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"


### it makes more sense to make it object oriented too much duplication 


if __name__ == '__main__':

    args = parse_args()
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    invocation_dir = pathlib.Path(args.invocation_dir)

    robustfov_invocation_dir = invocation_dir / 'robustfov'
    bet_invocation_dir = invocation_dir / 'bet'
    flirt_invocation_dir = invocation_dir / 'flirt'
    input_subjects = args.input_subjects 
    robustfov_input_dir = input_dir
    robustfov_output_dir = output_dir / 'preprocess'
    bet_output_dir = robustfov_output_dir

    subjects_map = create_subject_map(robustfov_input_dir, sub_dirs=input_subjects)
    make_robustov_invocations(subjects_map, robustfov_output_dir, robustfov_invocation_dir)
    subjects_map_after_preprocess = create_subject_map(robustfov_output_dir, pattern= pathlib.Path('')/f'sub-*_{BASELINE_SESSION}_{MOSUF}')
    make_bet_invocations(subjects_map_after_preprocess, bet_output_dir, bet_invocation_dir, f=0.3, dry_run=args.dry_run)
    make_flirt_invocations(subjects_map_after_preprocess, input_dir, output_dir, flirt_invocation_dir, n_mca=args.n_mca, dry_run=args.dry_run)