import pathlib
import json
import argparse
from typing import Dict, List

#ppmi is a bids dataset there should be a cleaner way to create these invocations

BASELINE_SESSION = 'ses-BL'
# BASELINE_SESSION = 'ses-*'
ANATOMICAL = 'anat'
ACCUSITION = 'acq-sag3D'
MODALITY= 'T1w'
SUFFIX = '.nii.gz'
MOSUF = MODALITY + SUFFIX
ORIGINAL = 'ieee'
MCA = 'mca'
PATTERN = pathlib.Path('') / 'sub-*' / BASELINE_SESSION / ANATOMICAL / f'sub-*_{BASELINE_SESSION}_{ACCUSITION}_run-*_{MOSUF}'
REF = pathlib.Path.cwd() / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"

def find_scans(input_dir:pathlib.Path, pattern=PATTERN)->List[pathlib.Path]:

    return list(input_dir.glob(str(pattern)))

def scan_filed_dict(scan_path:pathlib.Path)-> Dict:
 
    subject_ID, session, _, run, modalitiy  = scan_path.name.split('_')
    return {
        'input_path': str(scan_path),
        'subject': subject_ID,
        'session': session,
        'run': run,
        'modality': modalitiy,
    }

def create_subject_map(input_dir: pathlib.Path, **kwargs):

    scans_paths = find_scans(input_dir) #how to add patterns?
    subjects_map = {}
    for scan_path in scans_paths:

        scan_fields = scan_filed_dict(scan_path)
        subject = scan_fields['subject']
        session = scan_fields['session']

        if subject not in subjects_map:
            subjects_map[subject] = {}

        if session not in subjects_map[subject]:
            subjects_map[subject][session] = {}

        subjects_map[subject][session] = scan_fields

    return subjects_map

#think about more than one session later

def create_flirt_invocation(subject:Dict, output_dir:pathlib.Path, reference=REF, dofs=12)->Dict:

    in_file = subject['input_path']
    out_file = output_dir / f"{subject['subject']}_{subject['session']}{SUFFIX}"
    out_matrix_file = out_file.with_suffix('.mat')

    invocations = {
        'in_file': str(in_file),
        'out_file': str(out_file),
        'out_matrix_file': str(out_matrix_file),
        'reference': str(reference),
        'dof': dofs
    }      

    return invocations

def write_invocation(subject_ID, session, invocation:Dict, output_dir:pathlib.Path, dry_run:bool=False):

    invocation_path = output_dir / f'{subject_ID}_{session}_invocation.json'
    if dry_run:
        print(f'Writing invocations to {invocation_path}')
        json.dump(invocation, invocation_path, indent=4)
    else:
        with open(invocation_path, 'w') as f:
            json.dump(invocation, f, indent=4)


def create_ieee_invocations(subjects_map:Dict, output_dir:pathlib.Path, invocation_dir:pathlib.Path, reference=REF, dofs=12, dry_run:bool=False):
    
    ieee_invocation_dir = invocation_dir / f'{ANATOMICAL}-{dofs}dofs' / ORIGINAL

    if dry_run:
        print(f'Invocations exist on {ieee_invocation_dir}')
    else:
        ieee_invocation_dir.mkdir(parents=True, exist_ok=True)

        for subject in subjects_map:
            for session in subjects_map[subject]:
                invocation = create_flirt_invocation(subjects_map[subject][session], output_dir, reference, dofs)
                subject_ID = subjects_map[subject][session]['subject']
                write_invocation(subject_ID, session, invocation, ieee_invocation_dir, dry_run)

def create_mca_invocations(subjects_map:Dict, output_dir:pathlib.Path, invocation_dir:pathlib.Path, reference=REF, dofs=12, n_mca:int=10, dry_run:bool=False):
    
    mca_invocation_dir = invocation_dir / f'{ANATOMICAL}-{dofs}dofs' / MCA

    if dry_run:
        print(f'Invocations exist on {mca_invocation_dir}')
    else:
        mca_invocation_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_mca):
        current_mca_invocation_dir = mca_invocation_dir / str(i+1)
        current_mca_invocation_dir.mkdir(parents=True, exist_ok=True)

        for subject in subjects_map:
            for session in subjects_map[subject]:

                invocation = create_flirt_invocation(subjects_map[subject][session], output_dir)
                subject_ID = subjects_map[subject][session]['subject']
                write_invocation(subject_ID, session, invocation, current_mca_invocation_dir, dry_run)

def make_flirt_invocation(input_dir:pathlib.Path, output_dir:pathlib.Path, invocation_dir:pathlib.Path, reference=REF, dofs:List=[12], n_mca:int=10, dry_run:bool=False):

    for dof in dofs:
        subjects_map = create_subject_map(input_dir)
        create_ieee_invocations(subjects_map, output_dir, invocation_dir, reference, dof, dry_run)
        create_mca_invocations(subjects_map, output_dir, invocation_dir, reference, dof, n_mca, dry_run)

def parse_args():

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='create invocation for MCA and Original data')
    parser.add_argument('--input_dir', type=str, help='path to the input directory')
    parser.add_argument('--output_dir', type=str, help='path to the output directory')
    parser.add_argument('--n_mca', type=int, default=10, help='number of MCA repettions')
    parser.add_argument('--dry-run', action='store_true', help='Dry run')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    invocation_dir = output_dir / 'invocations'
    make_flirt_invocation(input_dir, output_dir, invocation_dir, n_mca=args.n_mca, dry_run=args.dry_run)

    # input_dir = pathlib.Path.cwd() / 'test_ppmi_hc'
    # a = find_scans(input_dir)

    # # for i in a:
    # #     print(i)

    # # print(len(a))
    
    # b = create_subject_map(input_dir)
    # # for i in b:
    # #     print(i)
    # #     # print(k)
    # #     # print('---')
    # #     print(b[i]['ses-BL'])
    # # print(create_flirt_invocation(b['sub-110350']['ses-BL'], input_dir))
    # # write_invocation(create_flirt_invocation(b['109910']['ses-BL'], input_dir), input_dir)
    # create_ieee_invocations(b, input_dir, input_dir.parent)
    # create_mca_invocations(b, input_dir, input_dir.parent)

    # # print(b)