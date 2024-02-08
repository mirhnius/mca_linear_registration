import pathlib
import json
import argparse

BASELINE_SESSION = 'ses-BL'
ANATOMICAL = 'anat'
ACCUSITION = 'acq-sag3D'
T1_scan_suffix = 'T1w.nii.gz'
PATTERN = pathlib.Path('') / 'sub-*' / BASELINE_SESSION / ANATOMICAL / f'sub-*_{BASELINE_SESSION}_{ACCUSITION}_run-*_{T1_scan_suffix}'

def find_scans(input_dir, pattern=PATTERN):
    return list(input_dir.glob(str(pattern)))

def parse_args():

    parser = argparse.ArgumentParser(description='create invocation for MCA and Original data')
    parser.add_argument('--input_dir', type=str, help='path to the input directory')
    parser.add_argeument('--output_dir', type=str, help='path to the output directory')
    parser.add_argument('--n_mca', type=int, default=10, help='number of MCA repettions')
    parser.add_argument('--dry-run', action='store_true', help='Dry run')

    return parser.parse_args()

if __name__ == '__main__':
    input_dir = pathlib.Path.cwd() / 'test_ppmi_hc'
    a = find_scans(input_dir)

    for i in a:
        print(i)

    print(len(a))
    