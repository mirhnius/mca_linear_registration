
import json
import argparse
import pathlib 
from typing import Dict, List
from abc import ABC, abstractmethod

BASELINE_SESSION = 'ses-BL'
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


def read_subjects_paths(file_path:str)->List[str]:

    if file_path is None:
        return None
    
    with open(file_path, 'r') as f:
        subjects_paths = f.readlines()
    return [subject_path.strip() for subject_path in subjects_paths]    


def find_scans(input_dir:pathlib.Path, pattern=PATTERN, sub_dirs:List[str]=None)->List[pathlib.Path]:

    if sub_dirs is None:
        return list(input_dir.glob(str(pattern)))
    else:
        scan_paths = []
        pattern = pathlib.Path('') / BASELINE_SESSION / ANATOMICAL / f'sub-*_{BASELINE_SESSION}_{ACCUSITION}_{RUN}_{MOSUF}'

        for sub_dir in sub_dirs:
            scan_paths.extend(list(pathlib.Path(sub_dir).glob(str(pattern))))
    
            if len(list(pathlib.Path(sub_dir).glob(str(pattern)))) == 0:
                print(sub_dir)
        
        return scan_paths

def scan_filed_dict(scan_path:pathlib.Path)-> Dict:
 
    subject_ID, session, _, run, modalitiy  = scan_path.name.split('_')
    return {
        'input_path': str(scan_path),
        'subject': subject_ID,
        'session': session,
        'run': run,
        'modality': modalitiy,
    }

def create_subject_map(input_dir: pathlib.Path, pattern=PATTERN, sub_dirs:List[str]=None)->Dict:

    scans_paths = find_scans(input_dir, pattern, sub_dirs) #how to add patterns?
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

def updating_subject_map(subjects_map:Dict, input_dir:pathlib.Path)->Dict:
    
    for subject in subjects_map:
        for session in subjects_map[subject]:
            subjects_map[subject][session]['input_path'] = str(input_dir / f"{subject['subject']}_{subject['session']}{SUFFIX}")
    return subjects_map

class Preprocessing(ABC):

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.path, invocation_dir:pathlib.Path):
        self.subjects_maps = subjects_maps
        self.output_dir = output_dir
        self.invocation_dir = invocation_dir

    @abstractmethod
    def create_single_subject_invocation(self, subject:Dict)->Dict:
        pass

    def write_invocation(self, subject_ID: str, session: str, invocation:Dict, output_dir:pathlib.Path, dry_run:bool=False):

        invocation_path = self.invocation_dir / f'{subject_ID}_{session}_invocation.json'
        if dry_run:
            print(f'Writing invocations to {invocation_path}')
            json.dump(invocation, invocation_path, indent=4)
        else:
            with open(invocation_path, 'w') as f:
                json.dump(invocation, f, indent=4)

    def create_invocations(self,dry_run:bool=False):
            if dry_run:
                print(f'Invocations exist on {self.invocation_dir}')
            else:
                self.invocation_dir.mkdir(parents=True, exist_ok=True)
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            for subject in self.subjects_maps:
                for session in self.subjects_maps[subject]:

                    invocation = self.create_single_subject_invocation(self.subjects_maps[subject][session], self.output_dir)
                    subject_ID = self.subjects_maps[subject][session]['subject']
                    self.write_invocation(subject_ID, session, invocation, self.invocation_dir, dry_run)

class BET_preprocessing(Preprocessing):

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.path, invocation_dir:pathlib.Path, f:float=0.5):
        super().__init__(subjects_maps, output_dir, invocation_dir)
        self.f = f

    def create_single_subject_invocation(self, subject:Dict)->Dict:

        in_file = subject['input_path']
        out_file = self.output_dir / f"{subject['subject']}_{subject['session']}{SUFFIX}"

        invocation = {
            'infile': str(in_file),
            'maskfile': str(out_file), #output
            "fractional_intensity": self.f
        } 
        return invocation
    
class ROBUSTFOV_preprocessing(Preprocessing):

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.path, invocation_dir:pathlib.Path):
        super().__init__(subjects_maps, output_dir, invocation_dir)
    

    def create_single_subject_invocation(self, subject:Dict)->Dict:

        in_file = subject['input_path']
        out_file = self.output_dir / f"{subject['subject']}_{subject['session']}{SUFFIX}"

        invocation = {
        'in_file': str(in_file),
        'out_roi': str(out_file)
        }
        return invocation
    

class FLIRT_preprocessing(Preprocessing):

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.path, invocation_dir:pathlib.Path, ref:str=REF, dof:int=12):
        super().__init__(subjects_maps, output_dir, invocation_dir)
        self.ref = ref
        self.dofs = dof

    def create_single_subject_invocation(self, subject:Dict)->Dict:

        in_file = subject['input_path']
        out_file = self.output_dir / f"{subject['subject']}_{subject['session']}{SUFFIX}"
        out_matrix_file = self.output_dir / f"{subject['subject']}_{subject['session']}{MAT}"

        invocation = {
        'in_file': str(in_file),
        'out_filename': str(out_file),
        'reference': str(self.ref),
        'out_matrix_filename': str(out_matrix_file),
        'dof': self.dofs
        }
        return invocation
    
class FLIRT_IEEE_preprocessing(FLIRT_preprocessing):

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.path, invocation_dir:pathlib.Path, ref:str=REF, dof:int=12):
        super().__init__(subjects_maps, output_dir, invocation_dir, ref, dof)
       
        self.output_dir = self.output_dir / ORIGINAL

    def create_invocations(self, dry_run: bool = False):
        return super().create_invocations(dry_run)
    

class FLIRT_MCA_preprocessing(FLIRT_preprocessing):

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.path, invocation_dir:pathlib.Path, ref:str=REF, n_mca:int=10, dof:int=12):
        super().__init__(subjects_maps, output_dir, invocation_dir, ref, dof)
        self.n_mca = n_mca
        self.output_dir = self.output_dir / MCA

    def create_invocations(self, dry_run: bool = False):

        for i in range(self.n_mca):

            self.output_dir = self.output_dir / f"{i}"
            super().create_invocations(dry_run)
            self.output_dir = self.output_dir.parent



