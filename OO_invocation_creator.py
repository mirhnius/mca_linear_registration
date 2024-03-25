
import json
import argparse
import pathlib 
from typing import Dict, List
from abc import ABC, abstractmethod

BASELINE_SESSION = 'ses-BL'
ANATOMICAL = 'anat'
ACQUISITION = 'acq-sag3D'
MODALITY= 'T1w'
RUN = 'run-01'
SUFFIX = '.nii.gz'
MOSUF = MODALITY + SUFFIX
ORIGINAL = 'ieee'
MCA = 'mca'
FLIRT = 'flirt'
ANTS = 'ants'
MAT = '.mat'

PATTERN = pathlib.Path('') / 'sub-*' / BASELINE_SESSION / ANATOMICAL / f'sub-*_{BASELINE_SESSION}_{ACQUISITION}_{RUN}_{MOSUF}'
REF = pathlib.Path.cwd() / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"


def read_subjects_paths(file_path:str)->List[str]:
    """
    Read a file containing paths to subjects' directories and returns a list of paths.

    Paremeter:
        file_path: str, path to the file containing the subjects' paths.

    Return:
        List[str]: A list of subjects' paths.
    """

    if file_path is None:
        return None
    
    with open(file_path, 'r') as f:
        subjects_paths = f.readlines()
    return [subject_path.strip() for subject_path in subjects_paths]    


def find_scans(input_dir:pathlib.Path, pattern=PATTERN, sub_dirs:List[str]=None)->List[pathlib.Path]:
    """
    Finds and returns a list of scan paths based on the specified pattern.

    Parameters:
        input_dir (pathlib.Path): The directory to search within.
        pattern (pathlib.Path): The pattern to search for.
        sub_dirs (List[str]): Optional list of subdirectories to limit the search.

    Returns:
        List[pathlib.Path]: A list of scan paths that match the specified pattern.
    """

    if sub_dirs is None:
        return list(input_dir.glob(str(pattern)))
    else:
        scan_paths = []
        pattern = pathlib.Path('') / BASELINE_SESSION / ANATOMICAL / f'sub-*_{BASELINE_SESSION}_{ACQUISITION}_{RUN}_{MOSUF}'

        for sub_dir in sub_dirs:
            scan_paths.extend(list(pathlib.Path(sub_dir).glob(str(pattern))))
    
            if len(list(pathlib.Path(sub_dir).glob(str(pattern)))) == 0:
                print(sub_dir)
        
        return scan_paths

def scan_filed_dict(scan_path:pathlib.Path)-> Dict:
    """
    Extracts and return scan-related fields from the scan path.

    Parameters:
        scan_path (pathlib.Path): The scan path to extract fields from.
    
    Returns:
        Dict: A dictionary containing the extracted fields.
    """
 
    subject_ID, session, _, run, modalitiy  = scan_path.name.split('_')
    return {
        'input_path': str(scan_path),
        'subject': subject_ID,
        'session': session,
        'run': run,
        'modality': modalitiy,
    }

def create_subject_map(input_dir: pathlib.Path, pattern=PATTERN, sub_dirs:List[str]=None)->Dict:
    """
    Creates and returns a map of subjects and their scans based on the specified pattern.

    Parameters:
        input_dir (pathlib.Path): The directory to search within.
        pattern (pathlib.Path): The pattern match filenames against.
        sub_dirs (List[str]): Optional list of subdirectories to limit the search.
    
        Returns:
            Dict: A dictionary containing the subjects and their scans.
        """

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
    """
    Updates the input path of the subjects in the subjects_map.

    Parameters:
        subjects_map (Dict): A dictionary containing the subjects and their scans.
        input_dir (pathlib.Path): The replacement input directory.

    Returns:
        Dict: A dictionary containing the subjects and their updated scan paths.
    """
    
    for subject in subjects_map:
        for session in subjects_map[subject]:
            subjects_map[subject][session]['input_path'] = str(input_dir / f"{subject}_{session}{SUFFIX}")
    
    return subjects_map

class Preprocessing(ABC):
    """
    Abstract class for preprocessing steps.
    """

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.Path, invocation_dir:pathlib.Path):
        """
        Initializes the Preprocessing class.

        Parameters:
            subjects_maps (Dict): A dictionary mapping subjects to their information.
            output_dir (pathlib.Path): The directory where the outputs will be saved.
            invocation_dir (pathlib.Path): The directory where the invocations will be saved.
            
        """
        self.subjects_maps = subjects_maps
        self.output_dir = output_dir
        self.invocation_dir = invocation_dir

    @abstractmethod
    def create_single_subject_invocation(self, subject:Dict)->Dict:
        """
        Creates and returns a single subject invocation dictionary.

        Parameters:
            subject (Dict): A dictionary containing the subject information.
        
        Returns:
            Dict: A dictionary containing the subject invocation for the preprocessing steps.

        """
        pass

    def write_invocation(self, subject_ID: str, session: str, invocation:Dict, dry_run:bool=False):
        """
        Writes a single preprocessing invocation to a JSON file.

        Given a subject ID, session, and invocation dictionary, this method constructs
        a file name and writes the invocation dictionary to a JSON file within the invocation directory.

        Parameters:
            subject_ID (str): The subject ID.
            session (str): The session ID.
            invocation (Dict): The invocation dictionary.
            dry_run (bool): If True, the invocations are printed to the console instead of being written to files.
        """

        invocation_path = self.invocation_dir / f'{subject_ID}_{session}_invocation.json'
        if dry_run:
            print(f'Writing invocations to {invocation_path}')
            json.dump(invocation, invocation_path, indent=4)
        else:
            with open(invocation_path, 'w') as f:
                json.dump(invocation, f, indent=4)

    def create_invocations(self,dry_run:bool=False):
            """
            Generates and writes the preprocessing invocations JSON files for each subject.

            This method itereates throught all subjects and their respective sessions as defined in their subjects_maps,
            and generating a specific invocation for each subject and session. These invocations are then written to files
            within the invocation directory. Directories are created if they do not exist.

            Parameters:
                dry_run (bool): If True, the invocations are printed to the console instead of being written to files.
            
            Returns:
                None
            """
            if dry_run:
                print(f'Invocations exist on {self.invocation_dir}')
            else:
                self.invocation_dir.mkdir(parents=True, exist_ok=True)
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            for subject in self.subjects_maps:
                for session in self.subjects_maps[subject]:

                    invocation = self.create_single_subject_invocation(self.subjects_maps[subject][session])
                    subject_ID = self.subjects_maps[subject][session]['subject']
                    self.write_invocation(subject_ID, session, invocation, dry_run)


class BET_preprocessing(Preprocessing):

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.Path, invocation_dir:pathlib.Path, f:float=0.5):
        """
        Initializes the BET preprocessing class.

        Parameters:
            subjects_maps (Dict): A dictionary mapping subjects to their scans and associated data.
            output_dir (pathlib.Path): The directory where the output BET files will be saved.
            invocation_dir (pathlib.Path): The directory where the invocations will be saved.
            f (float): The fractional intensity threshold for BET. Deafult is 0.5.
    
        """
        super().__init__(subjects_maps, output_dir, invocation_dir)
        self.f = f

    def create_single_subject_invocation(self, subject:Dict)->Dict:
        """
        Generates the BET preprocessing invocation for a single subject.

        Parameters:
            subject (Dict): A dictionary containing details of the subject, including the input path.

        Returns:
            Dict: BET invocation parameters including input file and mask file paths and fractional intensity.
        """

        in_file = subject['input_path']
        out_file = self.output_dir / f"{subject['subject']}_{subject['session']}{SUFFIX}"

        invocation = {
            'infile': str(in_file),
            'maskfile': str(out_file), #output
            "fractional_intensity": self.f
        } 
        return invocation
    
class ROBUSTFOV_preprocessing(Preprocessing):
    """
    Initializes the ROBUSTFOV preprocessing class.

    Parameters:
        subjects_maps (Dict): A dictionary mapping subjects to their scans and associated data.
        output_dir (pathlib.Path): The directory where the output ROBUSTFOV files will be saved.
        invocation_dir (pathlib.Path): The directory where the invocations will be saved.
    """

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.Path, invocation_dir:pathlib.Path):
        super().__init__(subjects_maps, output_dir, invocation_dir)
    

    def create_single_subject_invocation(self, subject:Dict)->Dict:

        in_file = subject['input_path']
        out_file = self.output_dir / f"{subject['subject']}_{subject['session']}{SUFFIX}"

        invocation = {
        'in_file': str(in_file),
        'out_roi': str(out_file)
        }
        return invocation
    

class Registration(Preprocessing):

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.Path, invocation_dir:pathlib.Path, ref:str=REF, dof:int=12):
        """
        Initializes the preprocessing class.

        Parameters:
            subjects_maps (Dict): A dictionary mapping subjects to their scans and associated data.
            output_dir (pathlib.Path): The directory where the output FLIRT files will be saved.
            invocation_dir (pathlib.Path): The directory where the invocations will be saved.
            ref (str): The reference file for FLIRT. Default is MNI152.
            dof (int): The degrees of freedom. Default is 12.
        """
        super().__init__(subjects_maps, output_dir, invocation_dir)
        self.ref = ref
        self.dofs = dof

    
class FLIRT_IEEE_registration(Registration):

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.Path, invocation_dir:pathlib.Path, ref:str=REF, dof:int=12):
        """
        Initializes the FLIRT preprocessing class for IEEE standard with subject maps, output directory, invocation directory, reference image, and degrees of freedom.
        
        Parameters:
            subjects_maps (Dict): A dictionary mapping subjects to their scans and associated data.
            output_dir (pathlib.Path): The directory where the output FLIRT IEEE files will be saved.
            invocation_dir (pathlib.Path): The directory where the FLIRT IEEE invocation files will be written.
            ref (str): Path to the reference image against which registration is performed.
            dof (int): Degrees of freedom to be used by FLIRT for image registration. Default is 12.
        """
        super().__init__(subjects_maps, output_dir, invocation_dir, ref, dof)
       
        self.output_dir = self.output_dir / FLIRT/ f'anat-{str(self.dofs)}dofs' / ORIGINAL
        self.invocation_dir = self.invocation_dir / FLIRT / f'anat-{str(self.dofs)}dofs' / ORIGINAL

    def create_single_subject_invocation(self, subject:Dict)->Dict:

        """
        Generates and returns the FLIRT preprocessing invocation for a single subject.

        Parameters:
            subject (Dict): A dictionary containing details of the subject, including the input path.

        Returns:
            Dict: FLIRT invocation parameters including input file, reference file, output file, and degrees of freedom.
        """

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

    def create_invocations(self, dry_run: bool = False):
        return super().create_invocations(dry_run)
    

class FLIRT_MCA_registration(FLIRT_IEEE_registration):

    def __init__(self, subjects_maps:Dict, output_dir:pathlib.Path, invocation_dir:pathlib.Path, ref:str=REF, n_mca:int=10, dof:int=12):
        """
        Initializes the FLIRT registration class for MCA (Monte Carlo Arithmetic) with subject maps, output directory, invocation directory, reference image, degrees of freedom, and the number of MCA iterations.
        
        Parameters:
            subjects_maps (Dict): A dictionary mapping subjects to their scans and associated data.
            output_dir (pathlib.Path): The directory where the output FLIRT MCA files will be saved.
            invocation_dir (pathlib.Path): The directory where the FLIRT MCA invocation files will be written.
            ref (str): Path to the reference image against which registration is performed.
            n_mca (int): Number of MCA iterations to be performed for the preprocessing step.
            dof (int): Degrees of freedom to be used by FLIRT for image registration. Default is 12.
        """
        super().__init__(subjects_maps, output_dir, invocation_dir, ref, dof)
        self.n_mca = n_mca
        self.output_dir = self.output_dir.parent / MCA
        self.invocation_dir = self.invocation_dir.parent / MCA


    def create_invocations(self, dry_run: bool = False):
        """
        Generates and writes the FLIRT MCA registration invocations JSON files for each subject
        across specified neumebr of MCA iterations. Each iteration potentially generates a slightly different
        output file due to the randomness of the MCA algorithm.

        Parameters:
            dry_run (bool): If True, the invocations are printed to the console instead of being written to files.
        
        Returns:
            None
        """

        for i in range(self.n_mca):

            self.output_dir = self.output_dir / f"{i+1}"
            self.invocation_dir = self.invocation_dir / f"{i+1}"
            super().create_invocations(dry_run)
            self.output_dir = self.output_dir.parent
            self.invocation_dir = self.invocation_dir.parent

class ANTS_IEEE_registration(Registration):
     
    def __init__(self, subjects_maps:Dict, output_dir:pathlib.Path, invocation_dir:pathlib.Path, ref:str=REF, dof:int=12, t:str='a'):
        """
        Initializes the ANTS registration class for IEEE standard with subject maps, output directory, invocation directory, reference image, and degrees of freedom.
        
        Parameters:
            subjects_maps (Dict): A dictionary mapping subjects to their scans and associated data.
            output_dir (pathlib.Path): The directory where the output ANTS IEEE files will be saved.
            invocation_dir (pathlib.Path): The directory where the ANTS IEEE invocation files will be written.
            ref (str): Path to the reference image against which registration is performed.
            dof (int): Degrees of freedom to be used by ANTS for image registration. Default is 12.
            t (str): transformation mode 
        """
        super().__init__(subjects_maps, output_dir, invocation_dir, ref, dof)
        self.t = t
        self.output_dir = self.output_dir / ANTS / f'anat-{str(self.dofs)}dofs' / ORIGINAL
        self.invocation_dir = self.invocation_dir / ANTS / f'anat-{str(self.dofs)}dofs' / ORIGINAL

    def create_single_subject_invocation(self, subject:Dict)->Dict:

        """
        Generates and returns the ANTS preprocessing invocation for a single subject.

        Parameters:
            subject (Dict): A dictionary containing details of the subject, including the input path.

        Returns:
            Dict: FLIRT invocation parameters including input file, reference file, output prefix, and registration mode.
        """

        in_file = subject['input_path']
        output_prefix = self.output_dir / f"{subject['subject']}_{subject['session']}"
        
        invocation = {
        'moving': str(in_file),
        'fixed': str(self.ref),
        'output_prefix': str(output_prefix),
        'transform_type': self.t
        }
        return invocation

class ANTS_MCA_registration(ANTS_IEEE_registration):
    def __init__(self, subjects_maps:Dict, output_dir:pathlib.Path, invocation_dir:pathlib.Path, ref:str=REF, n_mca:int=10, dof:int=12, t:str="a"):
        """
        Initializes the ANTS registration class for MCA (Monte Carlo Arithmetic) with subject maps, output directory, invocation directory, reference image, degrees of freedom, and the number of MCA iterations.
        
        Parameters:
            subjects_maps (Dict): A dictionary mapping subjects to their scans and associated data.
            output_dir (pathlib.Path): The directory where the output ANTS MCA files will be saved.
            invocation_dir (pathlib.Path): The directory where the ANTS MCA invocation files will be written.
            ref (str): Path to the reference image against which registration is performed.
            n_mca (int): Number of MCA iterations to be performed for the preprocessing step.
            dof (int): Degrees of freedom to be used by ANTS for image registration. Default is 12.
            t (str): transformation mode 
        """
        super().__init__(subjects_maps, output_dir, invocation_dir, ref, dof, t)
        self.n_mca = n_mca
        self.output_dir = self.output_dir.parent / MCA
        self.invocation_dir = self.invocation_dir.parent / MCA


    def create_invocations(self, dry_run: bool = False):
        """
        Generates and writes the ANTS MCA registration invocations JSON files for each subject
        across specified neumebr of MCA iterations. Each iteration potentially generates a slightly different
        output file due to the randomness of the MCA algorithm.

        Parameters:
            dry_run (bool): If True, the invocations are printed to the console instead of being written to files.
        
        Returns:
            None
        """

        for i in range(self.n_mca):

            self.output_dir = self.output_dir / f"{i+1}"
            self.invocation_dir = self.invocation_dir / f"{i+1}"
            super().create_invocations(dry_run)
            self.output_dir = self.output_dir.parent
            self.invocation_dir = self.invocation_dir.parent   

def parse_args():

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='create invocation for MCA and Original data')
    parser.add_argument('--input_dir', type=str, help='path to the input directory')
    parser.add_argument('--output_dir', type=str, help='path to the output directory')
    parser.add_argument('--invocation_dir', type=str, default=pathlib.Path().cwd()/'invocations', help='path to invocation directory')
    parser.add_argument('--n_mca', type=int, default=10, help='number of MCA repettions')
    parser.add_argument('--dry-run', action='store_true', help='Dry run')
    # parser.add_argument('--dofs', type=int, nargs='+', default=[12], help='Degrees of freedom for flirt')
    parser.add_argument('--input_subjects', type=str, default=None, help='input subjects paths')
  
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    invocation_dir = pathlib.Path(args.invocation_dir)
    sub_dirs = read_subjects_paths(args.input_subjects)
    input_subjects = args.input_subjects 
    
    robustfov_input_dir = input_dir
    robustfov_output_dir = output_dir / 'preprocess'
    robustfov_invocation_dir = invocation_dir / 'robustfov'
    bet_invocation_dir = invocation_dir / 'bet'
    flirt_invocation_dir = invocation_dir / 'flirt'
    bet_output_dir = robustfov_output_dir

    subjects_map = create_subject_map(robustfov_input_dir, sub_dirs=sub_dirs)
    subjects_map_after_preprocess = updating_subject_map(subjects_map, robustfov_output_dir)

    ROBUSTFOV_preprocessing(subjects_map, robustfov_output_dir, robustfov_invocation_dir).create_invocations(dry_run=args.dry_run)
    
    BET_preprocessing(subjects_map_after_preprocess, bet_output_dir, bet_invocation_dir).create_invocations(dry_run=args.dry_run)
   
    FLIRT_IEEE_registration(subjects_map_after_preprocess, output_dir, flirt_invocation_dir).create_invocations(dry_run=args.dry_run)

    FLIRT_MCA_registration(subjects_map_after_preprocess, output_dir, flirt_invocation_dir, n_mca=args.n_mca).create_invocations(dry_run=args.dry_run)
    