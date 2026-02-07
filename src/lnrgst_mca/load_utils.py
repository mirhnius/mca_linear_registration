from re import sub
import numpy as np
import scipy
from pathlib import Path
from typing import Union

IEEE = "ieee"
MCA = "mca"
PATTERN = "*.mat"


def _ensure_4x4(mat: np.ndarray) -> np.ndarray:
    """ Standardize an existing matrix (3x4 or 4x4) to 4x4 shape.
    This function expects the input to already be a 2D matrix."""


    if mat.ndim != 2:
        raise ValueError(f"Input matrix must be 2D, got shape {mat.shape}.")   
        
    if mat.shape == (3,4):
        return np.vstack([mat, np.array([[0,0,0,1]])])
    
    if mat.shape == (4,4):
        return mat
    
    raise ValueError(f"Invalid matrix shape {mat.shape}. Expected (3, 4) or (4, 4).")


def is_matlab_file(filename: Union[str, Path]) -> bool:
    """
    Returns True if the file is a valid MATLAB file, False otherwise.
    """
    try:
        scipy.io.whosmat(str(filename))
        return True
    except Exception:
        return False
    

def load_file(filename: Union[str, Path]) -> np.ndarray:
    """
    Universal matrix loader.
    1. Detects file type.
    2. Parses sepcific format (split or reshape as needed).
    3. Standardizes to 4x4 shape.

    """

    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found.")
    
    try:
        if is_matlab_file(path):
            mat_dict = scipy.io.loadmat(str(path))
            #safely get the first variable that is not metadata
            key = next(k for k in mat_dict.keys() if not k.startswith("__"))
            raw_data = np.squeeze(mat_dict[key])  # Remove singleton dimensions

            #ANTS split format
            if raw_data.size ==  12:
                rotation = raw_data[:9].reshape(3,3)
                translation = raw_data[9:].reshape(3,1)
                mat = np.hstack((rotation, translation))
            else:
                mat = raw_data.reshape(-1, 4)  # Reshape to 2D if needed
        else:
            mat = np.loadtxt(str(path)).reshape(-1, 4)  # Reshape to 2D if needed    

        return _ensure_4x4(mat)
    
    except Exception as e:
        raise RuntimeError(f"Error loading {filename}: {e}") from e

def create_subject_list(inputfile: Union[str, Path], outputfile: Union[str, Path]):
    """
    Write subject IDs (directory names) from a file of paths to a new file.

    - inputfile: text file with one directory path per line
    - outputfile: destination text file with one subject ID per line

    """
    inputfile = Path(inputfile)
    outputfile = Path(outputfile)

    dir_names = []
    with open(inputfile, "r") as infile:
        for line in infile:
            path = Path(line.rstrip())  # removing the new line character
            dir_names.append(path.name + "\n")

    with open(outputfile, "w") as outfile:
        for dir in dir_names:
            outfile.write(dir)

    
def get_paths(parent_dir: Union[str, Path], subjects_file: Union[str, Path], n_mca: int = 10, pattern: str = "", ext: str = ".mat"):
    """
    Generate IEEE and MCA paths based on a list of subjects and read from a file.

    Parameters:
        parent_dir(Path): The parent dirctory containing the IEEE and MCA directories.
        subjects_file(Path): A file containing the list of subjects.
        n_mca(int): The number of MCA directories.
        pattern(str): The file name pattern.
        ext(str): The file extension.

    Returns:
        A directory with subject IDs as keys, ech containing paths to respective IEEE and MCA files.
    """
    # Read the subjects from the file
    subjects_file = Path(subjects_file)
    parent_dir = Path(parent_dir)

    subjects = []
    with open(subjects_file, "r") as file:
        for line in file:
            subjects.append(line.strip())

    # Generate the paths
    paths = {}
    for sub in subjects:
        
        filename = f"{sub}{pattern}{ext}"
        ieee_path = parent_dir / IEEE / filename
        mca_paths = [parent_dir / MCA / str(i) / filename for i in range(1, n_mca + 1)]

        paths[sub] = {IEEE: str(ieee_path), MCA: [str(p) for p in mca_paths]}

    return paths



def get_matrices(paths: dict):
    """
    Load the matrices from the paths.

    Parameters:
        paths(dict): The paths to the matrices.

    Returns:
        A dictionary containing the matrices.
        errors: A list of errors encountered during loading.
    """
    matrices = {}
    errors = []

    for sub, path_info in paths.items():
        sub_data = {}  #Temporary dictionary
        
        # 1. Load IEEE (Reference)
        try:
            sub_data[IEEE] = load_file(path_info[IEEE])
        except Exception as e:
            # Change 4: Better error message
            errors.append(f"Subject {sub} [IEEE Load Failed]: {e}")
            continue 

        # 2. Load MCA (Iterations)
        mca_matrices = []
        for mca_path in path_info[MCA]:
            try:
                mca_matrices.append(load_file(mca_path))
            except Exception as e:
                errors.append(f"Subject {sub} [MCA Iteration Failed]: {e}")
                continue
        
        # 3. Finalize Subject
        if mca_matrices:
            sub_data[MCA] = np.array(mca_matrices)
            matrices[sub] = sub_data 
        else:
            errors.append(f"Subject {sub}: No valid MCA matrices loaded.")

    return matrices, errors


def get_matrices_tensor(paths: dict):
    """
    Loads matrices into 4D Tensors for vectorized analysis.
    Returns: mca_tensor, ieee_tensor, subject_ids, errors
    """
    matrices, errors = get_matrices(paths)
    
    subject_ids = sorted(matrices.keys())
    mca_list = []
    ieee_list = []
    
    for sub in subject_ids:
        mca_list.append(matrices[sub][MCA])
        ieee_list.append(matrices[sub][IEEE])
        
    if not mca_list:
        return np.array([]), np.array([]), [], errors
        
    return np.array(mca_list), np.array(ieee_list), subject_ids, errors


if __name__ == "__main__":

    create_subject_list(Path("./PD_selected_paths.txt"), "./PD_selected_subjects.txt")
    create_subject_list(Path("./HC_selected_paths.txt"), "./HC_selected_subjects.txt")
    # subfile = Path().cwd() / "sub_list_test.txt"
    # test_path = Path().cwd() / "pipline" / "hc" / "outputs" / "ants" / "anat-12dofs"
    # paths = get_paths(test_path, subfile, pattern="_ses-BL0GenericAffine")
    # m, e = get_matrices(paths)
    # print(e)
