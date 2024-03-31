import numpy as np
import scipy
from pathlib import Path

IEEE = "ieee"
MCA = "mca"
PATTERN = "*.mat"


def create_subject_list(inputfile, outputfile):
    """
    This function receives a text file including the subjects' list
    and return the subjects' IDs as a text file.

    Parameter:
        inputfile: The input file's path.
        outputfile: The output file's path.

    Return:
        None

    """

    dir_names = []
    with open(inputfile, "r") as infile:
        for line in infile:
            path = Path(line.rstrip())  # removing the new line character
            dir_names.append(path.name + "\n")

    with open(outputfile, "w") as outfile:
        for dir in dir_names:
            outfile.write(dir)


def is_matlab_file(filename):
    """
    Check if a file is a matlab file.
    """
    try:
        scipy.io.whosmat(filename)
        return True
    except Exception:
        return False


def get_matrices_paths(parent_dir: Path, subjects_file: Path, n_mca: int = 10, pattern: str = None, ext: str = ".mat"):
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
    subjects = []
    with open(subjects_file, "r") as file:
        for line in file:
            subjects.append(line.strip())

    # Generate the paths
    paths = {}
    for sub in subjects:
        ieee_path = parent_dir / IEEE / f"{sub}{pattern}{ext}"
        mca_paths = [parent_dir / MCA / str(i) / f"{sub}{pattern}{ext}" for i in range(1, n_mca + 1)]

        paths[sub] = {IEEE: str(ieee_path), MCA: [str(p) for p in mca_paths]}

    return paths


def load_matlab_file(filename: str):
    """
    Load a matlab file and reshape it to affine matrix.

    Parameters:
        filename(str): The path to the file.

    Return:
        The affine matrix.
    """
    new_row = np.array([0, 0, 0, 1])
    try:
        mat = scipy.io.loadmat(filename)
        mat = next(iter(mat.values())).reshape((-1, 4))
        if mat.shape == (3, 4):  # adding the row to shape (4, 4) matrix
            mat = np.vstack((mat, new_row))
        return mat
    except Exception as e:
        raise RuntimeError(f"Error loading1 {filename}: {e}") from e


def load_text_file(filename: str):
    """
    Load a text file.

    Parameters:
        filename(str): The path to the file.

    Return:
        The affine matrix.
    """
    new_row = np.array([0, 0, 0, 1])
    try:

        mat = np.loadtxt(filename).reshape(-1, 4)
        if mat.shape == (3, 4):  # adding the row to shape (4, 4) matrix
            mat = np.vstack((mat, new_row))
        return mat

    except Exception as e:
        raise RuntimeError(f"Error loading2 {filename}: {e}") from e


def load_file(filename: str):
    """
    Load a file, either matlab or text.

    Parameters:
        filename(str): The path to the file.

    Return:
        The affine matrix.
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"{filename} not found.")

    if is_matlab_file(filename):
        return load_matlab_file(filename)
    return load_text_file(filename)


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
        try:
            matrices[sub] = {IEEE: load_file(path_info[IEEE])}
        except (FileNotFoundError, RuntimeError) as e:
            errors.append(f"Error loading3 {path_info[IEEE]}: {e}")
            continue

        mca_matrices = []
        for mca_path in path_info[MCA]:
            try:
                mca_matrices.append(load_file(mca_path))
            except (FileNotFoundError, RuntimeError) as e:
                errors.append(f"Error loading4 {mca_path}: {e}")
                continue
        if mca_matrices:
            # matrices[sub][MCA] = np.array(mca_matrices) refactor the nb later to use this
            matrices[sub][MCA] = mca_matrices

    return matrices, errors


if __name__ == "__main__":
    create_subject_list(Path("./PD_selected_paths.txt"), "./PD_selected_subjects.txt")
    create_subject_list(Path("./HC_selected_paths.txt"), "./HC_selected_subjects.txt")
    # subfile = Path().cwd() / "sub_list_test.txt"
    # test_path = Path().cwd() / "pipline" / "hc" / "outputs" / "ants" / "anat-12dofs"
    # paths = get_matrices_paths(test_path, subfile, pattern="_ses-BL0GenericAffine")
    # m, e = get_matrices(paths)
    # print(e)
