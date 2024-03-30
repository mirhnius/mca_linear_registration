import numpy as np
import scipy
from pathlib import Path

IEEE = "ieee"
MCA = "mca"
PATTERN = "*.mat"


def get_matrices_paths(parent_dir, pattern=PATTERN, n_mca=10):

    if isinstance(parent_dir, str):
        parent_dir = Path(parent_dir)
    ieee_dir = parent_dir / IEEE
    ieee_paths = list(ieee_dir.glob(pattern))
    subjects = [l.name.removesuffix(".mat") for l in ieee_paths]

    paths = {}
    for sub, sub_path in zip(subjects, ieee_paths):
        paths[sub] = {IEEE: str(sub_path)}
        paths[sub][MCA] = []

    mca_dir = parent_dir / MCA
    for i in range(1, n_mca + 1):
        this_itr_dir = mca_dir / str(i)
        mca_paths = list(this_itr_dir.glob(pattern))
        for sub_path in mca_paths:
            sub = sub_path.name.removesuffix(".mat")
            if sub in paths:
                paths[sub][MCA].append(str(sub_path))

    return paths


def is_matlab_file(filename):

    try:
        scipy.io.whosmat(filename)
        return True
    except ValueError:
        return False


# def loader(filename):

#     if is_matalb_file(filename):#maybe add ".mat"
#         scipy.io  cmplete this late to avoid duplication


def get_matrices(paths):

    matrices = {}
    for sub in paths.keys():

        mat_list = []
        for p in paths[sub][MCA]:
            if is_matlab_file(p):
                mat = scipy.io.loadmat(p)
                mat = next(iter(mat.values()))
                mat_list.append(mat.reshape((-1, 4)))
            else:
                mat_list.append(np.loadtxt(p))
        arr = np.array(mat_list)
        matrices[sub] = {MCA: arr}

        p_ieee = paths[sub][IEEE]
        if is_matlab_file(p_ieee):
            mat = scipy.io.loadmat(p_ieee)
            mat = next(iter(mat.values()))
            matrices[sub][IEEE] = mat.reshape((-1, 4))
        else:
            matrices[sub][IEEE] = np.loadtxt(p_ieee)

    return matrices


if __name__ == "__main__":
    dir = Path("./outputs/anat-12dofs")
    paths = get_matrices_paths(dir)
    # n_mca = 10
    n_subjects = len(paths)
