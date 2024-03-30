import numpy as np
from pathlib import Path

IEEE = "ieee"
MCA = "mca"
PATTERN = "*.mat"


def get_matrices_paths(parent_dir, pattern=PATTERN, n_mca=10):

    if isinstance(parent_dir, str):
        parent_dir = Path(parent_dir)
    ieee_dir = parent_dir / IEEE
    ieee_paths = list(ieee_dir.glob(pattern))
    subjects = [path.name.removesuffix(".mat") for path in ieee_paths]

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


def get_matrices(paths):

    matrices = {}
    for sub in paths.keys():

        mat_list = []
        for p in paths[sub][MCA]:
            mat_list.append(np.loadtxt(p))
        arr = np.array(mat_list)
        matrices[sub] = {MCA: arr}

        p_ieee = paths[sub][IEEE]
        matrices[sub][IEEE] = np.loadtxt(p_ieee)

    return matrices


if __name__ == "__main__":
    dir = Path("./outputs/anat-12dofs")
    paths = get_matrices_paths(dir)
    # n_mca = 10
    n_subjects = len(paths)
