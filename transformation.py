from pathlib import Path
IEEE = 'ieee'
MCA = 'mca'
PATTERN = "*.mat"

def get_matrices_paths(parent_dir, pattern=PATTERN, n_mca=10):
    ieee_dir = parent_dir / IEEE
    ieee_paths = list(ieee_dir.glob(pattern))
    subjects = [l.name.removesuffix('.mat') for l in ieee_paths]
    
    paths = {}
    for sub,sub_path in zip(subjects, ieee_paths):
        paths[sub] = {IEEE:str(sub_path)}
        paths[sub][MCA] = []
 
    mca_dir = parent_dir / MCA
    for i in range(n_mca):
        this_itr_dir = mca_dir / str(i)
        mca_paths = list(this_itr_dir.glob(pattern))
        for sub_path in mca_paths:
            sub = sub_path.name.removesuffix('.mat')
            if sub in paths:
                paths[sub][MCA].append(str(sub_path))

    return paths




if __name__ == '__main__':
    dir = Path('./outputs/anat-12dofs')
    get_matrices_paths(dir)