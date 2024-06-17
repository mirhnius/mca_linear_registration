import pathlib

BASELINE_SESSION = "ses-BL"
ANATOMICAL = "anat"
ACQUISITION = "acq-sag3D"
MODALITY = "T1w"
RUN = "run-01"
SUFFIX = ".nii.gz"
MOSUF = MODALITY + SUFFIX
ORIGINAL = "ieee"
MCA = "mca"
FLIRT = "flirt"
ANTS = "ants"
SPM = "spm"
MAT = ".mat"
NII = ".nii"

PATTERN = pathlib.Path("") / "sub-*" / BASELINE_SESSION / ANATOMICAL / f"sub-*_{BASELINE_SESSION}_{ACQUISITION}_{RUN}_{MOSUF}"
REF = pathlib.Path.cwd().parent.parent / "tpl-MNI152NLin2009cAsym_res-01_T1w_neck_5.nii.gz"
# REF = pathlib.Path.cwd().parent.parent / "tpl-MNI152NLin2009cAsym_res-01_T1w_neck_5.nii"
