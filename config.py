from pathlib import Path
from lnrgst_mca.load_utils import get_paths, get_matrices
from lnrgst_mca.constants import ANTS, FLIRT, SPM

MNI2009cAsym = "MNI152NLin2009cAsym_res-01"
MNI2009cSym = "MNI152NLin2009cSym_res-1"

MAD_bin_sizes = {
    FLIRT: {MNI2009cAsym: [5, 5], MNI2009cSym: [5, 5]},
    ANTS: {MNI2009cAsym: [15, 10], MNI2009cSym: [15, 10]},
    SPM: {MNI2009cAsym: [15, 10], MNI2009cSym: [15, 10]},
}

FD_sd_bin_sizes = {
    FLIRT: {MNI2009cAsym: [10, 10], MNI2009cSym: [10, 10]},
    ANTS: {MNI2009cAsym: [10, 1], MNI2009cSym: [10, 1]},
    SPM: {MNI2009cAsym: [50, 5], MNI2009cSym: [50, 5]},
}

FD_mean_bin_sizes = {
    FLIRT: {MNI2009cAsym: [10, 10], MNI2009cSym: [10, 10]},
    ANTS: {MNI2009cAsym: [10, 1], MNI2009cSym: [10, 1]},
    SPM: {MNI2009cAsym: [15, 10], MNI2009cSym: [15, 10]},
}

PD_list_path = Path("./PD_selected_subjects.txt")
HC_list_path = Path("./HC_selected_subjects.txt")
pipeline_path = Path("./pipline")
PD_path = pipeline_path / "pd" / "outputs"
HC_path = pipeline_path / "hc" / "outputs"
ddof = "anat-12dofs"

# the pattern is for affine only change. add the image too
# make it more efficent
configurations = {
    FLIRT: {
        MNI2009cAsym: {
            "failed_subjects_HC": [
                "sub-116230",
                "sub-4079",
                "sub-3620",
                "sub-3414",
            ],
            "failed_subjects_PD": ["sub-3709", "sub-3700", "sub-3403"],
            "PD_list": PD_list_path,
            "HC_list": HC_list_path,
            "path_PD": PD_path / FLIRT / MNI2009cAsym / ddof,
            "path_HC": HC_path / FLIRT / MNI2009cAsym / ddof,
            "pattern_PD": "_ses-BL",
            "pattern_HC": "_ses-BL",
        },
        MNI2009cSym: {
            "failed_subjects_HC": ["sub-3414", "sub-3620", "sub-116230", "sub-4079"],
            "failed_subjects_PD": [
                "sub-3403",
                "sub-3700",
                "sub-3709",
                "sub-40733",
            ],  # "sub-3777",
            "PD_list": PD_list_path,
            "HC_list": HC_list_path,
            "path_PD": PD_path / FLIRT / MNI2009cSym / ddof,
            "path_HC": HC_path / FLIRT / MNI2009cSym / ddof,
            "pattern_PD": "_ses-BL",
            "pattern_HC": "_ses-BL",
        },
    },
    ANTS: {
        MNI2009cAsym: {
            "failed_subjects_HC": ["sub-116230", "sub-3620"],
            "failed_subjects_PD": [],
            "PD_list": PD_list_path,
            "HC_list": HC_list_path,
            "path_PD": PD_path / ANTS / MNI2009cAsym / ddof,
            "path_HC": HC_path / ANTS / MNI2009cAsym / ddof,
            "pattern_PD": "_ses-BL0GenericAffine",
            "pattern_HC": "_ses-BL0GenericAffine",
        },
        MNI2009cSym: {
            "failed_subjects_HC": ["sub-116230", "sub-3969"],  # , "sub-3620"
            "failed_subjects_PD": [],
            "PD_list": PD_list_path,
            "HC_list": HC_list_path,
            "path_PD": PD_path / ANTS / MNI2009cSym / ddof,
            "path_HC": HC_path / ANTS / MNI2009cSym / ddof,
            "pattern_PD": "_ses-BL0GenericAffine",
            "pattern_HC": "_ses-BL0GenericAffine",
        },
    },
    SPM: {
        MNI2009cAsym: {
            "failed_subjects_HC": [
                "sub-116230",
                "sub-3620",
                "sub-3316",
                "sub-3361",
                "sub-3414",
                "sub-3570",
                "sub-3615",
                "sub-3811",
                "sub-3853",
                "sub-3970" "sub-3969",
                "sub-4079",
            ],
            "failed_subjects_PD": [
                "sub-3403",
                "sub-3586",
                "sub-3700",
                "sub-3709",
                "sub-3823",
                "sub-3960",
                "sub-3970",
                "sub-4121",
                "sub-40733",
                "sub-42264",
                "sub-75562",
            ],
            "PD_list": PD_list_path,
            "HC_list": HC_list_path,
            "path_PD": PD_path / SPM / MNI2009cAsym / ddof,
            "path_HC": HC_path / SPM / MNI2009cAsym / ddof,
            "pattern_PD": "_ses-BL",
            "pattern_HC": "_ses-BL",
        },
        MNI2009cSym: {
            "failed_subjects_HC": ["sub-3057", "sub-3104", "sub-3414", "sub-3615", "sub-3620", "sub-3853", "sub-4079", "sub-116230"],
            "failed_subjects_PD": [
                "sub-3365",
                "sub-3403",
                "sub-3700",
                "sub-3709",
                "sub-3960",
                "sub-40733",
                "sub-42264",
                "sub-75562",
            ],
            "PD_list": PD_list_path,
            "HC_list": HC_list_path,
            "path_PD": PD_path / SPM / MNI2009cSym / ddof,
            "path_HC": HC_path / SPM / MNI2009cSym / ddof,
            "pattern_PD": "_ses-BL",
            "pattern_HC": "_ses-BL",
        },
    },
}


def get_configurations(software, template):

    config = configurations[software][template]

    paths_PD = get_paths(config["path_PD"], config["PD_list"], pattern=config["pattern_PD"])
    paths_HC = get_paths(config["path_HC"], config["HC_list"], pattern=config["pattern_HC"])

    mat_dic_PD, error_PD = get_matrices(paths_PD)
    mat_dic_HC, error_HC = get_matrices(paths_HC)

    return config["failed_subjects_HC"], config["failed_subjects_PD"], mat_dic_PD, error_PD, mat_dic_HC, error_HC


if __name__ == "__main__":
    print(get_configurations("FSL", "mni152"))
