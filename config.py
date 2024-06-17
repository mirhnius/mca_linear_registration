from pathlib import Path
from lnrgst_mca.load_utils import get_paths, get_matrices


configurations = {
    "FSL": {
        "mni152": {
            "failed_subjects_HC": ["sub-116230", "sub-4079", "sub-3620"],
            "failed_subjects_PD": ["sub-3709", "sub-3700", "sub-3403"],
            "PD_list": Path("./PD_selected_subjects.txt"),
            "HC_list": Path("./HC_selected_subjects.txt"),
            "path_PD": Path("./pipline/pd/outputs/anat-12dofs"),
            "path_HC": Path("./pipline/hc/outputs/anat-12dofs"),
            "pattern_PD": "_ses-BL",
            "pattern_HC": "_ses-BL",
        }
    },
    "ANTS": {
        "mni152": {
            "failed_subjects_HC": ["sub-116230", "sub-3620"],
            "failed_subjects_PD": [],
            "PD_list": Path("./PD_selected_subjects.txt"),
            "HC_list": Path("./HC_selected_subjects.txt"),
            "path_PD": Path("./pipline/pd/outputs/ants/anat-12dofs"),
            "path_HC": Path("./pipline/hc/outputs/ants/anat-12dofs"),
            "pattern_PD": "_ses-BL0GenericAffine",
            "pattern_HC": "_ses-BL0GenericAffine",
        }
    },
    "SPM": {
        "mni152": {
            "failed_subjects_HC": [],
            "failed_subjects_PD": [],
            "PD_list": Path("./PD_selected_subjects.txt"),
            "HC_list": Path("./HC_selected_subjects.txt"),
            "path_PD": Path("./pipline/pd/outputs/spm/anat-12dofs"),
            "path_HC": Path("./pipline/hc/outputs/spm/anat-12dofs"),
            "pattern_PD": "_ses-BL",
            "pattern_HC": "_ses-BL",
        }
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
