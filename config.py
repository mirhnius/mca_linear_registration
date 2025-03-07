from pathlib import Path
from lnrgst_mca.load_utils import get_paths, get_matrices
from lnrgst_mca.constants import ANTS, FLIRT, SPM

MNI2009cAsym = "MNI152NLin2009cAsym_res-01"
MNI2009cSym = "MNI152NLin2009cSym_res-1"
short_template_names = {MNI2009cAsym: "Asym", MNI2009cSym: "Sym"}

FD_SD_x_lim = {
    FLIRT: {MNI2009cAsym: None, MNI2009cSym: None},
    ANTS: {MNI2009cAsym: None, MNI2009cSym: None},
    SPM: {MNI2009cAsym: (0, 8e-7), MNI2009cSym: None},
}

MAD_bin_sizes = {
    FLIRT: {MNI2009cAsym: [5, 5], MNI2009cSym: [5, 5]},
    ANTS: {MNI2009cAsym: [15, 10], MNI2009cSym: [15, 10]},
    SPM: {MNI2009cAsym: [15, 10], MNI2009cSym: [15, 10]},
}

FD_sd_bin_sizes = {
    FLIRT: {MNI2009cAsym: [10, 15], MNI2009cSym: [10, 15]},
    ANTS: {MNI2009cAsym: [10, 1], MNI2009cSym: [10, 1]},
    SPM: {MNI2009cAsym: [15, 10], MNI2009cSym: [15, 10]},
}

FD_mean_bin_sizes = {
    FLIRT: {MNI2009cAsym: [15, 10], MNI2009cSym: [10, 10]},
    ANTS: {MNI2009cAsym: [15, 1], MNI2009cSym: [10, 1]},
    SPM: {MNI2009cAsym: [15, 15], MNI2009cSym: [15, 15]},
}

palette_colors = {
    SPM: {MNI2009cAsym: "#dda0dd", MNI2009cSym: "#800080"},
    ANTS: {MNI2009cAsym: "#cd5c5c", MNI2009cSym: "#8b0000"},
    FLIRT: {MNI2009cAsym: "#90ee90", MNI2009cSym: "#008000"},
}
failed_palette_colors = {
    SPM: {
        MNI2009cAsym: "#d0dda0",  # Complementary to light purple (yellowish-green)
        MNI2009cSym: "#808000",  # Complementary to dark purple (olive green)
    },
    ANTS: {
        MNI2009cAsym: "#5ccdc5",  # Complementary to red (cyan)
        MNI2009cSym: "#00008b",  # Complementary to dark red (navy blue)
    },
    FLIRT: {
        MNI2009cAsym: "#ee9090",  # Complementary to light green (light red)
        MNI2009cSym: "#800000",  # Complementary to dark green (maroon)
    },
}
palette_colors_similarity_measures = {
    FLIRT: {MNI2009cAsym: {"mutualinfo":"#9ACD32", 
                           "normmi":"#90ee0d", 
                           "normcorr":"#90eeb3", 
                           "corratio":"#90ee90"
                        }
            }
    }

failed_palette_colors_similarity_measures = {
    FLIRT: {MNI2009cAsym: {"mutualinfo":"#6532cd", 
                           "normmi":"#6f11f2", 
                           "normcorr":"#6f114c", 
                           "corratio":"#6f116f"
                        }
          }
    }
cost_function_names = {"mutualinfo":"MI","normmi":"NMI","normcorr":"NCC","corratio":"CR"}

PD_list_path = Path("./PD_selected_subjects.txt")
HC_list_path = Path("./HC_selected_subjects.txt")
pipeline_path = Path("./pipline")
PD_path = pipeline_path / "pd" / "outputs"
HC_path = pipeline_path / "hc" / "outputs"
cost_exp_path = Path("/home/niusham/projects/rrg-glatard/niusham/mca_linear_registration/metrics_exp")
verrou_path = Path("/home/niusham/projects/rrg-glatard/niusham/mca_linear_registration/verrou")
PD_path_cost = cost_exp_path / "pd" / "output"
HC_path_cost = cost_exp_path / "hc" / "output"
verrou_PD_path = verrou_path / "pd" / "output"
verrou_HC_path = verrou_path / "hc" / "output"
ddof = "anat-12dofs"

# the pattern is for affine only change. add the image too
# make it more efficent
configurations = {
    FLIRT: {
        MNI2009cAsym: {
            "failed_subjects_HC": [
                "sub-116230",
                "sub-3620",
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
            "failed_subjects_HC": ["sub-3620", "sub-116230"],
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
                "sub-3104",
                "sub-116230",
                "sub-3620",
                "sub-3316",
                "sub-3361",
                "sub-3570",
                "sub-3615",
                "sub-3767",
                "sub-3811",
                "sub-3853",
                "sub-3969",
                "sub-4067",
                "sub-4079",
            ],
            "failed_subjects_PD": [
                "sub-3365",
                "sub-3403",
                "sub-3586",
                "sub-3700",
                "sub-3709",
                "sub-3823",
                "sub-3960",
                "sub-3970",
                "sub-40733",
                "sub-42264",
                "sub-75562",
                "sub-106703",
                "sub-139982",
            ],
            "PD_list": PD_list_path,
            "HC_list": HC_list_path,
            "path_PD": PD_path / SPM / MNI2009cAsym / ddof,
            "path_HC": HC_path / SPM / MNI2009cAsym / ddof,
            "pattern_PD": "_ses-BL",
            "pattern_HC": "_ses-BL",
        },
        MNI2009cSym: {
            "failed_subjects_HC": ["sub-3057", "sub-3615", "sub-3620", "sub-3853", "sub-4079", "sub-116230", "sub-4067"],
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

configurations_cost = {
    # "leastsq": {
    #     FLIRT: {
    #         "MNI2009cAsym: {
    #             "failed_subjects_HC": [
    #                 "sub-116230",
    #                 "sub-3620",
    #             ],
    #             "failed_subjects_PD": ["sub-3709", "sub-3700", "sub-3403"],
    #             "PD_list": PD_list_path,
    #             "HC_list": HC_list_path,
    #             "path_PD": PD_path / "leastsq" / FLIRT / MNI2009cAsym / ddof,
    #             "path_HC": HC_path / "leastsq" / FLIRT / MNI2009cAsym / ddof,
    #             "pattern_PD": "_ses-BL",
    #             "pattern_HC": "_ses-BL",
    #         }
    #     }
    # },
    "mutualinfo": {
        FLIRT: {
            MNI2009cAsym: {
                "failed_subjects_HC": ["sub-116230", "sub-3620", "sub-3853"],
                "failed_subjects_PD": ["sub-3709", "sub-3700", "sub-3403", "sub-40733"],
                "PD_list": PD_list_path,
                "HC_list": HC_list_path,
                "path_PD": PD_path_cost / "mutualinfo" / FLIRT / MNI2009cAsym / ddof,
                "path_HC": HC_path_cost / "mutualinfo" / FLIRT / MNI2009cAsym / ddof,
                "pattern_PD": "_ses-BL",
                "pattern_HC": "_ses-BL",
            }
        }
    },
    "normcorr": {
        FLIRT: {
            MNI2009cAsym: {
                "failed_subjects_HC": [
                    "sub-116230",
                    "sub-3620",
                ],
                "failed_subjects_PD": ["sub-3709", "sub-3700", "sub-3403", "sub-40733"],
                "PD_list": PD_list_path,
                "HC_list": HC_list_path,
                "path_PD": PD_path_cost / "normcorr" / FLIRT / MNI2009cAsym / ddof,
                "path_HC": HC_path_cost / "normcorr" / FLIRT / MNI2009cAsym / ddof,
                "pattern_PD": "_ses-BL",
                "pattern_HC": "_ses-BL",
            }
        }
    },
    "normmi": {
        FLIRT: {
            MNI2009cAsym: {
                "failed_subjects_HC": [
                    "sub-116230",
                    "sub-3620",
                ],
                "failed_subjects_PD": ["sub-3709", "sub-3700", "sub-40733"],
                "PD_list": PD_list_path,
                "HC_list": HC_list_path,
                "path_PD": PD_path_cost / "normmi" / FLIRT / MNI2009cAsym / ddof,
                "path_HC": HC_path_cost / "normmi" / FLIRT / MNI2009cAsym / ddof,
                "pattern_PD": "_ses-BL",
                "pattern_HC": "_ses-BL",
            }
        }
    },
    "corratio": {
        FLIRT: {
            MNI2009cAsym: {
                "failed_subjects_HC": [
                    "sub-116230",
                    "sub-3620",
                ],
                "failed_subjects_PD": ["sub-3709", "sub-3700", "sub-3403"],
                "PD_list": PD_list_path,
                "HC_list": HC_list_path,
                "path_PD": PD_path_cost / "normcorr" / FLIRT / MNI2009cAsym / ddof,
                "path_HC": HC_path_cost / "normcorr" / FLIRT / MNI2009cAsym / ddof,
                "pattern_PD": "_ses-BL",
                "pattern_HC": "_ses-BL",
            }
        },
    },
}

verrou_configuration = {
    SPM: {
        MNI2009cAsym: {
            "failed_subjects_HC": [
                "sub-3104",
                "sub-116230",
                "sub-3620",
                "sub-3316",
                "sub-3361",
                "sub-3570",
                "sub-3615",
                "sub-3767",
                "sub-3811",
                "sub-3853",
                "sub-3969",
                "sub-4067",
                "sub-4079",
            ],
            "failed_subjects_PD": [
                "sub-3365",
                "sub-3403",
                "sub-3586",
                "sub-3700",
                "sub-3709",
                "sub-3823",
                "sub-3960",
                "sub-3970",
                "sub-40733",
                "sub-42264",
                "sub-75562",
                "sub-106703",
                "sub-139982",
            ],
            "PD_list": PD_list_path,
            "HC_list": HC_list_path,
            "path_PD": verrou_PD_path / SPM / MNI2009cAsym / ddof,
            "path_HC": verrou_HC_path / SPM / MNI2009cAsym / ddof,
            "pattern_PD": "_ses-BL",
            "pattern_HC": "_ses-BL",
        }
    }
}


def get_configurations(software, template, cost_function=False, verrou=False):

    if verrou:
        config = verrou_configuration[software][template]
    elif cost_function:
        config = configurations_cost[cost_function][software][template]
    else:
        config = configurations[software][template]

    paths_PD = get_paths(config["path_PD"], config["PD_list"], pattern=config["pattern_PD"])
    paths_HC = get_paths(config["path_HC"], config["HC_list"], pattern=config["pattern_HC"])

    mat_dic_PD, error_PD = get_matrices(paths_PD)
    mat_dic_HC, error_HC = get_matrices(paths_HC)

    return config["failed_subjects_HC"], config["failed_subjects_PD"], mat_dic_PD, error_PD, mat_dic_HC, error_HC


if __name__ == "__main__":
    print(get_configurations("FSL", "mni152"))
