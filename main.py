# import metrics_utils
import load_utils
from plot_utils import plotter
import metrics_utils
from copy import deepcopy
import numpy as np

# import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from fsl.transform import affine


# changning it later in a way to be abe to use it for ieee
def transformation_dictionary_to_arrays(mat_dic, n_mca=10):

    # Initialize numpy arrays to store transformation parameters for all subjects
    num_subjects = len(mat_dic)
    scales_mca = np.zeros((num_subjects, n_mca, 3))
    translations_mca = np.zeros((num_subjects, n_mca, 3))
    angles_mca = np.zeros((num_subjects, n_mca, 3))
    shears_mca = np.zeros((num_subjects, n_mca, 3))

    # Populate the numpy arrays
    for sub_idx, (sub, matrices) in enumerate(mat_dic.items()):
        for i, matrix in enumerate(matrices["mca"]):
            scales, translations, angles, shears = affine.decompose(matrix, shears=True, angles=True)
            scales_mca[sub_idx, i, :] = scales
            translations_mca[sub_idx, i, :] = translations
            angles_mca[sub_idx, i, :] = np.array(angles)
            shears_mca[sub_idx, i, :] = shears

    return scales_mca, translations_mca, angles_mca, shears_mca


def basic_info_plotter(group1, group2, software, variable, figure_size=(9, 4), y_lim_mean=None, y_lim_sd=None, path=None, **kwargs):

    plt.figure(figsize=figure_size)
    plotter(np.mean(group1, axis=1), np.mean(group2, axis=1), title=f"{variable} Mean {software}", ylim=y_lim_mean, path=path, **kwargs)

    plt.figure(figsize=figure_size)
    plotter(np.std(group1, axis=1), np.std(group2, axis=1), title=f"{variable} SD {software}", ylim=y_lim_sd, path=path, **kwargs)


def copy_and_remove_keys(original_dict, keys_to_copy):

    dict_copy = deepcopy(original_dict)
    new_dict = {}
    for key in keys_to_copy:
        new_dict[key] = dict_copy[key]
        dict_copy.pop(key)

    return dict_copy, new_dict


if __name__ == "__main__":

    # software = "FSL"
    # failed_subjects_HC = ["sub-116230", "sub-4079", "sub-3620"]
    # failed_subjects_PD = ["sub-3709", "sub-3700", "sub-3403"]
    # path = Path().cwd() / "outputs_plots" / "diagrams" / software

    # path_PD = Path("./pipline/pd/outputs/anat-12dofs")
    # path_HC = Path("./pipline/hc/outputs/anat-12dofs")

    # paths_PD = load_utils.get_paths(path_PD, Path("./PD_selected_subjects.txt"), pattern="_ses-BL")
    # paths_HC = load_utils.get_paths(path_HC, Path("./HC_selected_subjects.txt"), pattern="_ses-BL")

    software = "ANTS"
    failed_subjects_HC = ["sub-116230", "sub-3620"]
    failed_subjects_PD = []
    path = Path().cwd() / "outputs_plots" / "diagrams" / software

    path_PD = Path("./pipline/pd/outputs/ants/anat-12dofs")
    path_HC = Path("./pipline/hc/outputs/ants/anat-12dofs")

    paths_PD = load_utils.get_paths(path_PD, Path("./PD_selected_subjects.txt"), pattern="_ses-BL0GenericAffine")
    paths_HC = load_utils.get_paths(path_HC, Path("./HC_selected_subjects.txt"), pattern="_ses-BL0GenericAffine")

    mat_dic_PD, error_PD = load_utils.get_matrices(paths_PD)
    mat_dic_HC, error_HC = load_utils.get_matrices(paths_HC)

    if error_PD or error_HC:
        print(error_PD, error_HC)
        raise Exception("There is an issue with mat files")

    mat_dic_fine_PD, mat_dic_failed_PD = copy_and_remove_keys(mat_dic_PD, failed_subjects_PD)
    mat_dic_fine_HC, mat_dic_failed_HC = copy_and_remove_keys(mat_dic_HC, failed_subjects_HC)

    scales_mca_PD, translation_mca_PD, angles_mca_PD, shears_mca_PD = transformation_dictionary_to_arrays(mat_dic_PD)
    scales_mca_HC, translation_mca_HC, angles_mca_HC, shears_mca_HC = transformation_dictionary_to_arrays(mat_dic_HC)

    scales_mca_fine_PD, translation_mca_fine_PD, angles_mca_fine_PD, shears_mca_fine_PD = transformation_dictionary_to_arrays(mat_dic_fine_PD)
    scales_mca_fine_HC, translation_mca_fine_HC, angles_mca_fine_HC, shears_mca_fine_HC = transformation_dictionary_to_arrays(mat_dic_fine_HC)

    scales_mca_failed_PD, translation_mca_failed_PD, angles_mca_failed_PD, shears_mca_failed_PD = transformation_dictionary_to_arrays(
        mat_dic_failed_PD
    )
    scales_mca_failed_HC, translation_mca_failed_HC, angles_mca_failed_HC, shears_mca_failed_HC = transformation_dictionary_to_arrays(
        mat_dic_failed_HC
    )

    n_fine_HC = len(mat_dic_fine_HC)
    n_fine_PD = len(mat_dic_fine_PD)

    n_failed_HC = len(mat_dic_fine_HC)
    n_failed_PD = len(mat_dic_fine_PD)

    angles_mca_PD = np.degrees(angles_mca_PD)
    angles_mca_HC = np.degrees(angles_mca_HC)

    angles_mca_fine_PD = np.degrees(angles_mca_fine_PD)
    angles_mca_fine_HC = np.degrees(angles_mca_fine_HC)

    angles_mca_failed_PD = np.degrees(angles_mca_failed_PD)
    angles_mca_failed_HC = np.degrees(angles_mca_failed_HC)

    basic_info_plotter(
        translation_mca_fine_PD,
        translation_mca_fine_HC,
        software=software,
        variable="Translations",
        path=path,
        axis_labels=["x", "y", "z"],
        y_lim_sd=[(0, 0.08), (0, 0.2), (0, 0.06)],
    )
    basic_info_plotter(
        angles_mca_fine_PD,
        angles_mca_fine_HC,
        software=software,
        variable="Angles",
        path=path,
        axis_labels=["x", "y", "z"],
        y_lim_sd=[(0, 4), (0, 0.2), (0, 0.2)],
    )
    basic_info_plotter(
        scales_mca_fine_PD,
        scales_mca_fine_HC,
        software=software,
        variable="Scales",
        path=path,
        axis_labels=["x", "y", "z"],
        y_lim_sd=[(0, 0.01), (0, 0.1), (0, 1)],
    )
    basic_info_plotter(
        shears_mca_fine_PD,
        shears_mca_fine_HC,
        software=software,
        variable="Shears",
        path=path,
        axis_labels=["x", "y", "z"],
        y_lim_sd=[(0, 0.05), (0, 0.1), (0, 1)],
    )

    basic_info_plotter(
        translation_mca_failed_PD,
        translation_mca_failed_HC,
        software=software,
        variable="Translations failed",
        path=path,
        axis_labels=["x", "y", "z"],
    )
    basic_info_plotter(
        angles_mca_failed_PD, angles_mca_failed_HC, software=software, variable="Angles failed", path=path, axis_labels=["x", "y", "z"]
    )
    basic_info_plotter(
        scales_mca_failed_PD, scales_mca_failed_HC, software=software, variable="Scales failed", path=path, axis_labels=["x", "y", "z"]
    )
    basic_info_plotter(
        shears_mca_failed_PD, shears_mca_failed_HC, software=software, variable="Shears failed", path=path, axis_labels=["x", "y", "z"]
    )

    result_fine_PD = metrics_utils.FD_all_subjects(translation_mca_fine_PD, angles_mca_fine_PD)
    result_fine_HC = metrics_utils.FD_all_subjects(translation_mca_fine_HC, angles_mca_fine_HC)

    basic_info_plotter(result_fine_PD, result_fine_HC, software=software, variable="Framewise Displacement", path=path, figure_size=(4, 4))
    #    y_lim_sd=[(0,4), (0,0.2), (0,0.2)])

    result_failed_PD = metrics_utils.FD_all_subjects(translation_mca_failed_PD, angles_mca_failed_PD)
    result_failed_HC = metrics_utils.FD_all_subjects(translation_mca_failed_HC, angles_mca_failed_HC)

    basic_info_plotter(result_failed_PD, result_failed_HC, software=software, variable="Framewise Displacement failed", path=path, figure_size=(4, 4))

    result_all_PD = metrics_utils.FD_all_subjects(translation_mca_PD, angles_mca_PD)
    result_all_HC = metrics_utils.FD_all_subjects(translation_mca_HC, angles_mca_HC)

    basic_info_plotter(result_all_PD, result_all_HC, software=software, variable="Framewise Displacement All", path=path, figure_size=(4, 4))

    # leargest_indces(np.std(result_fine_PD, axis=1), mat_dic_fine_PD, n=10)
    # leargest_indces(np.std(result_fine_HC, axis=1), mat_dic_fine_HC, n=10)
