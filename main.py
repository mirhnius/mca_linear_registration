# import metrics_utils
import load_utils
from plot_utils import plotter
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


def basic_info_plotter(group1, group2, software, figure_size=(9, 4), y_lim_mean=None, y_lim_sd=None, **kwargs):

    plt.figure(figsize=figure_size)
    plotter(np.mean(group1, axis=1), np.mean(group2, axis=1), f"Scales Mean {software}", y_lim_mean, kwargs)

    plt.figure(figsize=figure_size)
    plotter(np.std(group1, axis=1), np.std(group2, axis=1), f"Scales SD {software}", y_lim_sd, kwargs)


def copy_and_remove_keys(original_dict, keys_to_copy):

    dict_copy = deepcopy(original_dict)
    new_dict = {}
    for key in keys_to_copy:
        new_dict[key] = dict_copy[key]
        dict_copy.pop(key)

    return new_dict, dict_copy


if __name__ == "__main__":

    software = "FSL"
    failed_subjects_HC = []
    failed_subjects_PD = []

    path_PD = Path("./pipline/pd/outputs/anat-12dofs")
    path_HC = Path("./pipline/hc/outputs/anat-12dofs")

    paths_PD = load_utils.get_paths(path_PD, Path("./PD_selected_subjects.txt"), pattern="_ses-BL")
    paths_HC = load_utils.get_paths(path_HC, Path("./HC_selected_subjects.txt"), pattern="_ses-BL")

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

    angles_mca_PD = np.degree(angles_mca_PD)
    angles_mca_HC = np.degree(angles_mca_HC)

    basic_info_plotter(translation_mca_fine_PD, translation_mca_fine_HC)
    basic_info_plotter(angles_mca_fine_PD, angles_mca_fine_HC)
    basic_info_plotter(scales_mca_fine_PD, scales_mca_fine_HC)
    basic_info_plotter(shears_mca_fine_PD, shears_mca_fine_HC)

    basic_info_plotter(translation_mca_failed_PD, translation_mca_failed_HC)
    basic_info_plotter(angles_mca_failed_PD, angles_mca_failed_HC)
    basic_info_plotter(scales_mca_failed_PD, scales_mca_failed_HC)
    basic_info_plotter(shears_mca_failed_PD, shears_mca_failed_HC)

    # result_HC = np.zeros((n_hc_fsl,10))
    # for i in range(n_hc_fsl):
    #     for j in range(10):
    #         result_HC[i,j] = metrics_utils.framewise_displacement(translations_mca_HC[i,j],
    # angles_mca_HC[i,j], translations_ieee_HC[i], np.degrees(angles_ieee_HC[i]))
