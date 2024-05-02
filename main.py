import load_utils
from plot_utils import plotter
import metrics_utils
from copy import deepcopy
import numpy as np

# import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from fsl.transform import affine


def largest_indces(array, dict_, n=4):
    largest_idx = np.argsort(array, axis=0)[-n:]
    return np.array(list(dict_.keys()))[largest_idx], largest_idx


# changning it later in a way to be abe to use it for ieee
def transformation_dictionary_to_arrays(mat_dic, mode="mca", n_mca=10):

    # Initialize numpy arrays to store transformation parameters for all subjects
    num_subjects = len(mat_dic)
    scales_mca = np.zeros((num_subjects, n_mca, 3))
    translations_mca = np.zeros((num_subjects, n_mca, 3))
    angles_mca = np.zeros((num_subjects, n_mca, 3))
    shears_mca = np.zeros((num_subjects, n_mca, 3))

    # Populate the numpy arrays
    for sub_idx, (_, matrices) in enumerate(mat_dic.items()):

        if n_mca == 1 and mode == "ieee":
            scales, translations, angles, shears = affine.decompose(matrices[mode], shears=True, angles=True)
            scales_mca[sub_idx, 0, :] = scales
            translations_mca[sub_idx, 0, :] = translations
            angles_mca[sub_idx, 0, :] = np.array(angles)
            shears_mca[sub_idx, 0, :] = shears
            continue

        for i, matrix in enumerate(matrices[mode]):
            scales, translations, angles, shears = affine.decompose(matrix, shears=True, angles=True)
            scales_mca[sub_idx, i, :] = scales
            translations_mca[sub_idx, i, :] = translations
            angles_mca[sub_idx, i, :] = np.array(angles)
            shears_mca[sub_idx, i, :] = shears

    return scales_mca.squeeze(), translations_mca.squeeze(), angles_mca.squeeze(), shears_mca.squeeze()


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


def concatenate_mca_matrices(mat_dic):
    # Get a list of the "mca" matrices in the dictionary
    mca_matrices = [matrices["mca"] for matrices in mat_dic.values()]

    # Stack the matrices along a new axis
    return np.stack(mca_matrices)


if __name__ == "__main__":

    software = "FSL"
    failed_subjects_HC = ["sub-116230", "sub-4079", "sub-3620", "sub-3369"]
    failed_subjects_PD = ["sub-3709", "sub-3700", "sub-3403"]
    path = Path().cwd() / "outputs_plots" / "diagrams" / software

    path_PD = Path("./pipline/pd/outputs/anat-12dofs")
    path_HC = Path("./pipline/hc/outputs/anat-12dofs")

    paths_PD = load_utils.get_paths(path_PD, Path("./PD_selected_subjects.txt"), pattern="_ses-BL")
    paths_HC = load_utils.get_paths(path_HC, Path("./HC_selected_subjects.txt"), pattern="_ses-BL")

    # software = "ANTS"
    # failed_subjects_HC = ["sub-116230", "sub-3620"]
    # failed_subjects_PD = []
    # path = Path().cwd() / "outputs_plots" / "diagrams" / software

    # path_PD = Path("./pipline/pd/outputs/ants/anat-12dofs")
    # path_HC = Path("./pipline/hc/outputs/ants/anat-12dofs")

    # paths_PD = load_utils.get_paths(path_PD, Path("./PD_selected_subjects.txt"), pattern="_ses-BL0GenericAffine")
    # paths_HC = load_utils.get_paths(path_HC, Path("./HC_selected_subjects.txt"), pattern="_ses-BL0GenericAffine")

    mat_dic_PD, error_PD = load_utils.get_matrices(paths_PD)
    mat_dic_HC, error_HC = load_utils.get_matrices(paths_HC)

    if error_PD or error_HC:
        print(error_PD, error_HC)
        raise Exception("There is an issue with mat files")

    mat_dic_fine_PD, mat_dic_failed_PD = copy_and_remove_keys(mat_dic_PD, failed_subjects_PD)
    mat_dic_fine_HC, mat_dic_failed_HC = copy_and_remove_keys(mat_dic_HC, failed_subjects_HC)

    scales_ieee_PD, translation_ieee_PD, angles_ieee_PD, shears_ieee_PD = transformation_dictionary_to_arrays(mat_dic_PD, n_mca=1, mode="ieee")
    scales_ieee_HC, translation_ieee_HC, angles_ieee_HC, shears_ieee_HC = transformation_dictionary_to_arrays(mat_dic_HC, n_mca=1, mode="ieee")

    scales_ieee_fine_PD, translation_ieee_fine_PD, angles_ieee_fine_PD, shears_ieee_fine_PD = transformation_dictionary_to_arrays(
        mat_dic_fine_PD, n_mca=1, mode="ieee"
    )
    scales_ieee_fine_HC, translation_ieee_fine_HC, angles_ieee_fine_HC, shears_ieee_fine_HC = transformation_dictionary_to_arrays(
        mat_dic_fine_HC, n_mca=1, mode="ieee"
    )

    scales_ieee_failed_PD, translation_ieee_failed_PD, angles_ieee_failed_PD, shears_ieee_failed_PD = transformation_dictionary_to_arrays(
        mat_dic_failed_PD, n_mca=1, mode="ieee"
    )
    scales_ieee_failed_HC, translation_ieee_failed_HC, angles_ieee_failed_HC, shears_ieee_failed_HC = transformation_dictionary_to_arrays(
        mat_dic_failed_HC, n_mca=1, mode="ieee"
    )

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

    angles_ieee_PD = np.degrees(angles_ieee_PD)
    angles_ieee_HC = np.degrees(angles_ieee_HC)

    angles_ieee_fine_PD = np.degrees(angles_ieee_fine_PD)
    angles_ieee_fine_HC = np.degrees(angles_ieee_fine_HC)

    angles_ieee_failed_PD = np.degrees(angles_ieee_failed_PD)
    angles_ieee_failed_HC = np.degrees(angles_ieee_failed_HC)

    basic_info_plotter(
        translation_mca_fine_PD,
        translation_mca_fine_HC,
        software=software,
        variable="Translations",
        path=path,
        axis_labels=["x", "y", "z"],
        y_lim_mean=[(-25, 25), (-50, 20), (-45, 45)],
        y_lim_sd=[(0, 0.08), (0, 0.2), (0, 0.06)],
        ylable="(mm)",
    )
    # y_lim_sd=[(0, 0.08), (0, 0.2), (0, 0.06)]

    basic_info_plotter(
        angles_mca_fine_PD,
        angles_mca_fine_HC,
        software=software,
        variable="Angles",
        path=path,
        axis_labels=["x", "y", "z"],
        y_lim_sd=[(0, 4), (0, 0.2), (0, 0.2)],
        ylable="(degree)",
    )
    # fsl y_lim_mean=[(-35, 5), (-9, 7), (-7, 5)],
    # y_lim_sd=[(0, 4), (0, 0.2), (0, 0.2)],

    basic_info_plotter(
        scales_mca_fine_PD,
        scales_mca_fine_HC,
        software=software,
        variable="Scales",
        path=path,
        axis_labels=["x", "y", "z"],
        y_lim_sd=[(0, 0.01), (0, 0.1), (0, 1)],
    )
    # y_lim_sd=[(0, 0.01), (0, 0.1), (0, 1)],

    basic_info_plotter(
        shears_mca_fine_PD,
        shears_mca_fine_HC,
        software=software,
        variable="Shears",
        path=path,
        axis_labels=["x", "y", "z"],
        y_lim_sd=[(0, 0.05), (0, 0.1), (0, 1)],
    )
    # y_lim_sd=[(0, 0.05), (0, 0.1), (0, 1)],
    basic_info_plotter(
        translation_mca_failed_PD,
        translation_mca_failed_HC,
        software=software,
        variable="Translations failed",
        path=path,
        axis_labels=["x", "y", "z"],
        ylable="(mm)",
    )
    basic_info_plotter(
        angles_mca_failed_PD,
        angles_mca_failed_HC,
        software=software,
        variable="Angles failed",
        path=path,
        axis_labels=["x", "y", "z"],
        ylable="(degree)",
    )
    basic_info_plotter(
        scales_mca_failed_PD, scales_mca_failed_HC, software=software, variable="Scales failed", path=path, axis_labels=["x", "y", "z"]
    )
    basic_info_plotter(
        shears_mca_failed_PD, shears_mca_failed_HC, software=software, variable="Shears failed", path=path, axis_labels=["x", "y", "z"]
    )

    result_fine_PD = metrics_utils.FD_all_subjects(translation_mca_fine_PD, angles_mca_fine_PD)
    result_fine_HC = metrics_utils.FD_all_subjects(translation_mca_fine_HC, angles_mca_fine_HC)

    result_failed_PD = metrics_utils.FD_all_subjects(translation_mca_failed_PD, angles_mca_failed_PD)
    result_failed_HC = metrics_utils.FD_all_subjects(translation_mca_failed_HC, angles_mca_failed_HC)

    result_all_PD = metrics_utils.FD_all_subjects(translation_mca_PD, angles_mca_PD)
    result_all_HC = metrics_utils.FD_all_subjects(translation_mca_HC, angles_mca_HC)

    basic_info_plotter(
        result_fine_PD,
        result_fine_HC,
        software=software,
        variable="Framewise Displacement",
        path=path,
        figure_size=(5, 4),
        ylable="(mm)",
        y_lim_sd=[(0, 2)],
    )
    #    y_lim_sd=[(0,4), (0,0.2), (0,0.2)])
    #  fsl y_lim_sd=[(0,.3)]
    # y_lim_mean=[(15,110)]
    basic_info_plotter(
        result_failed_PD, result_failed_HC, software=software, variable="Framewise Displacement failed", path=path, figure_size=(5, 4), ylable="(mm)"
    )
    basic_info_plotter(
        result_all_PD, result_all_HC, software=software, variable="Framewise Displacement All", path=path, figure_size=(5, 4), ylable="(mm)"
    )

    np.savetxt(path.parent / f"{software}_FD_PD_all.txt", result_all_PD)
    np.savetxt(path.parent / f"{software}_FD_HC_all.txt", result_all_HC)

    np.savetxt(path.parent / f"{software}_FD_PD_failed.txt", result_failed_PD)
    np.savetxt(path.parent / f"{software}_FD_HC_failed.txt", result_failed_HC)

    np.savetxt(path.parent / f"{software}_FD_PD_fine.txt", result_fine_PD)
    np.savetxt(path.parent / f"{software}_FD_HC_fine.txt", result_fine_HC)

    new_result_fine_PD = metrics_utils.FD_all_subjects(translation_mca_fine_PD, angles_mca_fine_PD, translation_ieee_fine_PD, angles_ieee_fine_PD)
    new_result_fine_HC = metrics_utils.FD_all_subjects(translation_mca_fine_HC, angles_mca_fine_HC, translation_ieee_fine_HC, angles_ieee_fine_HC)

    new_result_failed_PD = metrics_utils.FD_all_subjects(
        translation_mca_failed_PD, angles_mca_failed_PD, translation_ieee_failed_PD, angles_ieee_failed_PD
    )
    new_result_failed_HC = metrics_utils.FD_all_subjects(
        translation_mca_failed_HC, angles_mca_failed_HC, translation_ieee_failed_HC, angles_ieee_failed_HC
    )

    new_result_all_PD = metrics_utils.FD_all_subjects(translation_mca_PD, angles_mca_PD, translation_ieee_PD, angles_ieee_PD)
    new_result_all_HC = metrics_utils.FD_all_subjects(translation_mca_HC, angles_mca_HC, translation_ieee_HC, angles_ieee_HC)

    basic_info_plotter(
        new_result_fine_PD,
        new_result_fine_HC,
        software=software,
        variable="Framewise Displacement (IEEE Reference)",
        path=path,
        figure_size=(5, 4),
        ylable="(mm)",
        y_lim_sd=[(0, 0.5)],
        y_lim_mean=[(0, 1)],
    )
    basic_info_plotter(
        new_result_failed_PD,
        new_result_failed_HC,
        software=software,
        variable="Framewise Displacement failed (IEEE Reference)",
        path=path,
        figure_size=(5, 4),
        ylable="(mm)",
    )
    basic_info_plotter(
        new_result_all_PD,
        new_result_all_HC,
        software=software,
        variable="Framewise Displacement All(IEEE Reference)",
        path=path,
        figure_size=(5, 4),
        ylable="(mm)",
    )

    np.savetxt(path.parent / f"{software}_new_FD_PD_all.txt", new_result_all_PD)
    np.savetxt(path.parent / f"{software}_new_FD_HC_all.txt", new_result_all_HC)

    np.savetxt(path.parent / f"{software}_new_FD_PD_failed.txt", new_result_failed_PD)
    np.savetxt(path.parent / f"{software}_new_FD_HC_failed.txt", new_result_failed_HC)

    np.savetxt(path.parent / f"{software}_new_FD_PD_fine.txt", new_result_fine_PD)
    np.savetxt(path.parent / f"{software}_new_FD_HC_fine.txt", new_result_fine_HC)

    # largest_indces(np.std(result_all_PD, axis=1), mat_dic_PD, n=10)
    # largest_indces(np.std(result_all_HC, axis=1), mat_dic_HC, n=10)

    # not quite usefull
    # mat_mca_HC  = concatenate_mca_matrices(mat_dic_HC)
    # mat_mca_PD  = concatenate_mca_matrices(mat_dic_PD)

    # fd_improved_HC = metrics_utils.FD_all_subjects_improved(mat_mca_HC, 100)
    # largest_indces(np.std(fd_improved_HC, axis=1), mat_dic_HC, n=10)

    # fd_improved_PD = metrics_utils.FD_all_subjects_improved(mat_mca_PD, 100)
    # largest_indces(np.std(fd_improved_PD, axis=1), mat_dic_PD, n=10)

    import pandas as pd

    df_fd_pd_new = pd.DataFrame(new_result_all_PD)
    df_fd_pd_new["new_fd"] = df_fd_pd_new.apply(lambda row: row.values, axis=1)
    df_fd_pd_new = df_fd_pd_new[["new_fd"]]
    df_fd_pd_new.index = mat_dic_PD.keys()

    df_fd_pd_old = pd.DataFrame(result_all_PD)
    df_fd_pd_old["old_fd"] = df_fd_pd_old.apply(lambda row: row.values, axis=1)
    df_fd_pd_old = df_fd_pd_old[["old_fd"]]
    df_fd_pd_old.index = mat_dic_PD.keys()

    df_combined_pd = pd.concat([df_fd_pd_new, df_fd_pd_old], axis=1)
    df_combined_pd["Group"] = "PD"

    df_fd_hc_new = pd.DataFrame(new_result_all_HC)
    df_fd_hc_new.index = mat_dic_HC.keys()
    df_fd_hc_new["new_fd"] = df_fd_hc_new.apply(lambda row: row.values, axis=1)
    df_fd_hc_new = df_fd_hc_new[["new_fd"]]

    df_fd_hc_old = pd.DataFrame(result_all_HC)
    df_fd_hc_old.index = mat_dic_HC.keys()
    df_fd_hc_old["old_fd"] = df_fd_hc_old.apply(lambda row: row.values, axis=1)
    df_fd_hc_old = df_fd_hc_old[["old_fd"]]

    df_combined_hc = pd.concat([df_fd_hc_new, df_fd_hc_old], axis=1)
    df_combined_hc["Group"] = "HC"

    df_combined = pd.concat([df_combined_pd, df_combined_hc], axis=0)
    df_combined.to_csv("fds_fsl.csv")
