from lnrgst_mca.plot_utils import plotter, hist_plotter
from lnrgst_mca import metrics_utils
from config import get_configurations, FD_mean_bin_sizes, FD_sd_bin_sizes
from copy import deepcopy
from scipy import stats
import numpy as np
import argparse

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


def concatenate_cohorts(g1, g2):
    if len(g1) == 0:
        return g2
    if len(g2) == 0:
        return g2
    return np.concatenate([g1, g2])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--template", type=str, help="template name")
    parser.add_argument("-s", "--software", type=str, help="software name")
    parser.add_argument("-d", "--diagram_path", type=str, help="path to save output plots")

    args = parser.parse_args()
    template = args.template
    software = args.software
    diagram_path = Path(args.diagram_path) / software / template
    diagram_path.mkdir(parents=True, exist_ok=True)

    if software not in FD_mean_bin_sizes or software not in FD_sd_bin_sizes:
        raise ValueError(f"unknown software {software}")

    if template not in FD_mean_bin_sizes[software].keys() or template not in FD_sd_bin_sizes[software].keys():
        print(FD_mean_bin_sizes[software].keys())
        raise ValueError(f"unknown template {template}")

    # add something to check validity of s and t
    # loading subject matrix dictonaries and list of failed subjects
    failed_subjects_HC, failed_subjects_PD, mat_dic_PD, error_PD, mat_dic_HC, error_HC = get_configurations(software, template)

    # cheking if there is problem in loading mca matrices
    if error_PD or error_HC:
        print(error_PD, error_HC)
        raise Exception("There is an issue with mat files")

    # deviding dictionaries to failed and fine subjects (based on IEEE QC)
    mat_dic_fine_PD, mat_dic_failed_PD = copy_and_remove_keys(mat_dic_PD, failed_subjects_PD)
    mat_dic_fine_HC, mat_dic_failed_HC = copy_and_remove_keys(mat_dic_HC, failed_subjects_HC)

    # decomposing the matrices to 12 parameters

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

    # change radians vectors to degrees
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

    # plotting mean and standard deviation for 12 parameters

    basic_info_plotter(
        translation_mca_fine_PD,
        translation_mca_fine_HC,
        software=software,
        variable="Translations",
        path=diagram_path,
        axis_labels=["x", "y", "z"],
        ylable="(mm)",
        y_lim_mean=[(-14, 12), (-60, 25), (-42, 61)],
        y_lim_sd=[(0, 0.1), (0, 0.2), (0, 0.2)],
    )

    basic_info_plotter(
        angles_mca_fine_PD,
        angles_mca_fine_HC,
        software=software,
        variable="Angles",
        path=diagram_path,
        axis_labels=["x", "y", "z"],
        ylable="(degree)",
        y_lim_mean=[(-40, 5), (-10.5, 8), (-11, 11)],
        y_lim_sd=[(0, 0.5), (0, 0.25), (0, 0.26)],
    )

    basic_info_plotter(
        scales_mca_fine_PD,
        scales_mca_fine_HC,
        software=software,
        variable="Scales",
        path=diagram_path,
        axis_labels=["x", "y", "z"],
        y_lim_mean=[(0.825, 1.27), (0.8, 1.32), (0.7, 1.35)],
        y_lim_sd=[(0, 0.002), (0, 0.0025), (0, 0.005)],
    )

    basic_info_plotter(
        shears_mca_fine_PD,
        shears_mca_fine_HC,
        software=software,
        variable="Shears",
        path=diagram_path,
        axis_labels=["x", "y", "z"],
        y_lim_mean=[(-0.065, 0.065), (-0.045, 0.045), (-0.22, 0.105)],
        y_lim_sd=[(0, 0.003), (0, 0.004), (0, 0.008)],
    )

    basic_info_plotter(
        translation_mca_failed_PD,
        translation_mca_failed_HC,
        software=software,
        variable="Translations failed",
        path=diagram_path,
        axis_labels=["x", "y", "z"],
        ylable="(mm)",
    )
    basic_info_plotter(
        angles_mca_failed_PD,
        angles_mca_failed_HC,
        software=software,
        variable="Angles failed",
        path=diagram_path,
        axis_labels=["x", "y", "z"],
        ylable="(degree)",
    )
    basic_info_plotter(
        scales_mca_failed_PD, scales_mca_failed_HC, software=software, variable="Scales failed", path=diagram_path, axis_labels=["x", "y", "z"]
    )
    basic_info_plotter(
        shears_mca_failed_PD, shears_mca_failed_HC, software=software, variable="Shears failed", path=diagram_path, axis_labels=["x", "y", "z"]
    )

    # calculating framewise displacment

    result_fine_PD = metrics_utils.FD_all_subjects(translation_mca_fine_PD, angles_mca_fine_PD)
    result_fine_HC = metrics_utils.FD_all_subjects(translation_mca_fine_HC, angles_mca_fine_HC)

    result_failed_PD = metrics_utils.FD_all_subjects(translation_mca_failed_PD, angles_mca_failed_PD)
    result_failed_HC = metrics_utils.FD_all_subjects(translation_mca_failed_HC, angles_mca_failed_HC)

    result_all_PD = metrics_utils.FD_all_subjects(translation_mca_PD, angles_mca_PD)
    result_all_HC = metrics_utils.FD_all_subjects(translation_mca_HC, angles_mca_HC)

    result_fine_ieee_PD = metrics_utils.FD_all_subjects(translation_ieee_fine_PD[:, np.newaxis, :], angles_ieee_fine_PD[:, np.newaxis, :])
    result_fine_ieee_HC = metrics_utils.FD_all_subjects(translation_ieee_fine_HC[:, np.newaxis, :], angles_ieee_fine_HC[:, np.newaxis, :])

    result_failed_ieee_PD = metrics_utils.FD_all_subjects(translation_ieee_failed_PD[:, np.newaxis, :], angles_ieee_failed_PD[:, np.newaxis, :])
    result_failed_ieee_HC = metrics_utils.FD_all_subjects(translation_ieee_failed_HC[:, np.newaxis, :], angles_ieee_failed_HC[:, np.newaxis, :])

    result_all_ieee_PD = metrics_utils.FD_all_subjects(translation_ieee_PD[:, np.newaxis, :], angles_ieee_PD[:, np.newaxis, :])
    result_all_ieee_HC = metrics_utils.FD_all_subjects(translation_ieee_HC[:, np.newaxis, :], angles_ieee_HC[:, np.newaxis, :])

    # plotting framewise displacment
    basic_info_plotter(
        result_fine_PD,
        result_fine_HC,
        software=software,
        variable="Framewise Displacement",
        path=diagram_path,
        figure_size=(5, 4),
        ylable="(mm)",
        y_lim_mean=[(10, 110)],
        y_lim_sd=[(0, 0.3)],
    )

    basic_info_plotter(
        result_failed_PD,
        result_failed_HC,
        software=software,
        variable="Framewise Displacement failed",
        path=diagram_path,
        figure_size=(5, 4),
        ylable="(mm)",
    )
    basic_info_plotter(
        result_all_PD, result_all_HC, software=software, variable="Framewise Displacement All", path=diagram_path, figure_size=(5, 4), ylable="(mm)"
    )

    all_fd_mca_failed = concatenate_cohorts(result_failed_PD, result_failed_HC)
    all_fd_mca_fine = concatenate_cohorts(result_fine_PD, result_fine_HC)

    hist_plotter(
        datasets=[np.mean(all_fd_mca_fine, axis=1), np.mean(all_fd_mca_failed, axis=1)],
        title=f"Mean FD: {software} - {template}",
        path=diagram_path,
        bins=FD_mean_bin_sizes[software][template],
        labels=["Passed", "failed"],
    )
    hist_plotter(
        datasets=[np.std(all_fd_mca_fine, axis=1), np.std(all_fd_mca_failed, axis=1)],
        title=f"SD of  FD: {software} - {template}",
        path=diagram_path,
        bins=FD_sd_bin_sizes[software][template],
        labels=["Passed", "failed"],
    )

    # saving FD
    # path = diagram_path.parent
    path = diagram_path
    np.savetxt(path / f"{software}_FD_PD_all.txt", result_all_PD)
    np.savetxt(path / f"{software}_FD_HC_all.txt", result_all_HC)

    np.savetxt(path / f"{software}_FD_PD_failed.txt", result_failed_PD)
    np.savetxt(path / f"{software}_FD_HC_failed.txt", result_failed_HC)

    np.savetxt(path / f"{software}_FD_PD_fine.txt", result_fine_PD)
    np.savetxt(path / f"{software}_FD_HC_fine.txt", result_fine_HC)

    np.savetxt(path / f"{software}_FD_ieee_PD_all.txt", result_all_ieee_PD)
    np.savetxt(path / f"{software}_FD_ieee_HC_all.txt", result_all_ieee_HC)

    np.savetxt(path / f"{software}_FD_ieee_PD_failed.txt", result_failed_ieee_PD)
    np.savetxt(path / f"{software}_FD_ieee_HC_failed.txt", result_failed_ieee_HC)

    np.savetxt(path / f"{software}_FD_ieee_PD_fine.txt", result_fine_ieee_PD)
    np.savetxt(path / f"{software}_FD_ieee_HC_fine.txt", result_fine_ieee_HC)

    # calculating the mean absolute difference between MCA framewise displacements and the IEEE framewise displacement
    mad_fine_PD = metrics_utils.mean_absolute_difference(result_fine_PD, result_fine_ieee_PD)
    mad_fine_HC = metrics_utils.mean_absolute_difference(result_fine_HC, result_fine_ieee_HC)

    mad_failed_PD = metrics_utils.mean_absolute_difference(result_failed_PD, result_failed_ieee_PD)
    mad_failed_HC = metrics_utils.mean_absolute_difference(result_failed_HC, result_failed_ieee_HC)

    mad_all_PD = metrics_utils.mean_absolute_difference(result_all_PD, result_all_ieee_PD)
    mad_all_HC = metrics_utils.mean_absolute_difference(result_all_HC, result_all_ieee_HC)

    all_mad_failed = concatenate_cohorts(mad_failed_PD, mad_failed_HC)
    all_mad_fine = concatenate_cohorts(mad_fine_PD, mad_fine_HC)
    hist_plotter(
        datasets=[all_mad_fine, all_mad_failed],
        title=f"Mean Absolute Difference of FD: {software} - {template}",
        path=diagram_path,
        labels=["Fine", "Failed"],
    )

    # saving MAD
    np.savetxt(path / f"{software}_mad_fine_PD.txt", mad_fine_PD)
    np.savetxt(path / f"{software}_mad_fine_HC.txt", mad_fine_HC)

    np.savetxt(path / f"{software}_mad_failed_PD.txt", mad_failed_PD)
    np.savetxt(path / f"{software}_mad_failed_HC.txt", mad_failed_HC)

    np.savetxt(path / f"{software}_mad_all_PD.txt", mad_all_PD)
    np.savetxt(path / f"{software}_mad_all_HC.txt", mad_all_HC)

    # Printing information

    t, p = stats.ttest_ind(np.log(np.std(result_all_PD, axis=1)), np.log(np.std(result_all_HC, axis=1)))
    t_fine, p_fine = stats.ttest_ind(np.log(np.std(result_fine_PD, axis=1)), np.log(np.std(result_fine_HC, axis=1)))

    print(f"t:{t:.3f} p:{p:.3f}")
    print(f"t_fine:{t_fine:.3f} p_fine:{p_fine:.3f}")
    print(f"Mean of Mean Absolute error in fine registrations across subjects: {np.mean(all_mad_fine)}")
    print(f"Mean of  Mean Absolute error in failed registrations across subjects: {np.mean(all_mad_failed)}")
    print(f"Max Mean Absolute error in fine registrations across subjects: {np.max(all_mad_fine)}")
    print(f"Max Mean Absolute error in failed registrations across subjects: {np.max(all_mad_failed)}")
