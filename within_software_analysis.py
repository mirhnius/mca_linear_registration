import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from copy import deepcopy
from fsl.transform import affine

from lnrgst_mca import metrics_utils
from lnrgst_mca.plot_utils import plotter, swarm_QC  # , hist_plotter
from config import (
    get_configurations,
    FD_mean_bin_sizes,
    FD_sd_bin_sizes,
    short_template_names,
    failed_palette_colors,
    palette_colors,
    palette_colors_similarity_measures,
    failed_palette_colors_similarity_measures,
    cost_function_names
)  # , FD_SD_x_lim


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
        return g1
    return np.concatenate([g1, g2], axis=0)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def validate_arguments(args):

    if args.software not in FD_mean_bin_sizes or args.software not in FD_sd_bin_sizes:
        raise ValueError(f"Unknown software: {args.software}")

    if args.template not in FD_mean_bin_sizes[args.software].keys() or args.template not in FD_sd_bin_sizes[args.software].keys():
        raise ValueError(f"Unknown template: {args.template}")


def decompose_and_convert(mat_dict, n_mca=10, mode="mca"):
    scales, translations, angles, shears = transformation_dictionary_to_arrays(mat_dict, n_mca=n_mca, mode=mode)
    angles = np.degrees(angles)
    return scales, translations, angles, shears


def save_array(software, results, path, fmt="%.18e"):

    for key, value in results.items():
        np.savetxt(path / f"{software}_{key}.txt", value, fmt=fmt)


def recall(predictions):

    TP = predictions.count(-1)
    FN = predictions.count(1)

    return TP / (TP + FN) if (TP + FN) > 0 else 0


def generate_report(path, software, template, t, p, t_fine, p_fine, p_shapiro, all_mad_fine, all_mad_failed, IDs_failed, probabilities, predictions):
    with open(path / "report.txt", "w") as f:
        f.write("-------------- {} - {} --------------\n".format(software, template))
        f.write("HC vs PD mannwhitneyu on standard deviation of framewise displacement for all subjects  stat:{:.3f} p:{:.3f}\n".format(t, p))
        f.write(
            "HC vs PD mannwhitneyu on standard deviation of framewise displacement for passed QC subjects  stat:{:.3f} p:{:.3f}\n".format(
                t_fine, p_fine
            )
        )
        f.write("Mean of Mean Absolute Difference of passed QC subjects: {:.3e} mm\n".format(np.mean(all_mad_fine)))
        f.write("Mean of Mean Absolute Difference of failed QC subjects: {:.3e} mm\n".format(np.mean(all_mad_failed)))
        f.write("Max of Mean Absolute Difference of passed QC subjects: {:.3e} mm\n".format(np.max(all_mad_fine)))
        f.write("Max of Mean Absolute Difference of failed QC subjects: {:.3e} mm\n".format(np.max(all_mad_failed)))
        f.write("Number of passed QC subjects with higher than 1 mm Mean Absolute Difference: {}\n".format(np.sum(all_mad_fine >= 1.0)))
        f.write("Number of passed QC subjects with higher than 0.2 mm Mean Absolute Difference: {}\n".format(np.sum(all_mad_fine >= 0.2)))
        f.write("\n")
        f.write(f"Normality test {p_shapiro} \n")
        for key, value in predictions.items():
            f.write(f"Predictions using {key}:\n")
            f.write(f"{value}\n")
            f.write(f"recall: {recall(value)}\n")
        f.write("\n")
        f.write("------------------------------------------------------------\n")
        f.write("\n")
        f.write("pdf estimations for failed subjects:\n")
        for i, id in enumerate(IDs_failed):
            f.write("{}: {:.3e}\n".format(id, probabilities[i]))
        f.write("------------------------------------------------------------\n")


if __name__ == "__main__":

    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template", type=str, help="Template name", required=True)
    parser.add_argument("-s", "--software", type=str, help="Software name", required=True)
    parser.add_argument("-d", "--diagram_path", type=str, help="Path to save output plots", required=True)
    parser.add_argument("-c", "--cost_function", type=str, help="Cost function", default=False)
    parser.add_argument("-v", "--verrou", type=bool, help="Cost function", default=False)
    args = parser.parse_args()

    validate_arguments(args)

    template = args.template
    software = args.software
    cost_function = args.cost_function
    verrou = args.verrou
    diagram_path = Path(args.diagram_path) / software / template
    if args.cost_function:
        diagram_path = diagram_path / cost_function
    diagram_path.mkdir(parents=True, exist_ok=True)

    # Load configurations
    failed_subjects_HC, failed_subjects_PD, mat_dic_PD, error_PD, mat_dic_HC, error_HC = get_configurations(software, template, cost_function, verrou)

    if error_PD or error_HC:
        logging.error("Error in loading MCA matrices: %s, %s", error_PD, error_HC)
        print(error_PD, error_HC)
        raise Exception("There is an issue with mat files")

    mat_dic_fine_PD, mat_dic_failed_PD = copy_and_remove_keys(mat_dic_PD, failed_subjects_PD)
    mat_dic_fine_HC, mat_dic_failed_HC = copy_and_remove_keys(mat_dic_HC, failed_subjects_HC)

    # Decompose matrices
    scales_mca_fine_PD, translations_mca_fine_PD, angles_mca_fine_PD, shears_mca_fine_PD = decompose_and_convert(mat_dic_fine_PD)
    scales_mca_fine_HC, translations_mca_fine_HC, angles_mca_fine_HC, shears_mca_fine_HC = decompose_and_convert(mat_dic_fine_HC)

    scales_mca_failed_PD, translations_mca_failed_PD, angles_mca_failed_PD, shears_mca_failed_PD = decompose_and_convert(mat_dic_failed_PD)
    scales_mca_failed_HC, translations_mca_failed_HC, angles_mca_failed_HC, shears_mca_failed_HC = decompose_and_convert(mat_dic_failed_HC)

    scales_ieee_fine_PD, translation_ieee_fine_PD, angles_ieee_fine_PD, shears_ieee_fine_P = decompose_and_convert(
        mat_dic_fine_PD, n_mca=1, mode="ieee"
    )
    scales_ieee_fine_HC, translation_ieee_fine_HC, angles_ieee_fine_HC, shears_ieee_fine_HC = decompose_and_convert(
        mat_dic_fine_HC, n_mca=1, mode="ieee"
    )

    scales_ieee_failed_PD, translation_ieee_failed_PD, angles_ieee_failed_PD, shears_ieee_failed_PD = decompose_and_convert(
        mat_dic_failed_PD, n_mca=1, mode="ieee"
    )
    scales_ieee_failed_HC, translation_ieee_failed_HC, angles_ieee_failed_HC, shears_ieee_failed_HC = decompose_and_convert(
        mat_dic_failed_HC, n_mca=1, mode="ieee"
    )

    # Calculate framewise displacement
    FD_mca_results = {
        "FD_PD_fine": metrics_utils.FD_all_subjects(translations_mca_fine_PD, angles_mca_fine_PD),
        "FD_HC_fine": metrics_utils.FD_all_subjects(translations_mca_fine_HC, angles_mca_fine_HC),
        "FD_PD_failed": metrics_utils.FD_all_subjects(translations_mca_failed_PD, angles_mca_failed_PD),
        "FD_HC_failed": metrics_utils.FD_all_subjects(translations_mca_failed_HC, angles_mca_failed_HC),
    }

    FD_mca_results["FD_all_fine"] = concatenate_cohorts(FD_mca_results["FD_PD_fine"], FD_mca_results["FD_HC_fine"])
    FD_mca_results["FD_all_failed"] = concatenate_cohorts(FD_mca_results["FD_PD_failed"], FD_mca_results["FD_HC_failed"])
    FD_mca_results["FD_all_HC"] = concatenate_cohorts(FD_mca_results["FD_HC_fine"], FD_mca_results["FD_HC_failed"])
    FD_mca_results["FD_all_PD"] = concatenate_cohorts(FD_mca_results["FD_PD_fine"], FD_mca_results["FD_PD_failed"])

    FD_ieee_results = {
        "FD_ieee_PD_fine": metrics_utils.FD_all_subjects(translation_ieee_fine_PD[:, np.newaxis, :], angles_ieee_fine_PD[:, np.newaxis, :]),
        "FD_ieee_HC_fine": metrics_utils.FD_all_subjects(translation_ieee_fine_HC[:, np.newaxis, :], angles_ieee_fine_HC[:, np.newaxis, :]),
        "FD_ieee_PD_failed": metrics_utils.FD_all_subjects(translation_ieee_failed_PD[:, np.newaxis, :], angles_ieee_failed_PD[:, np.newaxis, :]),
        "FD_ieee_HC_failed": metrics_utils.FD_all_subjects(translation_ieee_failed_HC[:, np.newaxis, :], angles_ieee_failed_HC[:, np.newaxis, :]),
    }

    FD_ieee_results["FD_ieee_all_fine"] = concatenate_cohorts(FD_ieee_results["FD_ieee_PD_fine"], FD_ieee_results["FD_ieee_HC_fine"])
    FD_ieee_results["FD_ieee_all_failed"] = concatenate_cohorts(FD_ieee_results["FD_ieee_PD_failed"], FD_ieee_results["FD_ieee_HC_failed"])

    # calculating the mean absolute difference between MCA framewise displacements and the IEEE framewise displacement
    MAD_results = {
        "mad_fine_PD": metrics_utils.mean_absolute_difference(FD_mca_results["FD_PD_fine"], FD_ieee_results["FD_ieee_PD_fine"]),
        "mad_fine_HC": metrics_utils.mean_absolute_difference(FD_mca_results["FD_HC_fine"], FD_ieee_results["FD_ieee_HC_fine"]),
        "mad_failed_PD": metrics_utils.mean_absolute_difference(FD_mca_results["FD_PD_failed"], FD_ieee_results["FD_ieee_PD_failed"]),
        "mad_failed_HC": metrics_utils.mean_absolute_difference(FD_mca_results["FD_HC_failed"], FD_ieee_results["FD_ieee_HC_failed"]),
    }

    MAD_results["all_mad_failed"] = concatenate_cohorts(MAD_results["mad_failed_PD"], MAD_results["mad_failed_HC"])
    MAD_results["all_mad_fine"] = concatenate_cohorts(MAD_results["mad_fine_PD"], MAD_results["mad_fine_HC"])
    MAD_results["all_mad"] = concatenate_cohorts(MAD_results["all_mad_fine"], MAD_results["all_mad_failed"])
    # "mad_all_PD": metrics_utils.mean_absolute_difference(FD_mca_results[result_all_PD, FD_mca_results[result_all_ieee_PD),
    # "mad_all_HC": metrics_utils.mean_absolute_difference(FD_mca_results[result_all_HC, FD_mca_results[result_all_ieee_HC)}

    # plotting mean and standard deviation for 12 parameters

    # basic_info_plotter(
    #     translations_mca_fine_PD,
    #     translations_mca_fine_HC,
    #     software=software,
    #     variable="Translations",
    #     path=diagram_path,
    #     axis_labels=["x", "y", "z"],
    #     ylable="(mm)",
    #     y_lim_mean=[(-14, 12), (-60, 25), (-42, 61)],
    #     y_lim_sd=[(0, 0.1), (0, 0.2), (0, 0.2)],
    # )

    # basic_info_plotter(
    #     angles_mca_fine_PD,
    #     angles_mca_fine_HC,
    #     software=software,
    #     variable="Angles",
    #     path=diagram_path,
    #     axis_labels=["x", "y", "z"],
    #     ylable="(degree)",
    #     y_lim_mean=[(-40, 5), (-10.5, 8), (-11, 11)],
    #     y_lim_sd=[(0, 0.5), (0, 0.25), (0, 0.26)],
    # )

    # basic_info_plotter(
    #     scales_mca_fine_PD,
    #     scales_mca_fine_HC,
    #     software=software,
    #     variable="Scales",
    #     path=diagram_path,
    #     axis_labels=["x", "y", "z"],
    #     y_lim_mean=[(0.825, 1.27), (0.8, 1.32), (0.7, 1.35)],
    #     y_lim_sd=[(0, 0.002), (0, 0.0025), (0, 0.005)],
    # )

    # basic_info_plotter(
    #     shears_mca_fine_PD,
    #     shears_mca_fine_HC,
    #     software=software,
    #     variable="Shears",
    #     path=diagram_path,
    #     axis_labels=["x", "y", "z"],
    #     y_lim_mean=[(-0.065, 0.065), (-0.045, 0.045), (-0.22, 0.105)],
    #     y_lim_sd=[(0, 0.003), (0, 0.004), (0, 0.008)],
    # )

    # basic_info_plotter(
    #     translations_mca_failed_PD,
    #     translations_mca_failed_HC,
    #     software=software,
    #     variable="Translations failed",
    #     path=diagram_path,
    #     axis_labels=["x", "y", "z"],
    #     ylable="(mm)",
    # )
    # basic_info_plotter(
    #     angles_mca_failed_PD,
    #     angles_mca_failed_HC,
    #     software=software,
    #     variable="Angles failed",
    #     path=diagram_path,
    #     axis_labels=["x", "y", "z"],
    #     ylable="(degree)",
    # )
    # basic_info_plotter(
    #     scales_mca_failed_PD, scales_mca_failed_HC, software=software, variable="Scales failed", path=diagram_path, axis_labels=["x", "y", "z"]
    # )
    # basic_info_plotter(
    #     shears_mca_failed_PD, shears_mca_failed_HC, software=software, variable="Shears failed", path=diagram_path, axis_labels=["x", "y", "z"]
    # )

    # # plotting framewise displacment
    # basic_info_plotter(
    #     FD_mca_results["FD_PD_fine"],
    #     FD_mca_results["FD_HC_fine"],
    #     software=software,
    #     variable="Framewise Displacement",
    #     path=diagram_path,
    #     figure_size=(5, 4),
    #     ylable="(mm)",
    #     y_lim_mean=[(10, 110)],
    #     y_lim_sd=[(0, 0.3)],
    # )

    # basic_info_plotter(
    #     FD_mca_results["FD_PD_failed"],
    #     FD_mca_results["FD_HC_failed"],
    #     software=software,
    #     variable="Framewise Displacement failed",
    #     path=diagram_path,
    #     figure_size=(5, 4),
    #     ylable="(mm)",
    # )
    # basic_info_plotter(
    #     FD_mca_results["FD_all_PD"],
    #     FD_mca_results["FD_all_HC"],
    #     software=software,
    #     variable="Framewise Displacement All",
    #     path=diagram_path,
    #     figure_size=(5, 4),
    #     ylable="(mm)",
    # )

    # hist_plotter(
    #     datasets=[np.mean(FD_mca_results["FD_all_fine"], axis=1), np.mean(FD_mca_results["FD_all_failed"], axis=1)],
    #     title=f"Mean FD: {software} - {template}",
    #     path=diagram_path,
    #     bins=FD_mean_bin_sizes[software][template],
    #     labels=["Passed", "failed"],
    #     xlabel="(mm)",
    # )
    # hist_plotter(
    #     datasets=[np.std(FD_mca_results["FD_all_fine"], axis=1), np.std(FD_mca_results["FD_all_failed"], axis=1)],
    #     title=f"SD of  FD: {software} - {template}",
    #     path=diagram_path,
    #     bins=FD_sd_bin_sizes[software][template],
    #     labels=["Passed", "failed"],
    #     xlim=FD_SD_x_lim[software][template],
    #     xlabel="(mm)",
    # )

    # hist_plotter(
    #     datasets=[MAD_results["all_mad_fine"], MAD_results["all_mad_failed"]],
    #     title=f"Mean Absolute Difference of FD: {software} - {template}",
    #     path=diagram_path,
    #     labels=["Passed", "Failed"],
    #     xlabel="(mm)",
    # )

    # hist_plotter(
    #     datasets=[np.log10(np.mean(FD_mca_results["FD_all_fine"], axis=1)), np.log10(np.mean(FD_mca_results["FD_all_failed"], axis=1))],
    #     title=f"log10 Mean FD Per Subject: {software} - {short_template_names[template]}",
    #     path=diagram_path,
    #     bins=FD_mean_bin_sizes[software][template],
    #     labels=["Passed", "failed"],
    #     xlabel="Value(log10) (mm)",
    #     txt=False
    # )

    # hist_plotter(
    #     datasets=[np.log10(np.std(FD_mca_results["FD_all_fine"], axis=1)), np.log10(np.std(FD_mca_results["FD_all_failed"], axis=1))],
    #     title=f"log10 SD of FD Per Subject: {software} - {short_template_names[template]}",
    #     path=diagram_path,
    #     bins=FD_sd_bin_sizes[software][template],
    #     labels=["Passed", "Failed"],
    #     xlabel="log10(value) (mm)",
    #     txt=False
    # )

    # plotter(np.log(np.mean(FD_mca_results["FD_PD_fine"], axis=1))
    # , np.log(np.mean(FD_mca_results["FD_HC_fine"], axis=1)),
    # title=f"Mean of FD Per Subject {software} - {short_template_names[template]}: Cohort Comparison", path=diagram_path, ylable="Value(log10) (mm)")

    cost_function_name = None if cost_function is None else cost_function_names[cost_function]
    
    plotter(np.log(np.std(FD_mca_results["FD_PD_fine"], axis=1)),
     np.log(np.std(FD_mca_results["FD_HC_fine"], axis=1)),
     title=f"SD of FD Per Subject {software.upper()} - {short_template_names[template]}: Cohort Comparison", path=diagram_path, ylable="log10(Value) (mm)", axis_labels=[f'{cost_function_name} simliarity measure'])
    
    failed_palette = failed_palette_colors[software][template] if cost_function is None else failed_palette_colors_similarity_measures[software][template][cost_function]
    palette = palette_colors[software][template] if cost_function is None else palette_colors_similarity_measures[software][template][cost_function]

    swarm_QC(
        np.std(FD_mca_results["FD_PD_fine"], axis=1),
        np.std(FD_mca_results["FD_all_failed"], axis=1),
        software,
        short_template_names[template],
        palette,
        failed_palette,
        path=diagram_path,
        cost_function=cost_function_name
    )

    # saving
    path = diagram_path / "reports"
    path.mkdir(parents=True, exist_ok=True)
    # Save IDs
    IDs = {
        "IDs_fine": concatenate_cohorts(np.array(list(mat_dic_fine_PD.keys())), np.array(list(mat_dic_fine_HC.keys()))),
        "IDs_failed": concatenate_cohorts(np.array(list(mat_dic_failed_PD.keys())), np.array(list(mat_dic_failed_HC.keys()))),
    }
    IDs["IDs_all"] = concatenate_cohorts(IDs["IDs_fine"], IDs["IDs_failed"])
    save_array(software, IDs, path, fmt="%s")

    from scipy.stats import shapiro

    stat, p_shapiro = shapiro(np.std(FD_mca_results["FD_all_fine"], axis=1))
    print("Normality test", p_shapiro)

    # t test on standard deviation of cohorts to test if varibility of healthy cohort is different from parkinsonian paitents
    # since even the log data is not normal t-test is not reliable here.
    # t, p = stats.ttest_ind(np.log(np.std(FD_mca_results["FD_all_PD"], axis=1)), np.log(np.std(FD_mca_results["FD_all_HC"], axis=1)))
    # t_fine, p_fine = stats.ttest_ind(np.log(np.std(FD_mca_results["FD_PD_fine"], axis=1)), np.log(np.std(FD_mca_results["FD_HC_fine"], axis=1)))

    from scipy.stats import mannwhitneyu

    stat, p = mannwhitneyu(np.std(FD_mca_results["FD_all_PD"], axis=1), np.std(FD_mca_results["FD_all_HC"], axis=1), alternative="two-sided")
    stat_fine, p_fine = mannwhitneyu(
        np.std(FD_mca_results["FD_PD_fine"], axis=1), np.std(FD_mca_results["FD_HC_fine"], axis=1), alternative="two-sided"
    )

    # fit a normal distribution to sd of passed qc subjects: use for to study wether failed subjects are outlier
    # This assumes that data is normal which is not in my case

    # from scipy.stats import norm
    # log_std_fine = np.log(np.std(FD_mca_results["FD_all_fine"], axis=1))
    # fine_mean_of_std = np.mean(log_std_fine)
    # fine_std_of_std = np.std(log_std_fine)

    # pdf_fine = norm.pdf(log_std_fine, loc=fine_mean_of_std, scale=fine_std_of_std)
    # threshold = np.percentile(pdf_fine, 5)  # Set threshold as the 5th percentile
    # log_std_failed = np.log(np.std(FD_mca_results["FD_all_failed"], axis=1))
    # probabilities = norm.pdf(log_std_failed, loc=fine_mean_of_std, scale=fine_std_of_std)

    # print(threshold)
    # print(probabilities)
    # print(probabilities < threshold)

    # Using kde which is a non parametric pdf estimation method
    # putting threshold on density by finding density on the 5th quantile and compare to density for failed subjects
    # "If the density of a failed subject is lower than the density of 95% of passed subjects, it is an outlier."
    from sklearn.neighbors import KernelDensity

    std_fine = np.std(FD_mca_results["FD_all_fine"], axis=1)
    std_failed = np.std(FD_mca_results["FD_all_failed"], axis=1)

    kde = KernelDensity(kernel="exponential", bandwidth=0.01).fit(std_fine.reshape(-1, 1))

    probs_fine = kde.score_samples(std_fine.reshape(-1, 1))
    threshold = np.percentile(probs_fine, 5)
    probabilities_log = kde.score_samples(std_failed.reshape(-1, 1))
    prediction_kde = [-1 if p < threshold else 1 for p in probabilities_log]
    probabilities = np.exp(probabilities_log)
    from sklearn.ensemble import IsolationForest

    threshold_sd = np.percentile(std_fine, 95)
    quintile = [-1 if sd > threshold_sd else 1 for sd in std_failed]

    # Fit Isolation Forest on passed data
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(std_fine.reshape(-1, 1))

    # Predict anomalies for failed data (-1 = anomaly, 1 = normal)
    predictions_iso_forest = clf.predict(std_failed.reshape(-1, 1)).tolist()
    print("Isolation forest prediction", predictions_iso_forest)

    from sklearn.svm import OneClassSVM

    clf = OneClassSVM(kernel="rbf", nu=0.05, gamma=0.1)
    clf.fit(std_fine.reshape(-1, 1))

    predictions_svm = clf.predict(std_failed.reshape(-1, 1)).tolist()  # -1 for anomaly, 1 for normal
    print("SVM predictions:", predictions_svm)

    predictions = {"quintile": quintile, "KDE": prediction_kde, "isolation_forest": predictions_iso_forest, "svm": predictions_svm}
    print("kde", prediction_kde)
    np.savetxt(path / "probabilities_failed.txt", probabilities)
    generate_report(
        path,
        software,
        template,
        stat,
        p,
        stat_fine,
        p_fine,
        p_shapiro,
        MAD_results["all_mad_fine"],
        MAD_results["all_mad_failed"],
        IDs["IDs_failed"],
        probabilities,
        predictions,
    )

    save_array(software, MAD_results, path)
    save_array(software, FD_mca_results, path)
    save_array(software, FD_ieee_results, path)

    record = {
        "index": f"{software} - {template}",
        "stat": stat,
        "p": p,
        "stat_fine": stat_fine,
        "p_fine": p_fine,
        "mean MAD passed": np.mean(MAD_results["all_mad_fine"]),
        "max MAD passed": np.max(MAD_results["all_mad_fine"]),
        "mean MAD failed": np.mean(MAD_results["all_mad_failed"]),
        "max MAD failed": np.max(MAD_results["all_mad_failed"]),
        "> 1 mm passed MAD": np.sum(MAD_results["all_mad_fine"] >= 1.0),
        "> 0.2 mm passed MAD": np.sum(MAD_results["all_mad_fine"] >= 0.2),
    }

    # from sklearn.neighbors import KernelDensity

    # kde = KernelDensity(kernel="gaussian", bandwidth=0.01).fit(np.std(FD_mca_results["FD_all_fine"], axis=1).reshape(-1, 1))
    # log_probs = kde.score_samples(np.std(FD_mca_results["FD_all_failed"], axis=1).reshape(-1, 1))
    # probs = np.exp(log_probs)
    # print(probs)

    # from sklearn.mixture import GaussianMixture

    # # Fit GMM on passed data
    # gmm = GaussianMixture(n_components=1, random_state=42)
    # gmm.fit(np.std(FD_mca_results["FD_all_fine"], axis=1).reshape(-1, 1))

    # # Calculate probabilities for failed data
    # probs = gmm.score_samples(np.std(FD_mca_results["FD_all_failed"], axis=1).reshape(-1, 1))
    # print(np.exp(probs))

    output_csv = path / "output.csv"
    df = pd.DataFrame([record])
    df.set_index("index", inplace=True)
    if output_csv.exists():
        df.to_csv(output_csv, mode="a", header=False)
    else:
        df.to_csv(output_csv)
    logging.info("Results saved to %s", output_csv)

    FD_all = np.concatenate([FD_mca_results["FD_all_fine"], FD_mca_results["FD_all_failed"]], axis=0)
    FD_all_std = np.std(FD_all, axis=1)

    FD_SD_name = f"FD_std_{software.upper()}_{template}_{cost_function}" if cost_function else f"FD_std_{software}_{template}"
    MAD_name = f"MAD_{software.upper()}_{template}_{cost_function}" if cost_function else f"MAD_{software}_{template}"
    QC_status_name = f"QC_status_{software.upper()}_{template}_{cost_function}" if cost_function else f"QC_status_{software}_{template}"
    df = pd.DataFrame({"Participant_ID": IDs["IDs_all"], FD_SD_name: FD_all_std, MAD_name: MAD_results["all_mad"]})
    df[QC_status_name] = ["fine" if id in IDs["IDs_fine"] else "failed" for id in IDs["IDs_all"]]
    df["cohort"] = ["HC" if id in mat_dic_HC.keys() else "PD" for id in IDs["IDs_all"]]
    df.to_csv(path / "FD_MAD.csv", index=False)
