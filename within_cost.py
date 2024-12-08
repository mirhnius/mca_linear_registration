import logging
from pathlib import Path
from scipy import stats
import pandas as pd
import numpy as np
import argparse
from lnrgst_mca import metrics_utils
from lnrgst_mca.plot_utils import hist_plotter
from config import get_configurations, FD_mean_bin_sizes, FD_sd_bin_sizes, FD_SD_x_lim
from within_software_analysis import (
    transformation_dictionary_to_arrays,
    basic_info_plotter,
    copy_and_remove_keys,
    concatenate_cohorts,
)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def validate_arguments(args):
    if args.software not in FD_mean_bin_sizes or args.software not in FD_sd_bin_sizes:
        raise ValueError(f"Unknown software: {args.software}")

    if args.template not in FD_mean_bin_sizes[args.software].keys() or args.template not in FD_sd_bin_sizes[args.software].keys():
        raise ValueError(f"Unknown template: {args.template}")

def decompose_and_convert(mat_dict, n_mca=1, mode="ieee"):
    scales, translations, angles, shears = transformation_dictionary_to_arrays(mat_dict, n_mca=n_mca, mode=mode)
    angles = np.degrees(angles)
    return scales, translations, angles, shears

def plot_parameters(data_PD, data_HC, variable, software, path, axis_labels, y_lim_mean=None, y_lim_sd=None, ylable=""):
    basic_info_plotter(
        data_PD,
        data_HC,
        software=software,
        variable=variable,
        path=path,
        axis_labels=axis_labels,
        ylable=ylable,
        y_lim_mean=y_lim_mean,
        y_lim_sd=y_lim_sd,
    )

def save_results(result_dict, path, prefix):
    for key, result in result_dict.items():
        np.savetxt(path / f"{prefix}_{key}.txt", result)

def calculate_statistics_and_plot(result_PD, result_HC, software, template, diagram_path):
    t, p = stats.ttest_ind(np.log(np.std(result_PD, axis=1)), np.log(np.std(result_HC, axis=1)))
    logging.info("T-test results: t=%.3f, p=%.3f", t, p)

    hist_plotter(
        datasets=[np.log(np.std(result_PD, axis=1)), np.log(np.std(result_HC, axis=1))],
        title=f"Log SD of FD: {software} - {template}",
        path=diagram_path,
        labels=["PD", "HC"],
        xlabel="Log(value) (mm)",
    )

def calculate_framewise_displacement(translations, angles):
    return metrics_utils.FD_all_subjects(translations, angles)

def save_displacement_and_mad(results, ieee_results, mad_results, prefix, path):
    """
    Save framewise displacement (FD) and mean absolute difference (MAD) results to files.

    Args:
        results (dict): FD results for different subject groups.
        ieee_results (dict): IEEE-based FD results for comparison.
        mad_results (dict): MAD results between FD and IEEE-based FD.
        prefix (str): Prefix for file names (e.g., software name).
        path (Path): Directory to save the output files.
    """
    # Save FD results
    for key, value in results.items():
        np.savetxt(path / f"{prefix}_FD_{key}.txt", value)
    # Save IEEE FD results
    for key, value in ieee_results.items():
        np.savetxt(path / f"{prefix}_FD_ieee_{key}.txt", value)
    # Save MAD results
    for key, value in mad_results.items():
        np.savetxt(path / f"{prefix}_MAD_{key}.txt", value)

def calculate_mean_absolute_difference(results, ieee_results):
    mad_results = {}
    for key in results:
        mad_results[key] = metrics_utils.mean_absolute_difference(results[key], ieee_results[key])
    return mad_results

def generate_report(path, software, template, t, p, t_fine, p_fine, all_mad_fine, all_mad_failed, IDs_failed, probabilities):
    with open(path / "report.txt", "w") as f:
        f.write("-------------- {} - {} --------------\n".format(software, template))
        f.write("HC vs PD t-test on standard deviation of framewise displacement for all subjects  t:{:.3f} p:{:.3f}\n".format(t, p))
        f.write("HC vs PD t-test on standard deviation of framewise displacement for passed QC subjects  t:{:.3f} p:{:.3f}\n".format(t_fine, p_fine))
        f.write("Mean of Mean Absolute Difference of passed QC subjects: {:.3e} mm\n".format(np.mean(all_mad_fine)))
        f.write("Mean of Mean Absolute Difference of failed QC subjects: {:.3e} mm\n".format(np.mean(all_mad_failed)))
        f.write("Max of Mean Absolute Difference of passed QC subjects: {:.3e} mm\n".format(np.max(all_mad_fine)))
        f.write("Max of Mean Absolute Difference of failed QC subjects: {:.3e} mm\n".format(np.max(all_mad_failed)))
        f.write("Number of passed QC subjects with higher than 1 mm Mean Absolute Difference: {}\n".format(np.sum(all_mad_fine >= 1.0)))
        f.write("Number of passed QC subjects with higher than 0.2 mm Mean Absolute Difference: {}\n".format(np.sum(all_mad_fine >= 0.2)))
        f.write("------------------------------------------------------------\n")
        f.write("\n")
        f.write("The probabilities of failed subjects belonging to the fine distribution:\n")
        for i, id in enumerate(IDs_failed):
            f.write("{}: {:.3e}\n".format(id, probabilities[i]))

def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template", type=str, help="Template name", required=True)
    parser.add_argument("-s", "--software", type=str, help="Software name", required=True)
    parser.add_argument("-d", "--diagram_path", type=str, help="Path to save output plots", required=True)
    parser.add_argument("-c", "--cost_function", type=str, help="Cost function", default=None)
    args = parser.parse_args()

    validate_arguments(args)

    diagram_path = Path(args.diagram_path) / args.software / args.template
    if args.cost_function:
        diagram_path = Path(args.diagram_path) / args.cost_function / args.software / args.template
    diagram_path.mkdir(parents=True, exist_ok=True)

    # Load configurations
    failed_subjects_HC, failed_subjects_PD, mat_dic_PD, error_PD, mat_dic_HC, error_HC = get_configurations(
        args.software, args.template, args.cost_function
    )

    if error_PD or error_HC:
        logging.error("Error in loading MCA matrices: %s, %s", error_PD, error_HC)
        raise Exception("There is an issue with mat files")

    mat_dic_fine_PD, mat_dic_failed_PD = copy_and_remove_keys(mat_dic_PD, failed_subjects_PD)
    mat_dic_fine_HC, mat_dic_failed_HC = copy_and_remove_keys(mat_dic_HC, failed_subjects_HC)

    # Decompose matrices
    scales_mca_fine_PD, translations_mca_fine_PD, angles_mca_fine_PD, shears_mca_fine_PD = decompose_and_convert(mat_dic_fine_PD)
    scales_mca_fine_HC, translations_mca_fine_HC, angles_mca_fine_HC, shears_mca_fine_HC = decompose_and_convert(mat_dic_fine_HC)

    scales_mca_failed_PD, translations_mca_failed_PD, angles_mca_failed_PD, shears_mca_failed_PD = decompose_and_convert(mat_dic_failed_PD)
    scales_mca_failed_HC, translations_mca_failed_HC, angles_mca_failed_HC, shears_mca_failed_HC = decompose_and_convert(mat_dic_failed_HC)

    scales_ieee_PD, translations_ieee_PD, angles_ieee_PD, shears_ieee_PD = decompose_and_convert(mat_dic_PD, mode="ieee")
    scales_ieee_HC, translations_ieee_HC, angles_ieee_HC, shears_ieee_HC = decompose_and_convert(mat_dic_HC, mode="ieee")

    # Calculate framewise displacement
    results = {
        "fine_PD": calculate_framewise_displacement(translations_mca_fine_PD, angles_mca_fine_PD),
        "fine_HC": calculate_framewise_displacement(translations_mca_fine_HC, angles_mca_fine_HC),
        "failed_PD": calculate_framewise_displacement(translations_mca_failed_PD, angles_mca_failed_PD),
        "failed_HC": calculate_framewise_displacement(translations_mca_failed_HC, angles_mca_failed_HC),
    }

    ieee_results = {
        "fine_PD": calculate_framewise_displacement(translations_ieee_PD[:, np.newaxis, :], angles_ieee_PD[:, np.newaxis, :]),
        "fine_HC": calculate_framewise_displacement(translations_ieee_HC[:, np.newaxis, :], angles_ieee_HC[:, np.newaxis, :]),
    }

    # Calculate mean absolute difference
    mad_results = calculate_mean_absolute_difference(results, ieee_results)

    # Save FD, IEEE FD, and MAD results
    save_displacement_and_mad(results, ieee_results, mad_results, args.software, diagram_path)

    # Perform t-tests
    t, p = stats.ttest_ind(np.log(np.std(results["fine_PD"], axis=1)), np.log(np.std(results["fine_HC"], axis=1)))
    t_fine, p_fine = stats.ttest_ind(np.log(np.std(results["fine_PD"], axis=1)), np.log(np.std(results["fine_HC"], axis=1)))

    all_mad_fine = concatenate_cohorts(mad_results["fine_PD"], mad_results["fine_HC"])
    all_mad_failed = concatenate_cohorts(mad_results["failed_PD"], mad_results["failed_HC"])

    probabilities = stats.norm.pdf(
        np.log(np.std(results["failed_PD"], axis=1)),
        np.mean(np.log(np.std(results["fine_PD"], axis=1))),
        np.std(np.log(np.std(results["fine_PD"], axis=1)))
    )

    IDs_failed = concatenate_cohorts(
        np.array(list(mat_dic_failed_PD.keys())),
        np.array(list(mat_dic_failed_HC.keys()))
    )

    # Generate report
    generate_report(
        diagram_path, args.software, args.template, t, p, t_fine, p_fine, all_mad_fine, all_mad_failed, IDs_failed, probabilities
    )

    logging.info("Processing completed successfully!")

if __name__ == "__main__":
    main()
