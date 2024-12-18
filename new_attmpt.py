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


def decompose_and_convert(mat_dict, n_mca=10, mode="mca"):
    scales, translations, angles, shears = transformation_dictionary_to_arrays(mat_dict, n_mca=n_mca, mode=mode)
    angles = np.degrees(angles)
    return scales, translations, angles, shears


def save_array(software, results, path):

    for key, value in results.items():
        np.savetxt(path / f"{software}_{key}.txt", value)


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
        f.write("------------------------------------------------------------\n")


if __name__ == "__main__":

    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template", type=str, help="Template name", required=True)
    parser.add_argument("-s", "--software", type=str, help="Software name", required=True)
    parser.add_argument("-d", "--diagram_path", type=str, help="Path to save output plots", required=True)
    parser.add_argument("-c", "--cost_function", type=str, help="Cost function", default=None)
    args = parser.parse_args()

    validate_arguments(args)

    template = args.template
    software = args.software
    cost_function = args.cost_functiom
    diagram_path = Path(args.diagram_path) / software / template
    if args.cost_function:
        diagram_path = diagram_path / cost_function
    diagram_path.mkdir(parents=True, exist_ok=True)

    # Load configurations
    failed_subjects_HC, failed_subjects_PD, mat_dic_PD, error_PD, mat_dic_HC, error_HC = get_configurations(
        args.software, args.template, args.cost_function
    )

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

    FD_ieee_results["FD_all_fine"] = concatenate_cohorts(FD_ieee_results["FD_ieee_PD_fine"], FD_ieee_results["FD_ieee_HC_fine"])
    FD_ieee_results["FD_all_failed"] = concatenate_cohorts(FD_ieee_results["FD_ieee_PD_failed"], FD_ieee_results["FD_ieee_HC_failed"])

    # calculating the mean absolute difference between MCA framewise displacements and the IEEE framewise displacement
    MAD_results = {
        "mad_fine_PD": metrics_utils.mean_absolute_difference(FD_mca_results["FD_fine_mca_PD"], FD_mca_results["FD_fine_ieee_PD"]),
        "mad_fine_HC": metrics_utils.mean_absolute_difference(FD_mca_results["FD_fine_mca_HC"], FD_mca_results["FD_fine_ieee_HC"]),
        "mad_failed_PD": metrics_utils.mean_absolute_difference(FD_mca_results["FD_failed_mca_PD"], FD_mca_results["FD_fine_ieee_PD"]),
        "mad_failed_HC": metrics_utils.mean_absolute_difference(FD_mca_results["FD_failed_mca_HC"], FD_mca_results["FD_fine_ieee_HC"]),
    }

    MAD_results["all_mad_failed"] = concatenate_cohorts(MAD_results["mad_failed_PD"], MAD_results["mad_failed_HC"])
    MAD_results["all_mad_fine"] = concatenate_cohorts(MAD_results["mad_fine_PD"], MAD_results["mad_fine_HC"])

    # "mad_all_PD": metrics_utils.mean_absolute_difference(FD_mca_results[result_all_PD, FD_mca_results[result_all_ieee_PD),
    # "mad_all_HC": metrics_utils.mean_absolute_difference(FD_mca_results[result_all_HC, FD_mca_results[result_all_ieee_HC)}

    # plotting mean and standard deviation for 12 parameters

    basic_info_plotter(
        translations_mca_fine_PD,
        translations_mca_fine_HC,
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
        translations_mca_failed_PD,
        translations_mca_failed_HC,
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

    # plotting framewise displacment
    basic_info_plotter(
        FD_mca_results["FD_PD_fine"],
        FD_mca_results["FD_HC_fine"],
        software=software,
        variable="Framewise Displacement",
        path=diagram_path,
        figure_size=(5, 4),
        ylable="(mm)",
        y_lim_mean=[(10, 110)],
        y_lim_sd=[(0, 0.3)],
    )

    basic_info_plotter(
        FD_mca_results["FD_failed_mca_PD"],
        FD_mca_results["FD_failed_mca_HC"],
        software=software,
        variable="Framewise Displacement failed",
        path=diagram_path,
        figure_size=(5, 4),
        ylable="(mm)",
    )
    basic_info_plotter(
        FD_mca_results["FD_all_PD"],
        FD_mca_results["FD_all_HC"],
        software=software,
        variable="Framewise Displacement All",
        path=diagram_path,
        figure_size=(5, 4),
        ylable="(mm)",
    )

    hist_plotter(
        datasets=[np.mean(FD_mca_results["FD_all_fine"], axis=1), np.mean(FD_mca_results["FD_all_failed"], axis=1)],
        title=f"Mean FD: {software} - {template}",
        path=diagram_path,
        bins=FD_mean_bin_sizes[software][template],
        labels=["Passed", "failed"],
        xlabel="(mm)",
    )
    hist_plotter(
        datasets=[np.std(FD_mca_results["FD_all_fine"], axis=1), np.std(FD_mca_results["FD_all_failed"], axis=1)],
        title=f"SD of  FD: {software} - {template}",
        path=diagram_path,
        bins=FD_sd_bin_sizes[software][template],
        labels=["Passed", "failed"],
        xlim=FD_SD_x_lim[software][template],
        xlabel="(mm)",
    )

    hist_plotter(
        datasets=[MAD_results["all_mad_fine"], MAD_results["all_mad_failed"]],
        title=f"Mean Absolute Difference of FD: {software} - {template}",
        path=diagram_path,
        labels=["Passed", "Failed"],
        xlabel="(mm)",
    )

    hist_plotter(
        datasets=[np.log(np.std(FD_mca_results["FD_all_fine"], axis=1)), np.log(np.std(FD_mca_results["FD_all_failed"], axis=1))],
        title=f"log SD of FD: {software} - {template}",
        path=diagram_path,
        labels=["Passed", "Failed"],
        xlabel="log(value) (mm)",
    )

    # saving
    path = diagram_path / "reports"
    # Save IDs
    IDs = {
        "IDs_fine": concatenate_cohorts(np.array(list(mat_dic_fine_PD.keys())), np.array(list(mat_dic_fine_HC.keys()))),
        "IDs_failed": concatenate_cohorts(np.array(list(mat_dic_failed_PD.keys())), np.array(list(mat_dic_failed_HC.keys()))),
    }
    IDs["IDs_all"] = concatenate_cohorts(IDs["IDs_fine"], IDs["IDs_failed"])
    save_array(software, IDs, path)
    # t test on standard deviation of cohorts
    t, p = stats.ttest_ind(np.log(np.std(FD_mca_results["FD_all_PD"], axis=1)), np.log(np.std(FD_mca_results["FD_all_HC"], axis=1)))
    t_fine, p_fine = stats.ttest_ind(
        np.log(np.std(FD_mca_results["FD_fine_mca_PD"], axis=1)), np.log(np.std(FD_mca_results["FD_fine_mca_HC"], axis=1))
    )

    #
    fine_mean_of_std = np.mean(np.log(np.std(FD_mca_results["FD_all_fine"], axis=1)))
    fine_std_of_std = np.std(np.log(np.std(FD_mca_results["FD_all_fine"], axis=1)))
    probabilities = stats.norm.pdf(np.log(np.std(FD_mca_results["FD_all_failed"], axis=1)), fine_mean_of_std, fine_std_of_std)

    np.savetxt(path / "probabilities_failed.txt", probabilities)
    generate_report(
        path, software, template, t, p, t_fine, p_fine, MAD_results["all_mad_fine"], MAD_results["all_mad_failed"], IDs["IDs_failed"], probabilities
    )

    save_array(software, MAD_results, path)
    save_array(software, FD_mca_results, path)
    save_array(software, FD_ieee_results, path)

    record = {
        "index": f"{software} - {template}",
        "t": t,
        "p": p,
        "t_fine": t_fine,
        "p_fine": p_fine,
        "mean MAD passed": np.mean(MAD_results["all_mad_fine"]),
        "max MAD passed": np.max(MAD_results["all_mad_fine"]),
        "mean MAD failed": np.mean(MAD_results["all_mad_failed"]),
        "max MAD failed": np.max(MAD_results["all_mad_failed"]),
        "> 1 mm passed MAD": np.sum(MAD_results["all_mad_fine"] >= 1.0),
        "> 0.2 mm passed MAD": np.sum(MAD_results["all_mad_fine"] >= 0.2),
    }

    output_csv = path / "output.csv"
    df = pd.DataFrame([record])
    df.set_index("index", inplace=True)
    if output_csv.exists():
        df.to_csv(output_csv, mode="a", header=False)
    else:
        df.to_csv(output_csv)
    logging.info("Results saved to %s", output_csv)
