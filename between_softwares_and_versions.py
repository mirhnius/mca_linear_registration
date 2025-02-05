import numpy as np
import copy
from lnrgst_mca.plot_utils import hist_plotter, plotter
from pathlib import Path

from itertools import combinations
from scipy import stats

from lnrgst_mca.constants import ANTS, FLIRT, SPM
from config import palette_colors

MNI2009cAsym = "MNI152NLin2009cAsym_res-01"
MNI2009cSym = "MNI152NLin2009cSym_res-1"
sd_same_software_bin_sizes = {
    FLIRT: [10, 10],
    ANTS: [25, 25],
    SPM: [50, 3],
}
mean_same_software_bin_sizes = {
    FLIRT: [50, 10],
    ANTS: [10, 10],
    SPM: [10, 10],
}


def read_all_versions(parent_path, softwares, templates, name_patterns, dtype=float):
    parent_path = Path(parent_path)
    data = {software: {template: None for template in templates} for software in softwares}

    for software in softwares:
        for template in templates:
            data_list = []
            for name_pattern in name_patterns:
                name = name_pattern(software, template)
                path = parent_path / software / template / name
                # "arrays" /
                if path.exists():
                    data_list.append(np.loadtxt(path, dtype=dtype))
            if data_list:
                data[software][template] = np.concatenate(data_list)

    return data


# refactor these two later


# support data[][] == None
def same_template_plots(templates, data, path, coordinates=None, std_bins=None, mean_bins=None, log=False, **kawrgs):

    if coordinates is None:
        coordinates_mean, coordinates_std = None, None
    else:
        coordinates_mean, coordinates_std = coordinates[0], coordinates[1]
    for template in templates:

        if log:
            same_template_data_mean = [np.log10(np.mean(data[software][template], axis=1)) for software in data.keys()]
            same_template_data_std = [np.log10(np.std(data[software][template], axis=1)) for software in data.keys()]
        else:
            same_template_data_mean = [np.mean(data[software][template], axis=1) for software in data.keys()]
            same_template_data_std = [np.std(data[software][template], axis=1) for software in data.keys()]

        hist_plotter(
            datasets=same_template_data_mean,
            title=f"Mean FD: different softwares - {template}",
            path=path,
            labels=list(data.keys()),
            coordinates=coordinates_mean,
            bins=mean_bins,
            **kawrgs,
        )

        hist_plotter(
            datasets=same_template_data_std,
            title=f"SD of FD: different softwares - {template}",
            path=path,
            labels=list(data.keys()),
            coordinates=coordinates_std,
            bins=std_bins,
            **kawrgs,
        )


# def same_software_plots(templates, data, path, labels=None, std_bins=None, mean_bins=None, log=False, **kawrgs):
# if labels is None:
#     labels = templates

# for software in data.keys():
#     if log:
#         same_software_data_mean = [np.log10(np.mean(data[software][template], axis=1)) for template in templates]
#         same_software_data_std = [np.log10(np.std(data[software][template], axis=1)) for template in templates]
#     else:
#         same_software_data_mean = [np.mean(data[software][template], axis=1) for template in templates]
#         same_software_data_std = [np.std(data[software][template], axis=1) for template in templates]

#     m_b = mean_bins[software] if mean_bins else None
#     s_b = std_bins[software] if std_bins else None
#     hist_plotter(
#         datasets=same_software_data_mean, title=f"Mean FD: different templates - {software}", path=path, labels=labels, bins=m_b, **kawrgs
#     )

#     hist_plotter(
#         datasets=same_software_data_std, title=f"SD of FD: different templates - {software}", path=path, labels=labels, bins=s_b, **kawrgs
#     )


def same_software_plots(templates, data, path, labels=None):
    if labels is None:
        labels = templates

    for software in data.keys():

        same_software_data_mean = [np.log10(np.mean(data[software][template], axis=1)) for template in templates]
        same_software_data_std = [np.log10(np.std(data[software][template], axis=1)) for template in templates]
        print(type(same_software_data_mean))
        print(same_software_data_mean)

        plotter(
            same_software_data_mean[0],
            same_software_data_mean[1],
            title=f"Mean FD: different templates - {software}",
            path=path,
            labels=labels,
            ylable="log10(value)(mm)",
            adjust=True,
            palette=[palette_colors[software][templates[0]], palette_colors[software][templates[1]]],
        )

        plotter(
            same_software_data_std[0],
            same_software_data_std[1],
            title=f"Standard Deviation of FD: Template Comparison - {software}",
            path=path,
            labels=labels,
            ylable="log10(value) (mm)",
            adjust=True,
            palette=[palette_colors[software][templates[0]], palette_colors[software][templates[1]]],
        )


def remove_failed_ids_by_software(d, parent_path, fine_IDs):

    data = copy.deepcopy(d)
    IDs_to_remove = {}
    for s in data.keys():
        templates = list(data[s].keys())
        failed_ids_set = set()
        for t in templates:
            path = parent_path / s / t / "IDs_failed_all.txt"
            if path.exists():
                failed_ids_set.update(np.loadtxt(path, dtype=str))
        IDs_to_remove[s] = failed_ids_set

    for s in data.keys():
        templates = list(data[s].keys())
        for t in templates:
            if IDs_to_remove is None or data[s][t] is None:
                continue
            mask = np.isin(fine_IDs[s][t], list(IDs_to_remove[s]))
            indices = np.where(mask)[0]
            data[s][t] = np.delete(data[s][t], indices, axis=0)

    return data


def remove_failed_ids_by_template(d, parent_path, fine_IDs, all_templates, all_softwares):

    data = copy.deepcopy(d)
    IDs_to_remove = {}
    for t in all_templates:
        failed_ids_set = set()
        for s in all_softwares:
            path = parent_path / s / t / "IDs_failed_all.txt"
            if path.exists():
                failed_ids_set.update(np.loadtxt(path, dtype=str))
        IDs_to_remove[t] = failed_ids_set

    for t in all_templates:
        for s in all_softwares:
            if IDs_to_remove is None or data[s][t] is None:
                continue
            mask = np.isin(fine_IDs[s][t], list(IDs_to_remove[t]))
            indices = np.where(mask)[0]
            data[s][t] = np.delete(data[s][t], indices, axis=0)

    return data


def perform_t_tests_same_software(data, all_templates, all_softwares):

    combos = list(combinations(range(len(all_templates)), 2))
    results = []

    for comb in combos:
        for s in all_softwares:
            g1, g2 = data[s][all_templates[comb[0]]], data[s][all_templates[comb[1]]]
            if g1 is None or g2 is None:
                continue

            t, p = stats.ttest_rel(np.log(np.std(g1, axis=1)), np.log(np.std(g2, axis=1)))
            results.append((s, all_templates[comb[0]], all_templates[comb[1]], t, p))

    return results


def perform_t_tests_same_templates(data, all_templates, all_softwares):

    combos = list(combinations(range(len(all_softwares)), 2))
    results = []

    for comb in combos:
        for t in all_templates:
            g1, g2 = data[all_softwares[comb[0]]][t], data[all_softwares[comb[1]]][t]
            if g1 is None or g2 is None:
                continue

            t_, p = stats.ttest_rel(np.log(np.std(g1, axis=1)), np.log(np.std(g2, axis=1)))
            results.append((t, all_softwares[comb[0]], all_softwares[comb[1]], t_, p))

    return results


def print_t_test_results_for_same_software(results):
    print("**********************************************")
    for result in results:
        s, t1, t2, t, p = result
        print(f"software:{s} between templates: {t1} - {t2}")
        print(f"t:{t:.2e} p:{p:.2e}")
    print("**********************************************")


def print_t_test_results_for_same_template(results):
    print("**********************************************")
    for result in results:
        t, s1, s2, t_, p = result
        print(f"templates:{t} between softwares: {s1} - {s2}")
        print(f"t:{t_:.2e} p:{p:.2e}")
    print("**********************************************")


if __name__ == "__main__":

    def func_HC(t, s):
        return f"{t}_FD_HC_fine.txt"

    def func_PD(t, s):
        return f"{t}_FD_PD_fine.txt"

    parent_path = Path("/home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/outputs_plots/diagrams")
    softwares = ["spm", "flirt", "ants"]
    templates = ["MNI152NLin2009cAsym_res-01", "MNI152NLin2009cSym_res-1"]
    data = read_all_versions(parent_path, softwares, templates, [func_HC, func_PD])

    same_template_plots(
        ["MNI152NLin2009cAsym_res-01"],
        data,
        Path("./outputs_plots/between_plots"),
        coordinates=[[200, 40], [7, 40]],
        std_bins=[1, 1, 25],
        mean_bins=[3, 15, 2],
        dists=[5, 10, 15],
    )

    # same_template_plots(
    #     ["MNI152NLin2009cSym_res-1"],
    #     data,
    #     Path("./outputs_plots/between_plots"),
    #     coordinates=[[80, 10], [7, 40]],
    #     std_bins=[1, 1, 25],
    #     mean_bins=[5, 5, 5],
    #     dists=[5, 10, 15],
    # )

    # same_software_plots(
    #     ["MNI152NLin2009cAsym_res-01", "MNI152NLin2009cSym_res-1"],
    #     data,
    #     Path("./outputs_plots/between_templates"),
    #     labels=["Asym", "Sym"],
    #     dists=[-10, -10],
    #     mean_bins=mean_same_software_bin_sizes,
    #     std_bins=sd_same_software_bin_sizes,
    # )

    # same_template_plots(
    #     ["MNI152NLin2009cSym_res-1"],
    #     data,
    #     Path("./outputs_plots/between_plots"),
    #     coordinates=[[80, 10], [7, 40]],
    #     std_bins=[1, 1, 25],
    #     mean_bins=[5, 5, 5],
    #     dists=[5, 10, 15],
    #     log=True,
    # )

    same_software_plots(
        templates=["MNI152NLin2009cAsym_res-01", "MNI152NLin2009cSym_res-1"],
        data=data,
        path=Path("./outputs_plots/between_templates"),
        labels=["Asym", "Sym"],
        # coordinates=[80, 10],
        # log=False,
        # xlabel="log10(Value) (mm)",
    )

    # def func_fd_passed(s, t):
    #     return "FD_fine_all.txt"

    # data = read_all_versions(parent_path, softwares, templates, [func_fd_passed], dtype=float)

    # def func_fd_passed_IDs(s, t):
    #     return "IDs_fine_all.txt"

    # fine_IDs = read_all_versions(parent_path, softwares, templates, [func_fd_passed_IDs], str)

    # data_for_software_comparison = remove_failed_ids_by_template(data, parent_path, fine_IDs, templates, softwares)
    # same_template_results = perform_t_tests_same_templates(data_for_software_comparison, templates, softwares)
    # print_t_test_results_for_same_template(same_template_results)

    # data_for_template_comparison = remove_failed_ids_by_software(data, parent_path, fine_IDs)
    # same_software_results = perform_t_tests_same_software(data_for_template_comparison, templates, softwares)
    # print_t_test_results_for_same_software(same_software_results)

    # def func_fd_all(s, t):
    #     return "FD_all.txt"

    # data = read_all_versions(parent_path, softwares, templates, [func_fd_all], dtype=float)
    # same_template_results = perform_t_tests_same_templates(data, templates, softwares)
    # print_t_test_results_for_same_template(same_template_results)
    # same_software_results = perform_t_tests_same_software(data, templates, softwares)
    # print_t_test_results_for_same_software(same_software_results)

    # def func_fd_passed(s, t):
    #     return "FD_fine_all.txt"

    # data = read_all_versions(parent_path, softwares, templates, [func_fd_passed], dtype=float)

    # def func_fd_passed_IDs(s, t):
    #     return "IDs_fine_all.txt"

    # fine_IDs = read_all_versions(parent_path, softwares, templates, [func_fd_passed_IDs], str)

    # data_for_software_comparison = remove_failed_ids_by_template(data, parent_path, fine_IDs, templates, softwares)
    # same_template_results = perform_t_tests_same_templates(data_for_software_comparison, templates, softwares)
    # print_t_test_results_for_same_template(same_template_results)

    # data = data_for_software_comparison
    # sd_data = {software: {template: np.std(array, axis=1) for template, array in templates.items()} for software, templates in data.items()}

    # from scipy.stats import f_oneway  # shapiro, levene,

    # anova_stat, anova_p = f_oneway(
    #     sd_data["ants"]["MNI152NLin2009cAsym_res-01"], sd_data["flirt"]["MNI152NLin2009cAsym_res-01"], sd_data["spm"]["MNI152NLin2009cAsym_res-01"]
    # )
    # anova_stat, anova_p = f_oneway(
    #     sd_data["ants"]["MNI152NLin2009cSym_res-1"], sd_data["flirt"]["MNI152NLin2009cSym_res-1"], sd_data["spm"]["MNI152NLin2009cSym_res-1"]
    # )

    # # from statsmodels.stats.anova import anova_lm
    # import pandas as pd

    # # from statsmodels.formula.api import ols

    # # sd_data = [
    # # {"software": software, "template": template, "sd": np.std(array, axis=1)}
    # # for software, templates in data.items()
    # # for template, array in templates.items()

    # # ]
    # # Step 1: Flatten the nested dictionary
    # flattened_data = []
    # subject_id = 1  # Unique identifier for each subject
    # for software, templates in sd_data.items():
    #     for template, sd_array in templates.items():
    #         subject_id = 1
    #         for sd in sd_array:
    #             flattened_data.append({"subject": subject_id, "software": software, "template": template, "sd": sd})
    #             subject_id += 1  # Increment subject ID for each SD value

    # # Step 2: Create a DataFrame
    # df = pd.DataFrame(flattened_data)
    # # df = pd.DataFrame(sd_data)

    # # Step 2: Perform two-way ANOVA
    # # Use `ols` to fit the model: "sd ~ C(software) + C(template) + C(software):C(template)"
    # # model = ols('sd ~ C(software) + C(template) + C(software):C(template)', data=df).fit()

    # # # Perform ANOVA
    # # anova_results = anova_lm(model)

    # import pingouin as pg

    # aov = pg.rm_anova(dv="sd", within=["software", "template"], subject="subject", data=df, detailed=True)

    # # Display the results
    # print(aov)
    # aov.to_csv("rm_anova_results.csv", index=False)

    # # from scipy.stats import kruskal  # for different subjects!!

    # # print(kruskal(sd_data['ants']['MNI152NLin2009cAsym_res-01'],
    # # sd_data['flirt']['MNI152NLin2009cAsym_res-01'], sd_data['spm']['MNI152NLin2009cAsym_res-01']))
    # # print(kruskal(sd_data['ants']['MNI152NLin2009cSym_res-1'],
    # #  sd_data['flirt']['MNI152NLin2009cSym_res-1'], sd_data['spm']['MNI152NLin2009cSym_res-1']))

    # from scipy.stats import friedmanchisquare

    # print(
    #     friedmanchisquare(
    #         sd_data["ants"]["MNI152NLin2009cAsym_res-01"],
    #         sd_data["flirt"]["MNI152NLin2009cAsym_res-01"],
    #         sd_data["spm"]["MNI152NLin2009cAsym_res-01"],
    #     )
    # )
    # print(
    #     friedmanchisquare(
    #         sd_data["ants"]["MNI152NLin2009cSym_res-1"], sd_data["flirt"]["MNI152NLin2009cSym_res-1"], sd_data["spm"]["MNI152NLin2009cSym_res-1"]
    #     )
    # )
