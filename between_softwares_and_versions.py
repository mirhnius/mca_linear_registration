import numpy as np
from lnrgst_mca.plot_utils import hist_plotter
from pathlib import Path

# from itertools import combinations
# from scipy import stats


def read_all_versions(parent_path, softwares, templates, name_patterns, dtype=int):
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
                    data_list.append(np.genfromtxt(path, dtype))
            if data_list:
                data[software][template] = np.concatenate(data_list)

    return data


# refactor these two later


# support data[][] == None
def same_template_plots(templates, data, path, coordinates=None, std_bins=None, mean_bins=None, **kawrgs):

    if coordinates is None:
        coordinates_mean, coordinates_std = None, None
    else:
        coordinates_mean, coordinates_std = coordinates[0], coordinates[1]
    for template in templates:
        same_template_data_mean = [np.mean(data[software][template], axis=1) for software in data.keys()]
        same_template_data_std = [np.std(data[software][template], axis=1) for software in data.keys()]
        print(same_template_data_mean)
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


def same_software_plots(templates, data, path, labels=None, std_bins=None, mean_bins=None, **kawrgs):
    if labels is None:
        labels = templates

    for software in data.keys():
        same_software_data_mean = [np.mean(data[software][template], axis=1) for template in templates]
        same_software_data_std = [np.std(data[software][template], axis=1) for template in templates]
        hist_plotter(
            datasets=same_software_data_mean, title=f"Mean FD: different templates - {software}", path=path, labels=labels, bins=mean_bins, **kawrgs
        )

        hist_plotter(
            datasets=same_software_data_std, title=f"SD of FD: different templates - {software}", path=path, labels=labels, bins=std_bins, **kawrgs
        )


if __name__ == "__main__":

    # func = lambda t, s: f"{t}_FD_HC_fine.txt"
    def func_HC(t, s):
        return f"{t}_FD_HC_fine.txt"

    def func_PD(t, s):
        return f"{t}_FD_PD_fine.txt"

    parent_path = Path("/home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/outputs_plots/diagrams")
    data = read_all_versions(parent_path, ["spm", "flirt", "ants"], ["MNI152NLin2009cAsym_res-01"], [func_HC, func_PD])
    # data = read_all_versions(parent_path, ["spm", "flirt", "ants"], ["MNI152NLin2009cAsym_res-01", "MNI152NLin2009cSym_res-1"], [func_HC, func_PD])
    # same_template_plots(
    #     ["MNI152NLin2009cAsym_res-01"],
    #     data,
    #     Path("./outputs_plots/between_plots"),
    #     coordinates=[[200, 40], [7, 40]],
    #     std_bins=[1, 1, 25],
    #     mean_bins=[3, 15, 2],
    #     dists=[5, 10, 15],
    # )
    # same_template_plots(
    #     ["MNI152NLin2009cSym_res-1"],
    #     data,
    #     Path("./outputs_plots/between_plots"),
    #     coordinates=[[80, 10], [7, 40]],
    #     std_bins=[1, 1, 25],
    #     mean_bins=[5, 5, 5],
    #     dists=[5, 10, 15],
    # )  # dists=[-23, -5, 10]

    # same_software_plots(["MNI152NLin2009cAsym_res-01",
    # "MNI152NLin2009cSym_res-1"],data, Path("./outputs_plots/between_templates"),  labels=["2009cAsym", "2009cSym"], dists=[-10,-10])
    # def func_fd_passed(s, t):
    #     return "FD_fine_all.txt"

    # data = read_all_versions(parent_path, ["spm", "flirt", "ants"], ["MNI152NLin2009cAsym_res-01", "MNI152NLin2009cSym_res-1"], [func_fd_passed])

    # def func_fd_passed_IDs(s, t):
    #     return "IDs_fine_all.txt"

    # fine_IDs = read_all_versions(
    #     parent_path, ["spm", "flirt", "ants"], ["MNI152NLin2009cAsym_res-01", "MNI152NLin2009cSym_res-1"], [func_fd_passed_IDs], str
    # )
    # IDs_to_remove = {}
    # for s in data.keys():
    #     templates = list(data[s].keys())
    #     failed_ids_set = set()
    #     for t in templates:
    #         path = parent_path / s / t / "IDs_failed_all.txt"
    #         if path.exists():
    #             failed_ids_set.update(np.loadtxt(path, dtype=str))
    #     IDs_to_remove[s] = failed_ids_set

    # for s in data.keys():
    #     templates = list(data[s].keys())
    #     for t in templates:
    #         if IDs_to_remove is None or data[s][t] is None:
    #             continue
    #         mask = np.isin(fine_IDs[s][t], IDs_to_remove)
    #         indices = np.where(mask)[0]
    #         data[s][t] = np.delete(data[s][t], indices)

    # all_templates = ["MNI152NLin2009cAsym_res-01", "MNI152NLin2009cSym_res-1"]
    # combos = list(combinations(range(len(all_templates)), 2))

    # for comb in combos:
    #     for s in data.keys():
    #         g1, g2 = data[s][all_templates[comb[0]]], data[s][all_templates[comb[1]]]
    #         if g1 is None or g2 is None:
    #             continue

    #         t, p = stats.ttest_rel(g1, g2)
    #         print(f"software:{s} between templates: {templates[comb[0]]} - {templates[comb[1]]}")

    # all_softwares = ["spm", "flirt", "ants"]
    # combos = list(combinations(range(len(all_softwares)), 2))

    # for comb in combos:
    #     for t in templates:
    #         g1, g2 = data[all_softwares[comb[0]]][t], data[all_softwares[comb[1]]][t]
    #         if g1 is None or g2 is None:
    #             continue

    #         t, p = stats.ttest_rel(g1, g2)
    #         print(f"templates:{t} between softwares: {all_softwares[comb[0]]} - {all_softwares[comb[1]]}")
