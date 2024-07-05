import numpy as np
from lnrgst_mca.plot_utils import hist_plotter
from pathlib import Path


def read_all_versions(parent_path, softwares, templates, name_pattern):
    parent_path = Path(parent_path)
    data = {software: {template: None for template in templates} for software in softwares}

    for software in softwares:
        for template in templates:
            name = name_pattern(software, template)
            path = parent_path / software / template / name
            # "arrays" /
            if path.exists():
                data[software][template] = np.loadtxt(path)

    return data


# refactor these two later


def same_template_plots(templates, data, path):
    for template in templates:
        same_template_data_mean = [np.mean(data[software][template], axis=1) for software in data.keys()]
        same_template_data_std = [np.std(data[software][template], axis=1) for software in data.keys()]
        print(same_template_data_mean)
        hist_plotter(
            datasets=same_template_data_mean,
            title=f"Mean FD: different softwares - {template}",
            path=path,
            labels=list(data.keys()),
        )

        hist_plotter(
            datasets=same_template_data_std,
            title=f"SD of FD: different softwares - {template}",
            path=path,
            labels=list(data.keys()),
        )


def same_software_plots(templates, data, path):
    for software in data.keys():
        same_software_data_mean = [np.mean(data[software][template], axis=1) for template in templates]
        same_software_data_std = [np.std(data[software][template], axis=1) for template in templates]
        hist_plotter(
            datasets=same_software_data_mean,
            title=f"Mean FD: different templates - {software}",
            path=path,
            labels=templates,
        )

        hist_plotter(
            datasets=same_software_data_std,
            title=f"SD of FD: different templates - {software}",
            path=path,
            labels=templates,
        )


if __name__ == "__main__":

    # func = lambda t, s: f"{t}_FD_HC_fine.txt"
    def func(t, s):
        return f"{t}_FD_HC_fine.txt"

    parent_path = Path("/home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/outputs_plots/diagrams")
    data = read_all_versions(parent_path, ["spm", "flirt", "ants"], ["MNI152NLin2009cAsym_res-01"], func)
    same_template_plots(["MNI152NLin2009cAsym_res-01"], data, "./outputs_plots/between_plots")
