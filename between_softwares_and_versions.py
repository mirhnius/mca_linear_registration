import numpy as np
from lnrgst_mca.plot_utils import hist_plotter


def read_all_versions(parent_path, softwares, templates, name_pattern):
    for software, template in zip(softwares, templates):
        data = {software: {template: None for template in templates} for software in softwares}
        name = name_pattern(software, template)
        path = parent_path / software / template / "arrays" / name
        if not path.exists():
            continue
        data[software][template] = np.loadtxt(path)

    return data


def same_template_plots(templates, data, path):
    for template in templates:
        same_template_data_mean = [np.mean(data[software][template], axis=1) for software in data.keys()]
        same_template_data_mean = [np.std(data[software][template], axis=1) for software in data.keys()]
        hist_plotter(
            datasets=same_template_data_mean,
            title=f"Mean FD: different softwares - {template}",
            path=path,
            labels=data.keys(),
        )

        hist_plotter(
            datasets=same_template_data_mean,
            title=f"SD of FD: different softwares - {template}",
            path=path,
            labels=data.keys(),
        )


def same_software_plots(template, softwares, data, path):
    for software in softwares:
        same_software_data_mean = [np.mean(data[software][template], axis=1) for template in template]
        same_software_data_mean = [np.std(data[software][template], axis=1) for software in template]
        hist_plotter(
            datasets=same_software_data_mean,
            title=f"Mean FD: different softwares - {template}",
            path=path,
            labels=data.keys(),
        )

        hist_plotter(
            datasets=same_software_data_mean,
            title=f"SD of FD: different softwares - {template}",
            path=path,
            labels=data.keys(),
        )
