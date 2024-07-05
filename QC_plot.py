import argparse
from pathlib import Path

from lnrgst_mca.plot_utils import QC_plotter
from lnrgst_mca.load_utils import get_paths
from lnrgst_mca.constants import SPM, FLIRT, ANTS, SUFFIX, SUFFIX_PATTERNS

template = "tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz"


def QC_plots(
    input_parent_dir,
    sub_list_path,
    output_parent_dir,
    templates_names,
    templates_paths,
    softwares=[SPM, FLIRT, ANTS],
    pattern=None,
    ext=None,
    ddof=12,
    **kwargs,
):
    if pattern is None:
        try:
            pattern = [SUFFIX_PATTERNS[s] for s in softwares]
        except Exception as e:
            raise RuntimeError(f"The software in the list is not supported: {e}") from e
    if ext is None:
        ext = [SUFFIX] * len(softwares)
    # add something to check if len temp_name=tem_path
    dof_dir = f"anat-{str(ddof)}dofs"
    for i, temp_name in enumerate(templates_names):
        for index, soft in enumerate(softwares):
            try:
                registered_image_dir = input_parent_dir / soft / temp_name / dof_dir
                if not registered_image_dir.is_dir():
                    raise NotADirectoryError(f"The path '{registered_image_dir}' is not a directory")

                paths_map = get_paths(registered_image_dir, sub_list_path, pattern=pattern[index], ext=ext[index])
                output_QC = output_parent_dir / soft / temp_name / dof_dir
                output_QC.mkdir(parents=True, exist_ok=True)

                QC_plotter(paths_map, output_QC, template=templates_paths[i], **kwargs)

            except NotADirectoryError as e:
                print(e)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Generating QC images for the linear registration outputs \
                                     of three different tools (SPM, FSL, and ANTS) for both perturbed and unperturbed results."
    )
    # required = parser.add_argument_group('Required arguments')
    parser.add_argument("-t", "--template", nargs="+", type=str, required=True, default=template, help="templates' paths")
    parser.add_argument("-s", "--software", nargs="+", type=str, default=None, help="softwares' names")
    parser.add_argument("-i", "--input", type=str, nargs="+", required=True, help="path to input directories")
    parser.add_argument("-o", "--output", type=str, nargs="+", required=True, help="path to save output plots")
    parser.add_argument("-t_n", "--template_name", nargs="+", default=None, type=str, help="templates' names")
    parser.add_argument("-l", "--subject_list", nargs="+", default=None, type=str, help="tlist of subjects")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    templates = args.template
    softwares = args.software
    inputs = args.input
    outputs = args.output
    subject_lists = args.subject_list
    template_names = args.template_name

    if len(templates) != len(template_names):
        raise ValueError(f"{len(templates)} != {len(template_names)}")

    if len(subject_lists) != len(outputs):
        raise ValueError(f"{len(subject_lists)} != {len(outputs)}")

    if len(inputs) != len(outputs):
        raise ValueError(f"{len(inputs)} != {len(outputs)}")

    for i in range(len(inputs)):

        QC_plots(
            input_parent_dir=Path(inputs[i]),
            sub_list_path=Path(subject_lists[i]),
            output_parent_dir=Path(outputs[i]),
            templates_names=template_names,
            templates_paths=templates,
            softwares=softwares,
            display_mode="mosaic",
        )

        # fix the extra nii.gz in the names
        # add the option to get QC result only for ieee
