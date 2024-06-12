from lnrgst_mca.plot_utils import slice_plotter

# from lnrgst_mca.load_utils import get_paths
from pathlib import Path

template = "/home/niusha/Documents/Codes/mca_linear_registration/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz"
paths = [
    "/home/niusha/Documents/Codes/mca_linear_registration/outputs_new_pipline/flirt/ieee/sub-150110_ses-BL.nii.gz",
    "/home/niusha/Documents/Codes/mca_linear_registration/outputs_new_pipline/flirt/ieee/sub-3001_ses-BL.nii.gz",
]

slice_plotter(
    paths,
    template=template,
    title_prefix="test",
    output_dir=Path("/home/niusha/Documents/Codes/mca_linear_registration/"),
    display_mode="mosaic",
    levels=[0.4],
)
print("_".join([Path(paths[0]).parent.name, Path(paths[0]).name]))
