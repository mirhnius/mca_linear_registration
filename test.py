from lnrgst_mca.plot_utils import QC_plotter
from lnrgst_mca.load_utils import get_paths
from pathlib import Path

from lnrgst_mca.constants import SPM, FLIRT, ANTS, SUFFIX

# template = "/home/niusha/Documents/Codes/mca_linear_registration/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz"
# paths = [
#     "/home/niusha/Documents/Codes/mca_linear_registration/outputs_new_pipline/flirt/ieee/sub-150110_ses-BL.nii.gz",
#     "/home/niusha/Documents/Codes/mca_linear_registration/outputs_new_pipline/flirt/ieee/sub-3001_ses-BL.nii.gz",
# ]

# slice_plotter(
#     paths,
#     template=template,
#     title_prefix="test",
#     output_dir=Path("/home/niusha/Documents/Codes/mca_linear_registration/"),
#     display_mode="mosaic",
#     levels=[0.4],
# )
# print("_".join([Path(paths[0]).parent.name, Path(paths[0]).name]))

template = "tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz"
fsl_path_HC = Path("./pipline/hc/outputs")  # /anat-12dofs
fsl_paths_HC = get_paths(fsl_path_HC, Path("./HC_selected_subjects.txt"), pattern="_ses-BL", ext=".nii.gz")

# slice_plotter(fsl_paths_HC['sub-4079']['mca'],
#         template=template,
#         title_prefix="test",
#         output_dir=Path("./test"),
#         display_mode="mosaic",
#         levels=[0.4])

# paths_map = fsl_paths_HC

# templates = ["test"]
# softwares = [SPM, FSL, ANTS]

# input_parent_dir = Path("")
# ddof = 12
# output_parent_dir = Path("")
# dof_dir = f"anat-{str(ddof)}dofs" # maybe add list idk

# pattern="_ses-BL"
# ext=".nii.gz"

# pattern = [a,b,c]
# def get_paths(parent_dir: Path, subjects_file: Path=None, n_mca: int = 10, pattern: str = None, ext: str = ".mat"):
#     """
#     Generate IEEE and MCA paths based on a list of subjects and read from a file.

#     Parameters:
#         parent_dir(Path): The parent dirctory containing the IEEE and MCA directories.
#         subjects_file(Path): A file containing the list of subjects.
#         n_mca(int): The number of MCA directories.
#         pattern(str): The file name pattern.
#         ext(str): The file extension.

#     Returns:
#         A directory with subject IDs as keys, ech containing paths to respective IEEE and MCA files.
#     """
#     # Read the subjects from the file
#     print(subjects_file)
#     subjects = []
#     if subjects_file:
#         with open(subjects_file, "r") as file:
#             subjects.extend(line.strip() for line in file)
#     else:
#         print("@@@@@@@@@@@@@@@@@")
#         ieee_dir = parent_dir / "ieee"
#         print(ieee_dir)
#         if ieee_dir.exists():
#             print("wewq")
#             # Construct a regex pattern that captures the subject ID and the pattern from filenames
#             # Correct pattern
#             regex = re.compile(r'(sub-\d+)_([a-zA-Z0-9-]+)' + re.escape(ext))

#             for file in ieee_dir.glob(f'*{ext}'):
#                 print(file)
#                 match = regex.search(file.stem)
#                 # print(match)
#                 if match:
#                     subject_id = match.group(1)  # This captures 'sub-1000'
#                     pattern = match.group(2)    # This captures the 'pattern' part of the filename
#                     subjects.append((subject_id, pattern))  # Store as tuples


#     # Generate the paths
#     paths = {}
#     for sub in subjects:
#         ieee_path = parent_dir / IEEE / f"{sub}{pattern}{ext}"
#         mca_paths = [parent_dir / MCA / str(i) / f"{sub}{pattern}{ext}" for i in range(1, n_mca + 1)]

#         paths[sub] = {IEEE: str(ieee_path), MCA: [str(p) for p in mca_paths]}

#     return paths


# def find_scans(input_dir: pathlib.Path, pattern=PATTERN, sub_dirs: List[str] = None) -> List[pathlib.Path]:
#     """
#     Finds and returns a list of scan paths based on the specified pattern.

#     Parameters:
#         input_dir (pathlib.Path): The directory to search within.
#         pattern (pathlib.Path): The pattern to search for.
#         sub_dirs (List[str]): Optional list of subdirectories to limit the search.

#     Returns:
#         List[pathlib.Path]: A list of scan paths that match the specified pattern.
#     """

#     if sub_dirs is None:
#         return list(input_dir.glob(str(pattern)))

#     scan_paths = []
#     pattern = pathlib.Path("") / BASELINE_SESSION / ANATOMICAL / f"sub-*_{BASELINE_SESSION}_{ACQUISITION}_{RUN}_{MOSUF}"

#     for sub_dir in sub_dirs:
#         scan_paths.extend(list(pathlib.Path(sub_dir).glob(str(pattern))))

#         # if not list(pathlib.Path(sub_dir).glob(str(pattern))):
#         #     print(sub_dir)

#     return scan_paths


# def scan_filed_dict(scan_path: pathlib.Path) -> Dict:
#     """
#     Extracts and return scan-related fields from the scan path.

#     Parameters:
#         scan_path (pathlib.Path): The scan path to extract fields from.

#     Returns:
#         Dict: A dictionary containing the extracted fields.
#     """

#     subject_ID, session, _, run, modalitiy = scan_path.name.split("_")
#     return {
#         "input_path": str(scan_path),
#         "subject": subject_ID,
#         "session": session,
#         "run": run,
#         "modality": modalitiy,
# }


def QC_plots(
    input_parent_dir,
    sub_list_path,
    output_parent_dir,
    templates_names,
    templates_paths,
    softwares=[SPM, FLIRT, ANTS],
    pattern=["_ses-BL", "_ses-BL", "_ses-BL0GenericAffine"],
    ext=[SUFFIX] * 3,
    ddof=12,
    **kwargs,
):

    dof_dir = f"anat-{str(ddof)}dofs"
    for i, temp_name in enumerate(templates_names):
        for index, soft in enumerate(softwares):
            try:
                registered_image_dir = input_parent_dir / soft / temp_name / dof_dir
                if not registered_image_dir.is_dir():
                    raise NotADirectoryError(f"The path '{registered_image_dir.name}' is not a directory")

                paths_map = get_paths(registered_image_dir, sub_list_path, pattern=pattern[index], ext=ext[index])
                output_QC = output_parent_dir / soft / temp_name / dof_dir
                output_QC.mkdir(parents=True, exist_ok=True)

                QC_plotter(paths_map, output_QC, template=templates_paths[i], **kwargs)

            except NotADirectoryError as e:
                print(e)


if __name__ == "__main__":

    outputs_path_HC = Path("./pipline/hc/outputs")
    sub_list_path_HC = Path("./HC_selected_subjects.txt")

    outputs_path_PD = Path("./pipline/pd/outputs")
    sub_list_path_PD = Path("./PD_selected_subjects.txt")

    QC_plots(outputs_path_HC, sub_list_path_HC, Path("./test/hc"), [""], [template], [""], ["_ses-BL"], [SUFFIX], display_mode="mosaic")
    # QC_plots(outputs_path_PD, sub_list_path_PD, Path("./test/pd"),
    # [''], [template], [FLIRT], ["_ses-BL", "_ses-BLWarped"], [SUFFIX]*2, display_mode="mosaic")
#   _ses-BLWarped
