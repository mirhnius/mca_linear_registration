import lnrgst_mca.invocation_utils as invo
from pathlib import Path

pattern = Path("") / "sub-*.nii.gz"
in_path = Path("./pipline/hc/outputs/preprocess")
out_path = Path("./pipline/hc/outputs/flirt")


def scan_filed_dict_short(scan_path):

    subject_ID, session = scan_path.name.split("_")
    session = session.split(".")[0]
    return {
        "input_path": str(scan_path),
        "subject": subject_ID,
        "session": session,
    }


# p = Path('/home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/pipline/hc/outputs/preprocess/sub-3112_ses-BL.nii.gz')
# scan_filed_dict(p)


ref = Path("./tpl-MNI152NLin2009cAsym_res-01_T1w_neck_5.nii.gz")


for group in ["hc", "pd"]:
    in_path = Path(f"./pipline/{group}/outputs/preprocess")
    output_dir = Path(f"/home/niusham/metrics_exp/{group}/output")
    invocation_dir = Path(f"/home/niusham/metrics_exp/{group}/invocation")
    map_ = invo.create_subject_map(in_path, pattern=pattern, scanner=scan_filed_dict_short)
    for method in ["mutualinfo", "normcorr", "normmi", "leastsq"]:

        output_path = output_dir / method
        invocation_path = invocation_dir / method

        invo.FLIRT_IEEE_registration(map_, output_path, invocation_path, ref=ref, template_name="2009Asym", cost_function=method).create_invocations()

        invo.FLIRT_MCA_registration(map_, output_path, invocation_path, ref=ref, template_name="2009Asym", cost_function=method).create_invocations()
