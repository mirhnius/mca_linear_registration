#!/usr/bin/env python3

import pathlib
import argparse
import numpy as np
import pandas as pd

PD = "Parkinson's Disease"
HC = "Healthy Control"
COHORT = "COHORT_DEFINITION"
BASELINE = "BL"
VISIT = "visit"
ID = "bids_id"
SESSION = "ses-BL"
ANATOMICAL = "anat"
ACCUSITION = "acq-sag3D"
MODALITY = "T1w"
RUN = "run-01"
SUFFIX = ".nii.gz"
MOSUF = MODALITY + SUFFIX
ORIGINAL = "ieee"
MCA = "mca"
MAT = ".mat"
PATTERN = pathlib.Path("") / SESSION / ANATOMICAL / f"sub-*_{SESSION}_{ACCUSITION}_{RUN}_{MOSUF}"


def join_path(*args):
    return str(pathlib.Path(*args).resolve())


def valid_path(path_str, pattern=PATTERN):

    path = pathlib.Path(path_str)
    if path.exists():
        if len(list(pathlib.Path(path).glob(str(pattern)))) == 0:
            return False
        return True

    return False


parser = argparse.ArgumentParser(description="Select subjects from a list of subjects")
parser.add_argument("--input_dir", type=str, help="Path to parent directory containing subject directories")
parser.add_argument("--list_path", type=str, help="Path to tsv file containing subject list")
parser.add_argument("--n", type=int, default=50, help="Number of subjects to select")

args = parser.parse_args()

df = pd.read_csv(args.list_path)

df_healthy = df[(df[COHORT] == HC) & (df[VISIT] == BASELINE)]
df_healthy = df_healthy.drop_duplicates(subset=[ID])

df_parkinsons = df[(df[COHORT] == PD) & (df[VISIT] == BASELINE)]
df_parkinsons = df_parkinsons.drop_duplicates(subset=[ID])

HC_ID = df_healthy[ID]
PD_ID = df_parkinsons[ID]

df_healthy["path"] = df_healthy[ID].apply(lambda x: join_path(args.input_dir, x))
df_parkinsons["path"] = df_parkinsons[ID].apply(lambda x: join_path(args.input_dir, x))

df_healthy["valid"] = df_healthy["path"].apply(valid_path)
df_parkinsons["valid"] = df_parkinsons["path"].apply(valid_path)

df_healthy_exist = df_healthy[df_healthy["valid"] == True]
df_parkinsons_exist = df_parkinsons[df_parkinsons["valid"] == True]

HC_index_selected = np.random.choice(df_healthy_exist.index, args.n, replace=False)
PD_index_selected = np.random.choice(df_parkinsons_exist.index, args.n, replace=False)

HC_selected_paths = df_healthy.loc[HC_index_selected, "path"]
PD_selected_paths = df_parkinsons.loc[PD_index_selected, "path"]


HC_selected_paths.to_csv("HC_selected_paths.txt", index=False, header=False)
PD_selected_paths.to_csv("PD_selected_paths.txt", index=False, header=False)
