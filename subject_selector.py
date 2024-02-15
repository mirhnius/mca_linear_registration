#!/usr/bin/env python3

import pathlib
import argparse
import numpy as np
import pandas as pd

PD = "Parkinson's Disease"
HC = "Healthy Control"
COHORT = "COHORT_DEFINITION"

def join_path(*args):
    return str(pathlib.Path(*args).resolve())

parser = argparse.ArgumentParser(description='Select subjects from a list of subjects')
parser.add_argument('--input_dir', type=str, help='Path to parent directory containing subject directories')
parser.add_argument('--list_path', type=str, help='Path to tsv file containing subject list')
parser.add_argument('--n', type=int, default=50, help='Number of subjects to select')

args = parser.parse_args()

df = pd.read_csv(args.list_path, sep='\t')
df_healthy = df[df[COHORT] == HC]
df_parkinsons = df[df[COHORT] == PD]

HC_index_selected = np.random.choice(df_healthy.index, args.n, replace=False)
PD_index_selected = np.random.choice(df_parkinsons.index, args.n, replace=False)

HC_selected_ID = df_healthy.loc[HC_index_selected, COHORT]
PD_selected_ID = df_parkinsons.loc[PD_index_selected, COHORT]

HC_selected_paths = HC_selected_ID.applymap(lambda x: join_path(args.input_dir, x))
PD_selected_paths = PD_selected_ID.applymap(lambda x: join_path(args.input_dir, x))

HC_selected_paths.to_csv('HC_selected_paths.txt', index=False, header=False)
PD_selected_paths.to_csv('PD_selected_paths.txt', index=False, header=False)


```



