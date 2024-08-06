#!/bin/bash

#SBATCH --job-name=within_software_analysis
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=rrg-jbpoline
#SBATCH --array=0-5

softwares=("spm" "spm" "flirt" "flirt" "ants" "ants")
templates=("MNI152NLin2009cAsym_res-01" "MNI152NLin2009cSym_res-1" "MNI152NLin2009cAsym_res-01" "MNI152NLin2009cSym_res-1" "MNI152NLin2009cAsym_res-01" "MNI152NLin2009cSym_res-1")

software=${softwares[$SLURM_ARRAY_TASK_ID]}
template=${templates[$SLURM_ARRAY_TASK_ID]}

module load python/3.11.5
# source /home/niusham/py11/bin/activate
source /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/.venv/bin/activate
# /bin/python3 /home/niusham/.vscode-server/extensions/ms-python.python-2024.4.1/python_files/printEnvVariablesToFile.py /home/niusham/.vscode-server/extensions/ms-python.python-2024.4.1/python_files/deactivate/bash/envVars.txt
python3 ./within_software_analysis.py -t $template -s $software -d ./outputs_plots/diagrams