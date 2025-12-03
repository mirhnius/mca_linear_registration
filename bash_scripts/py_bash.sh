#!/bin/bash
#SBATCH --job-name=py_plotter
#SBATCH --output=py_plotter.out
#SBATCH --error=py_plotter.err
#SBATCH --time=1:30:0
#SBATCH --mem-per-cpu=2G
#SBATCH --account=rrg-jbpoline

module load python/3.11.5
source /home/niusham/py11/bin/activate
# python3 /home/niusham/.vscode-server/extensions/ms-python.python-2024.4.1/python_files/printEnvVariablesToFile.py /home/niusham/.vscode-server/extensions/ms-python.python-2024.4.1/python_files/deactivate/bash/envVars.txt
# python ./test.py
python3 QC_plot.py --template /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/tpl-MNI152NLin2009cSym_res-1_label-GM_probseg.nii.gz -s spm ants flirt -i /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/pipline/hc/outputs /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/pipline/pd/outputs -o /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/QC/hc /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/QC/pd -l /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/HC_selected_subjects.txt /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/PD_selected_subjects.txt -t_n MNI152NLin2009cSym_res-1