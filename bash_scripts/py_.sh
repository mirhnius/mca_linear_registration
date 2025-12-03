#!/bin/bash
#SBATCH --job-name=py_plotter
#SBATCH --output=py_plotter.out
#SBATCH --error=py_plotter.err
#SBATCH --time=3:30:0
#SBATCH --mem-per-cpu=2G
#SBATCH --account=rrg-jbpoline

# module load python/3.11.5
# source /home/niusham/py11/bin/activate
# python3 /home/niusham/.vscode-server/extensions/ms-python.python-2024.4.1/python_files/printEnvVariablesToFile.py /home/niusham/.vscode-server/extensions/ms-python.python-2024.4.1/python_files/deactivate/bash/envVars.txt
source ../../../rrg-jbpoline/niusham/mca_linear_registration/.venv/bin/activate
# python3 QC_plot.py --template /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz -s flirt -i ./metrics_exp/hc/output/normmi ./metrics_exp/pd/output/normmi -o ./QC_metrics/hc/normmi ./QC_metrics/pd/normmi -l /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/HC_selected_subjects.txt /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/PD_selected_subjects.txt -t_n 2009Asym
# python3 QC_plot.py --template /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz -s flirt -i ./metrics_exp/hc/output/normcorr ./metrics_exp/pd/output/normcorr -o ./QC_metrics/hc/normcorr ./QC_metrics/pd/normcorr -l /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/HC_selected_subjects.txt /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/PD_selected_subjects.txt -t_n 2009Asym
# python3 QC_plot.py --template /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz -s flirt -i ./metrics_exp/hc/output/leastsq ./metrics_exp/pd/output/leastsq -o ./QC_metrics/hc/leastsq ./QC_metrics/pd/leastsq -l /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/HC_selected_subjects.txt /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/PD_selected_subjects.txt -t_n 2009Asym
# python3 QC_plot.py --template /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz -s flirt -i ./metrics_exp/hc/output/mutualinfo ./metrics_exp/pd/output/mutualinfo -o ./QC_metrics/hc/mutualinfo ./QC_metrics/pd/mutualinfo -l /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/HC_selected_subjects.txt /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/PD_selected_subjects.txt -t_n 2009Asym

python3 QC_plot.py --template /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz -s spm -i ./verrou/hc/output ./verrou/pd/output -o ./QC_verrou/hc ./QC_verrou/pd -l /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/HC_selected_subjects.txt /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/PD_selected_subjects.txt -t_n MNI152NLin2009cAsym_res-01