#!/bin/bash
#SBATCH --job-name=different_metrics_mca
#SBATCH --output=different_metrics_mca_{%j}.out 
#SBATCH --error=different_metrics_mca_{%j}.err 
#SBATCH --time=7:30:0 
#SBATCH --ntasks=50 
#SBATCH --mem-per-cpu=4G 
#SBATCH --account=rrg-jbpoline 
#SBATCH --array=1-10

PD_DIR=$1
METRICS_DIR=$2

module load apptainer/1.2.4
source /home/niusham/py11/bin/activate
parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/container_images/glatard_fsl_6.0.4_fuzzy-2023-12-08-a22e376466e7.simg ./descriptors/flirt.json ./{1}" ::: ./metrics_exp/${PD_DIR}/invocation/${METRICS_DIR}/flirt/2009Asym/anat-12dofs/mca/${SLURM_ARRAY_TASK_ID}/*