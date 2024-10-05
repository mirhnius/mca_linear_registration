#!/bin/bash
#SBATCH --job-name=different_metrics
#SBATCH --output=different_metrics_{%j}.out 
#SBATCH --error=different_metrics_{%j}.err 
#SBATCH --time=0:30:0 
#SBATCH --ntasks=50 
#SBATCH --mem-per-cpu=2G 
#SBATCH --account=rrg-jbpoline 

PD_DIR=$1
METRICS_DIR=$2

module load apptainer/1.2.4
source /home/niusham/py11/bin/activate
parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/container_images/vnmd_fsl_6.0.4-2021-04-22-ac3439c3920c.simg ./descriptors/flirt.json ./{1}" ::: ./metrics_exp/${PD_DIR}/invocation/${METRICS_DIR}/flirt/2009Asym/anat-12dofs/ieee/*