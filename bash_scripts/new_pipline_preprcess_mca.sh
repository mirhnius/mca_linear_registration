#!/bin/bash
#SBATCH --job-name=mca_flirt_PD
#SBATCH --output=mca_flirt_PD.out
#SBATCH --error=mca_flirt_PD.err
#SBATCH --time=2:30:0
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=2G
#SBATCH --account=rrg-jbpoline
#SBATCH --array=1-10


module load apptainer/1.1.8
source /home/niusham/py11/bin/activate
#MCA

# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/container_images/glatard_fsl_6.0.4_fuzzy-2023-12-08-a22e376466e7.simg ./flirt-fuzzy.json ./{1}" ::: ./pipline/hc/invocations/flirt/anat-12dofs/mca/${SLURM_ARRAY_TASK_ID}/*
parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/container_images/glatard_fsl_6.0.4_fuzzy-2023-12-08-a22e376466e7.simg ./flirt-fuzzy.json ./{1}" ::: ./pipline/pd/invocations/flirt/anat-12dofs/mca/${SLURM_ARRAY_TASK_ID}/*
