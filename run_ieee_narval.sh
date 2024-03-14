#!/bin/bash
#SBATCH --job-name=flirt_PD
#SBATCH --output=flirt_ieee_PD.out
#SBATCH --error=flirt_ieee_PD.err
#SBATCH --time=2:30:0
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=2G
#SBATCH --account=rrg-jbpoline



module load apptainer/1.1.8
source /home/niusham/py11/bin/activate

# parallel "time bosh exec launch --imagepath ./fuzzy_fsl_6.0.4_latest.sif ./flirt-fuzzy.json ./{1}" ::: ./invocations_PD/anat-12dofs/mca/${SLURM_ARRAY_TASK_ID}/*
# parallel "time bosh exec launch --imagepath container_images/glatard_fsl_6.0.4_fuzzy-2023-12-08-a22e376466e7.simg descriptors/flirt-fuzzy.json ./{1}" ::: invocations/mca/${SLURM_ARRAY_TASK_ID}/*
# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/container_images/fuzzy_fsl_6.0.4_latest.sif ./flirt-fuzzy.json ./{1}" ::: ./invocations/anat-12dofs/mca/${SLURM_ARRAY_TASK_ID}/*
parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/container_images/vnmd_fsl_6.0.4-2021-04-22-ac3439c3920c.simg ./flirt.json ./{1}" ::: ./invocations_PD/anat-12dofs/ieee/*
