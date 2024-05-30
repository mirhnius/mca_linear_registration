#!/bin/bash

#SBATCH --job-name=preprocess_PD
#SBATCH --output=preprocess_PD.out
#SBATCH --error=preprocess_PD.err
#SBATCH --time=0:30:0
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=2G
#SBATCH --account=rrg-jbpoline
#SBATCH --array=1

module load apptainer/1.1.8
source /home/niusham/py11/bin/activate

# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/container_images/vnmd_fsl_6.0.4-2021-04-22-ac3439c3920c.simg ./robustfov.json ./{1}" ::: ./pipline/pd/invocations/robustfov/*

# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/container_images/vnmd_fsl_6.0.4-2021-04-22-ac3439c3920c.simg ./fsl_bet.json ./{1}" ::: ./pipline/pd/invocations/bet/*

parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/container_images/vnmd_fsl_6.0.4-2021-04-22-ac3439c3920c.simg ./flirt.json ./{1}" ::: ./pipline/pd/invocations/flirt/anat-12dofs/ieee/*
