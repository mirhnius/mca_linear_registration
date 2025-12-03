#!/bin/bash

#SBATCH --job-name=ieee_ants_PD
#SBATCH --output=ieee_ants_PD.out
#SBATCH --error=ieee_ants_PD.err
#SBATCH --time=5:30:0
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=5G
#SBATCH --account=rrg-jbpoline
#SBATCH --array=1


module load apptainer/1.1.8
source /home/niusham/py11/bin/activate


# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/ants_v2.5.0.sif ./descriptor/ANTS.json ./{1}" ::: ./pipline/hc/invocations/ants/anat-12dofs/ieee/*
parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/ants_v2.5.0.sif ./descriptors/ANTS.json ./{1}" ::: ./pipline/pd/invocations/ants/anat-12dofs/ieee/*
