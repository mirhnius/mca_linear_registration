#!/bin/bash

#SBATCH --job-name=mca_verrou_spm_HC_2009cAsym
#SBATCH --output=mca_verrou_spm_HC_2009cAsym.out
#SBATCH --error=mca_spm_HC_2009cAsym.err
#SBATCH --time=2:30:0
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=2G
#SBATCH --account=rrg-jbpoline
#SBATCH --array=1-10

module load apptainer/1.2.4 
source /home/niusham/py11/bin/activate


parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/verrousmp_latest.sif ./descriptors/verrou_run_affreg.json ./{1}" ::: verrou/hc/invocation/spm/MNI152NLin2009cAsym_res-01/anat-12dofs/mca/${SLURM_ARRAY_TASK_ID}/*
