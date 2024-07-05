#!/bin/bash

#SBATCH --job-name=ieee_ants_HC_sym2009
#SBATCH --output=ieee_ants_HC_sym2009.out
#SBATCH --error=ieee_ants_HC_sym2009.err
#SBATCH --time=2:30:0
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=2G
#SBATCH --account=rrg-jbpoline
#SBATCH --array=1

module load apptainer/1.2.4
source /home/niusham/py11/bin/activate


# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/ants_v2.5.0.sif ./ANTS.json ./{1}" ::: ./pipline/hc/invocations/ants/MNI152NLin2009cAsym_res-01/anat-12dofs/ieee/*
# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/ants_v2.5.0.sif ./ANTS.json ./{1}" ::: ./pipline/pd/invocations/ants/MNI152NLin2009cAsym_res-01/anat-12dofs/ieee/*
# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/ants_v2.5.0.sif ./descriptors/ANTS.json ./{1}" ::: ./pipline/pd/invocations/ants/MNI152NLin2009cSym_res-1/anat-12dofs/ieee/*
parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/ants_v2.5.0.sif ./descriptors/ANTS.json ./{1}" ::: ./pipline/hc/invocations/ants/MNI152NLin2009cSym_res-1/anat-12dofs/ieee/*
