#!/bin/bash
#SBATCH --job-name=ieee_spm_PD_sym2009
#SBATCH --output=ieee_spm_PD_sym2009.out
#SBATCH --error=ieee_spm_PD_sym2009.err
#SBATCH --time=0:30:0
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=2G
#SBATCH --account=rrg-jbpoline
#SBATCH --array=1

module load apptainer/1.2.4 
source /home/niusham/py11/bin/activate


# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/spm12_octave_new_latest.sif ./descriptors/Run_affreg.json ./{1}" ::: ./pipline/pd/invocations/spm/anat-12dofs/ieee/*
# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/spm12_octave_new_latest.sif ./descriptors/Run_affreg.json ./{1}" ::: ./pipline/hc/invocations/spm/anat-12dofs/ieee/*
# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/spm12_octave_new_latest.sif ./descriptors/Run_affreg.json ./{1}" ::: ./pipline/hc/invocations/spm/MNI152NLin2009cSym_res-1/anat-12dofs/ieee/*
parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-jbpoline/niusham/mca_linear_registration/spm12_octave_new_latest.sif ./descriptors/Run_affreg.json ./{1}" ::: ./pipline/pd/invocations/spm/MNI152NLin2009cSym_res-1/anat-12dofs/ieee/*
