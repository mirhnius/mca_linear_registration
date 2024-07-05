#!/bin/bash
#SBATCH --job-name=mca_ants_HC_2009cSym
#SBATCH --output=mca_ants_HC_2009cSym.out
#SBATCH --error=mca_ants_HC_2009cSym.err
#SBATCH --time=5:30:0
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=5G
#SBATCH --account=rrg-jbpoline
#SBATCH --array=1-10


module load apptainer/1.2.4
source /home/niusham/py11/bin/activate
#MCA

# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/yohanchatelain-ants-v2.5.0-fuzzy.simg ./ANTS-fuzzy.json ./{1}" ::: ./pipline/pd/invocations/ants/anat-12dofs/mca/${SLURM_ARRAY_TASK_ID}/*
# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/yohanchatelain-ants-v2.5.0-fuzzy.simg ./ANTS-fuzzy.json ./{1}" ::: ./pipline/hc/invocations/ants/anat-12dofs/mca/${SLURM_ARRAY_TASK_ID}/*
parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/yohanchatelain-ants-v2.5.0-fuzzy.simg ./descriptors/ANTS-fuzzy.json ./{1}" ::: ./pipline/hc/invocations/ants/MNI152NLin2009cSym_res-1/anat-12dofs/mca/${SLURM_ARRAY_TASK_ID}/*
# parallel "time bosh exec launch --imagepath /home/niusham/projects/rrg-glatard/brainhack-2023-linear-registration/yohanchatelain-ants-v2.5.0-fuzzy.simg ./descriptors/ANTS-fuzzy.json ./{1}" ::: ./pipline/pd/invocations/ants/MNI152NLin2009cSym_res-1/anat-12dofs/mca/${SLURM_ARRAY_TASK_ID}/*
