#!/bin/bash

# Verrou SPM test

SOURCE_IMAGE="sub-3000_out.nii"
TEMPLATE_IMAGE="./tpl-MNI152NLin2009cAsym_res-01_T1w_roi_neck_test.nii"
OUTPUT_PATH="spm_verrou_output"
# REGISTERED_IMAGE=""
# TRANSFORMATION_MATRIX=""
N_RUNS=10

for i in $(seq 1 $N_RUNS); do
    OUTPUT_DIR="${OUTPUT_PATH}/run_${i}"
    mkdir -p "$OUTPUT_DIR"

    docker run --rm -v $(pwd):/code -w /code \
        verrouspm valgrind --tool=verrou --rounding-mode=random --trace-children=yes octave --no-gui --eval "run_affreg('${SOURCE_IMAGE}', '${TEMPLATE_IMAGE}', '${OUTPUT_DIR}/registered_image.nii', '${OUTPUT_DIR}/transformation_matrix.txt')"
done
wait
echo "All runs completed"

