#!/bin/bash
# This is an example script for point cloud extraction with DANBO.
# It takes around 30 minutes for each subject on RTX 3080.

declare -a subjects=(
    'S1'
    'S5'
    'S6'
    'S7'
    'S8'
    'S9'
    'S11'
)
logbase='logs'
outputbase='extracted_points'

for index in "${!subjects[@]}"; do
    subject=${subjects[index]}
    expname="danbo_${subject}"

    # Step 1.: train DANBO for 10k iterations, this should take about 30 minutes on RTX 3080.
    python train.py --config-name danbo_vof basedir=${logbase} expname=${expname} dataset.subject=${subject} num_workers=16 iters=10000 

    # Step 2.: run marching cubes to extract the point cloud.
    python -m tools.extract_points --pretrained_path ${logbase}/${expname} --ckpt 10000 --threshold 20.0 --output_name ${outputbase}/anchor_pts_${subject}.th
done

