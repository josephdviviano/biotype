#!/bin/bash

for filename in $(ls /projects/colin/SPINS_hcp/data/*/scaled/spmT_0005.dscalar.nii); do
    subject=$(basename $(dirname $(dirname ${filename})))
    rsync -rav ${filename} ../data/${subject}_IMOB_glm-spmT-0005.dscalar.nii
done
