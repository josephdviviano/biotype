#!/bin/bash

for folder in $(ls -d biotype*/); do
    cd ${folder}
    echo "working on ${folder}"
    ../make-db-with-biotypes.py
    ../biotype_plot_scores.py
    ../biotype_plot_scores_subjectwise.py
    cd ../
done

#./biotype_stability_analysis.py
#./group_connectivity_differences.py
#./glass_brains.py

