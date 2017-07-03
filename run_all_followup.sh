#!/bin/bash

echo "resting state biotype: discovery sample"
cd biotype-restconn_test-restconn
make_db_with_biotypes.py
biotype_plot_scores.py
biotype_plot_scores_subjectwise.py
biotype_plot_xy_loadings.py no no flip
cd ..

echo "resting state biotype: replication sample"
cd biotype-restconn_test-restconn_replication
make_db_with_biotypes.py
biotype_plot_scores.py
biotype_plot_scores_subjectwise.py
biotype_plot_xy_loadings.py no no flip
cd ..

echo "imob biotype: discovery sample"
cd biotype-imobconn_test-imobconn
make_db_with_biotypes.py
biotype_plot_scores.py flip
biotype_plot_scores_subjectwise.py flip
biotype_plot_xy_loadings.py flip flip flip
cd ..

echo "imob biotype: replication sample"
cd biotype-imobconn_test-imobconn_replication
make_db_with_biotypes.py
biotype_plot_scores.py flip
biotype_plot_scores_subjectwise.py flip
biotype_plot_xy_loadings.py flip flip flip
cd ..

echo "ea biotype: discovery sample"
cd biotype-eaconn_test-eaconn
make_db_with_biotypes.py
biotype_plot_scores.py flip
biotype_plot_scores_subjectwise.py flip
biotype_plot_xy_loadings.py no no no
cd ..

echo "imob stat biotype: discovery sample"
cd biotype-imobstat_test-imobstat
make_db_with_biotypes.py
biotype_plot_scores.py flip
biotype_plot_scores_subjectwise.py flip
biotype_plot_xy_loadings.py no flip no
cd ..

echo "ea stat biotype: discovery sample"
cd biotype-eastat_test-eastat
make_db_with_biotypes.py
biotype_plot_scores.py
biotype_plot_scores_subjectwise.py
biotype_plot_xy_loadings.py no flip flip
cd ..

# compares all methods biotype stability
#biotype_stability_analysis.py

# computes stat differences between groups (both conn and stat)
#group_connectivity_differences.py

# plots group differences (note: inputs and flips currently defined within script)
glass_brains.py

# compares connectivity differences between methods / replications
echo 'assessing connectivity similarities'
biotype_compare_replication.py xbrain_biotype_rois_rest_dp_conn.csv no xbrain_biotype_rois_rest-replication_dp_conn.csv no biotype_rest-vs-replication
biotype_compare_replication.py xbrain_biotype_rois_imob_dp_conn.csv no xbrain_biotype_rois_imob-replication_dp_conn.csv no biotype_imob-vs-replication
biotype_compare_replication.py xbrain_biotype_rois_imob_dp_conn.csv no xbrain_biotype_rois_ea_dp_conn.csv no biotype_imob-vs-ea
biotype_compare_replication.py xbrain_biotype_rois_rest_dp_conn.csv no xbrain_biotype_rois_ea_dp_conn.csv no biotype_rest-vs-ea
biotype_compare_replication.py xbrain_biotype_rois_rest_dp_conn.csv no xbrain_biotype_rois_imob_dp_conn.csv no biotype_rest-vs-imob

# make summary tables from cross validation
summarize_results.sh # sticks everything in giant spreadsheets
make_reduced_tables.sh # makes a publication-friendly table
