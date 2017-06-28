#!/bin/bash
header="experiment,k,accuracy_train,accuracy_test,recall_train_macro,recall_test_macro,recall_train_micro,recall_test_micro,precision_train_macro,precision_test_macro,precision_train_micro,precision_test_micro,f1_train_macro,f1_test_macro,f1_train_micro,f1_test_micro,auc_train_macro,auc_test_macro,auc_train_micro,auc_test_micro,n_features_retained,k,s_accuracy_train,s_accuracy_test,s_recall_train_macro,s_recall_test_macro,s_recall_train_micro,s_recall_test_micro,s_precision_train_macro,s_precision_test_macro,s_precision_train_micro,s_precision_test_micro,s_f1_train_macro,s_f1_test_macro,s_f1_train_micro,s_f1_test_micro,s_auc_train_macro,s_auc_test_macro,s_auc_train_micro,s_auc_test_micro,s_n_features_retained"

echo ${header} > summary_ysplit.csv
for f in $(ls ysplit*/xbrain_results.csv); do
    name=$(echo ${f})
    mean=$(cat ${f} | tail -2 | head -1)
    sem=$(cat ${f} | tail -1)
    echo "${name},${mean},${sem}"
done >> summary_ysplit.csv


echo ${header} > summary_diagnose.csv
for f in $(ls diagnose*/xbrain_results.csv); do
    name=$(echo ${f})
    mean=$(cat ${f} | tail -2 | head -1)
    sem=$(cat ${f} | tail -1)
    echo "${name},${mean},${sem}"
done >> summary_diagnose.csv

echo ${header} > summary_biotype.csv
for f in $(ls biotype*/xbrain_results.csv); do
    name=$(echo ${f})
    mean=$(cat ${f} | tail -2 | head -1)
    sem=$(cat ${f} | tail -1)
    echo "${name},${mean},${sem}"
done >> summary_biotype.csv

