#!/usr/bin/env python

import pandas as pd
import csv, os, sys

lines = ['experiment,AUC,Accuracy,Recall,Precision,f1\n']
db = pd.read_csv(sys.argv[1])

# mean +/- std
for idx, row in db.iterrows():
   string = '{},{:0.2f} +/- {:0.2f},'.format(row['experiment'], row['auc_test_macro'], row['s_auc_train_macro'])
   string += '{:0.2f} +/- {:0.2f},'.format(row['accuracy_test'], row['s_accuracy_train'])
   string += '{:0.2f} +/- {:0.2f},'.format(row['recall_test_macro'], row['s_recall_train_macro'])
   string += '{:0.2f} +/- {:0.2f},'.format(row['precision_test_macro'], row['s_precision_train_macro'])
   string += '{:0.2f} +/- {:0.2f}\n'.format(row['f1_test_macro'], row['s_f1_train_macro'])
   lines.append(string)

with open(sys.argv[2], 'wb') as r:
    r.writelines(lines)

