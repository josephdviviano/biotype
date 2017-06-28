#!/usr/bin/env python

import pandas as pd
import csv, os, sys

lines = ['experiment,AUC,Accuracy,Recall,Precision,f1\n']
db = pd.read_csv(sys.argv[1])

for idx, row in db.iterrows():
   string = '{},{:0.2f} / {:0.2f},'.format(row['experiment'], row['auc_test_macro'], row['auc_train_macro'])
   string += '{:0.2f} / {:0.2f},'.format(row['accuracy_test'], row['accuracy_train'])
   string += '{:0.2f} / {:0.2f},'.format(row['recall_test_macro'], row['recall_train_macro'])
   string += '{:0.2f} / {:0.2f},'.format(row['precision_test_macro'], row['precision_train_macro'])
   string += '{:0.2f} / {:0.2f}\n'.format(row['f1_test_macro'], row['f1_train_macro'])
   lines.append(string)

with open(sys.argv[2], 'wb') as r:
    r.writelines(lines)

