#!/bin/bash

./make_table.py summary_ysplit.csv summary_ysplit_reduced.csv
./make_table.py summary_diagnose.csv summary_diagnose_reduced.csv
./make_table.py summary_biotype.csv summary_biotype_reduced.csv

