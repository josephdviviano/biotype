#!/bin/bash

for f in $(ls */xbrain_results.csv); do
    n=$(cat ${f} | wc -l)
    echo "${f}: ${n}"
done > output_counts.txt

