#!/bin/bash

echo "all 'removed' files will have the suffix .bak"
for timeseries in $(ls *RST*.csv); do
    n=$(head -n1 ${timeseries} | grep -o ',' | wc -l)
    if [ ${n} -lt 200 ]; then
        mv ${timeseries} ${timeseries}.bak
    fi
done

for timeseries in $(ls *EA_GM*.csv); do
    n=$(head -n1 ${timeseries} | grep -o ',' | wc -l)
    if [ ${n} -lt 600 ]; then
        mv ${timeseries} ${timeseries}.bak
    fi
done

for timeseries in $(ls *IMOB_GM*.csv); do
    n=$(head -n1 ${timeseries} | grep -o ',' | wc -l)
    if [ ${n} -lt 200 ]; then
        mv ${timeseries} ${timeseries}.bak
    fi
done

for timeseries in $(ls *SPRL*.csv); do
    n=$(head -n1 ${timeseries} | grep -o ',' | wc -l)
    if [ ${n} -lt 200 ]; then
        mv ${timeseries} ${timeseries}.bak
    fi
done

