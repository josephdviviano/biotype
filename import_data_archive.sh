#!/bin/bash

archive='/archive/data/'
datadir='/projects/jviviano/data/xbrain/data/'

# import all timeseries .csvs
find ${archive} -name '*roi-timeseries.csv' -exec rsync -a {} ${datadir} \;

