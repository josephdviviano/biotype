#!/bin/bash
./update_database \
    /projects/jviviano/data/xbrain/assets/database_new_jdv.csv \
    /projects/jviviano/data/xbrain/data/ \
    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

# filter for spins subjects only
head -1 /projects/jviviano/data/xbrain/assets/database_xbrain.csv > /projects/jviviano/data/xbrain/assets/database_xbrain_SPN.csv
cat /projects/jviviano/data/xbrain/assets/database_xbrain.csv | grep SPN >> /projects/jviviano/data/xbrain/assets/database_xbrain_SPN.csv

#./update_database \
#    /projects/jviviano/data/xbrain/assets/database_replication_spins.csv \
#    /projects/jviviano/data/xbrain/data/ \
#    /projects/jviviano/data/xbrain/assets/database_replication_xbrain.csv