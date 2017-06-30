#!/bin/bash
update_database.py \
    /projects/jviviano/data/xbrain/assets/database_new_jdv.csv \
    /projects/jviviano/data/xbrain/data/ \
    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

# filter for spins subjects only
head -1 /projects/jviviano/data/xbrain/assets/database_xbrain.csv > /projects/jviviano/data/xbrain/assets/database_xbrain_SPN.csv
cat /projects/jviviano/data/xbrain/assets/database_xbrain.csv | grep SPN >> /projects/jviviano/data/xbrain/assets/database_xbrain_SPN.csv

update_database.py \
    /projects/jviviano/data/xbrain/assets/database_replication_spins.csv \
    /projects/jviviano/data/xbrain/data/ \
    /projects/jviviano/data/xbrain/assets/database_replication_xbrain.csv

# filter for EPI data only (no _SPRL_ scans)
head -1 /projects/jviviano/data/xbrain/assets/database_replication_xbrain.csv > /projects/jviviano/data/xbrain/assets/database_replication_xbrain_nospiral.csv
cat /projects/jviviano/data/xbrain/assets/database_replication_xbrain.csv | grep RST >> /projects/jviviano/data/xbrain/assets/database_replication_xbrain_nospiral.csv

