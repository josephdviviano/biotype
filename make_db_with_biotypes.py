#!/usr/bin/env python

import pandas as pd
import numpy as np
import os

if 'replication' in os.path.basename(os.path.abspath(os.curdir)):
    a = pd.read_csv('/projects/jviviano/data/xbrain/assets/database_replication_xbrain.csv')
else:
    a = pd.read_csv('/projects/jviviano/data/xbrain/assets/database_xbrain.csv')

b = pd.read_csv('xbrain_database.csv')
db = a.merge(b)
db.to_csv('xbrain_database_with_biotypes.csv')

