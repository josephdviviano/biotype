#!/usr/bin/env python

from glob import glob
import re
import os

def extract_euler(logfile):
    with open(logfile) as fi:
        logtext = fi.read()
    p = re.compile(r"orig.nofix lheno =\s+(-?\d+), rheno =\s+(-?\d+)")
    results = p.findall(logtext)
    if len(results) != 1:
        raise Exception("Euler number could not be extracted from {}".format(logfile))
    lh_euler, rh_euler = results[0]

    return int(lh_euler), int(rh_euler)

f = open('qc_euler.csv', 'w')
f.write('id,lh,rh\n')
subjects = glob('XBRAIN_*')
for subject in subjects:
    subject_name = os.path.basename(subject)
    logfile = os.path.join(subject, 'scripts', 'recon-all.log')
    try:
        l, r = extract_euler(logfile)
    except Exception as e:
        print(e)
        continue
    f.write('{},{},{}\n'.format(subject_name, l, r))

f.close()


