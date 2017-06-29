#!/usr/bin/env python
"""
Usage:
    update_database <database> <folder> <output>

For each row in  <database>'s ID column, find associated neuroimaging data in
<folder> and inset a full path to that data in the relevant column. The updated
database will be written to a new file <output>
"""
import sys, os
import glob
import pandas as pd

def assert_n(l, n):
    if len(l) != n:
        raise Exception

def main(database, folder, output):
    db = pd.read_csv(database, index_col=0)
    subjects = db.index
    files = glob.glob(folder + '/*')
    files = filter(lambda x: '.bak' not in x, files) # remove 'bad' timeseries etc.

    # add columns
    db['ts_imi'] = ''
    db['ts_obs'] = ''
    db['ts_imi_resid'] = ''
    db['ts_obs_resid'] = ''
    db['ts_imob_gm'] = ''
    db['stat_imob_ch'] = ''
    db['stat_imob'] = ''
    db['stat_ea'] = ''
    db['ts_ea_gm'] = ''
    db['ts_rst'] = ''
    db['ts_ea1'] = ''
    db['ts_ea2'] = ''
    db['ts_ea3'] = ''
    db['ts_ea_vid_resid'] = ''
    db['ts_ea_cvid_resid'] = ''

    # invert reaction time
    db['scog_er40_crt_columnqcrt_value_inv'] = db['scog_er40_crt_columnqcrt_value']*-1

    for subject in subjects:

        # collect data
        matches = filter(lambda x: subject in x, files)

        ts = filter(lambda x: 'timeseries.csv' in x, matches)
        nii = filter(lambda x: '.nii' in x, matches)

        # preprocessed IMOB data from archive
        ts_imi = filter(lambda x: '_IMI_' in x, ts)
        ts_obs = filter(lambda x: '_OBS_' in x, ts)

        # residuals of the GLM fit to the IMOB data from archive
        ts_imi_resid = filter(lambda x: 'glm_IM' in x, ts)
        ts_obs_resid = filter(lambda x: 'glm_OB' in x, ts)

        # IMOB data preprocessed as resting state data
        ts_imob_gm = filter(lambda x: '_IMOB_GM_roi-timeseries' in x, ts)

        # EA data preprocessed as resting state data
        ts_ea_gm = filter(lambda x: '_EA_GM_roi-timeseries' in x, ts)

        # IMOB GLM data preprocessed as the archive and analyzed by colin hawco
        stat_imob_ch = filter(lambda x: '_IMOB_glm-spmT-0005' in x, nii)

        # IMOB GLM data preprocessed and analyzed as the archive
        stat_imob = filter(lambda x: '_IMOB_emote-contrast' in x, nii)

        # EA GLM data preprocessed and analyzed as the archive
        stat_ea = filter(lambda x: '_EA_empathic-mod' in x, nii)

        # preprocessed EA data from the archive
        ts_emp = filter(lambda x: '_EMP_' in x, ts)

        # residuals of the GLM fit to the EA data from the archive
        ts_emp_vid = filter(lambda x: 'glm_vid_' in x, ts)
        ts_emp_cvid = filter(lambda x: 'glm_cvid_' in x, ts)

        # preprocessed resting state data from the archive
        ts_rst = filter(lambda x: '_RST_' in x or '_SPRL_' in x, ts)
        if len(ts_rst) > 1:
            ts_rst = filter(lambda x: '_RST_' in x, ts) # prefer EPI over SPRL

        # ensure right number of inputs
        try:
            assert_n(ts_imi, 1)
            db.loc[subject, 'ts_imi'] = ts_imi[0]
        except:
            print('{}:IMI n={}'.format(subject, len(ts_imi)))

        try:
            assert_n(ts_obs, 1)
            db.loc[subject, 'ts_obs'] = ts_obs[0]
        except:
            print('{}:OBS n={}'.format(subject, len(ts_obs)))

        try:
            assert_n(ts_imi_resid, 1)
            db.loc[subject, 'ts_imi_resid'] = ts_imi_resid[0]
        except:
            print('{}:IM_resid n={}'.format(subject, len(ts_imi_resid)))

        try:
            assert_n(ts_obs_resid, 1)
            db.loc[subject, 'ts_obs_resid'] = ts_obs_resid[0]
        except:
            print('{}:OB_resid n={}'.format(subject, len(ts_obs_resid)))

        try:
            assert_n(ts_imob_gm, 1)
            db.loc[subject, 'ts_imob_gm'] = ts_imob_gm[0]
        except:
            print('{}:IMOB_GM n={}'.format(subject, len(ts_imob_gm)))

        try:
            assert_n(ts_ea_gm, 1)
            db.loc[subject, 'ts_ea_gm'] = ts_ea_gm[0]
        except:
            print('{}:EA_GM n={}'.format(subject, len(ts_ea_gm)))

        try:
            assert_n(stat_imob_ch, 1)
            db.loc[subject, 'stat_imob_ch'] = stat_imob_ch[0]
        except:
            print('{}:IMOB_STAT_CH n={}'.format(subject, len(stat_imob_ch)))

        try:
            assert_n(stat_imob, 1)
            db.loc[subject, 'stat_imob'] = stat_imob[0]
        except:
            print('{}:IMOB_STAT n={}'.format(subject, len(stat_imob)))

        try:
            assert_n(stat_ea, 1)
            db.loc[subject, 'stat_ea'] = stat_ea[0]
        except:
            print('{}:EA_STAT n={}'.format(subject, len(stat_ea)))

        try:
            assert_n(ts_rst, 1)
            db.loc[subject, 'ts_rst'] = ts_rst[0]
        except:
            print('{}:RST n={}'.format(subject, len(ts_rst)))

        try:
            assert_n(ts_emp_vid, 1)
            db.loc[subject, 'ts_ea_vid_resid'] = ts_emp_vid[0]
        except:
            print('{}:EA_vid_resid n={}'.format(subject, len(ts_emp_vid)))

        try:
            assert_n(ts_emp_cvid, 1)
            db.loc[subject, 'ts_ea_cvid_resid'] = ts_emp_cvid[0]
        except:
            print('{}:EA_cvid_resid n={}'.format(subject, len(ts_emp_cvid)))

        try:
            assert_n(ts_emp, 3)
            ts_emp.sort()
            db.loc[subject, 'ts_ea1'] = ts_emp[0]
            db.loc[subject, 'ts_ea2'] = ts_emp[1]
            db.loc[subject, 'ts_ea3'] = ts_emp[2]
        except:
            print('{}:EMP n={}'.format(subject, len(ts_emp)))


    db.to_csv(output)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(__doc__)

