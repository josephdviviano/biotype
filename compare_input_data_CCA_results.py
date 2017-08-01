#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

inputs = ['biotype-restconn_test-restconn/diagnosis_score_stats.csv',
          'biotype-restconn_test-restconn/biotype_score_stats.csv',
          'biotype-imobconn_test-imobconn/biotype_score_stats.csv',
          'biotype-eaconn_test-eaconn/biotype_score_stats.csv',
          'biotype-imobstat_test-imobstat/biotype_score_stats.csv',
          'biotype-eastat_test-eastat/biotype_score_stats.csv']
labels = ['Diagnostic Groups',
          'Biotype Groups: Resting State Connectivity',
          'Biotype Groups: Imitate/Observe Connectivity',
          'Biotype Groups: Empathic Accuracy Connectivity',
          'Biotype Groups: Imitate/Observe Task Activations',
          'Biotype Groups: Empathic Accuracy Task Activations']

xlabels = ['ER40 RT (inv)', 'Tasit 1', 'Tasit 2', 'Tasit 3', 'RMET', 'RAD',
           'Processing Speed', 'Working Memory', 'Verbal Learning', 'Visual Learning',
           'Reasoning', 'Attention/Vigilance']

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ts = []
for i, input_file in enumerate(inputs):
    print('loading: {}'.format(input_file))
    x = pd.read_csv(input_file)
    t = x['t'][0:12]
    ts.append(t)
    ax1.plot(t, label=labels[i])


cmap = plt.cm.Vega20c
colors = [cmap(i) for i in np.linspace(0, 1,len(ax1.lines))]
for i,j in enumerate(ax1.lines):
    j.set_color(colors[i])

ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=14)
ax1.set_xticks(range(0,12,1))
ax1.set_xticklabels(xlabels, rotation=45, ha='right')
ax1.set_xlabel('Cognitive Scores')
ax1.set_ylabel('t Score (Group Difference)')
fig1.savefig('cognitive_score_differences_bt_input_data.pdf')
