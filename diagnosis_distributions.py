#!/usr/env/python

clusters = mdl['clusters']
diagnosis = merged['Diagnosis']

for clst in np.unique(clusters):
    idx = np.where(clusters == clst)[0]
    dx = diagnosis[idx]
    plt.hist(dx)
    plt.show()
