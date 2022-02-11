#merge with same number of events for each number

import os
import numpy as np

import sys

from simplestat import statinf
import json
from sklearn.metrics import roc_auc_score as auc

#combine only cmax models
cmax=1000
if len(sys.argv)>1:
    cmax=int(sys.argv[1])

if not os.path.isdir("results"):exit()

fns=["results/"+zw+"/result.npz" for zw in os.listdir("results")[:min([1000,cmax+10])]]#there is a limit of 1000 files open at the same time
fns=[zw for zw in fns if os.path.isfile(zw)]

fs=[np.load(fn,allow_pickle=True) for fn in fns]


y_true=fs[0]["y_true"]
y_scores=[f["y_score"] for f in fs]

y_scores=y_scores[:cmax]

y_scores=np.array(y_scores)

aucs=[auc(y_true,y_score) for y_score in y_scores]
print("single models:")
print(json.dumps(statinf(aucs),indent=2))


y_score=np.sqrt(np.mean(y_scores**2,axis=0))


auc_score=auc(y_true,y_score)

print("ensemble:",auc_score)






