import json, os, numpy as np, openTSNE, warnings
warnings.filterwarnings('ignore')
from openTSNE import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import pbmc_regret as R, sweep as S
X=R.X; tr,te=R.train_idx,R.test_idx; Xtr,Xte=X[tr],X[te]
hd=NearestNeighbors(n_neighbors=15).fit(Xtr).kneighbors(Xte,return_distance=False)
d=pairwise_distances(Xte,Xtr,squared=True); idx=np.argsort(d,1)[:,:30]; dd=np.take_along_axis(d,idx,1)
w=np.exp(-dd/np.median(dd,1,keepdims=True)); w/=w.sum(1,keepdims=True)
for name,(perps,weights) in R.POL.items():
    if os.path.exists(f'pbmc/ho_{name}.json'): continue
    aff=S.make_aff(Xtr,perps,weights,seed=0); vals=[]
    for s in range(3):
        init=openTSNE.initialization.random(len(Xtr),random_state=s)
        Ytr=np.asarray(TSNE(n_jobs=2,random_state=s).fit(Xtr,affinities=aff,initialization=init))
        Yte=(Ytr[idx]*w[:,:,None]).sum(1)
        mp=NearestNeighbors(n_neighbors=15).fit(Ytr).kneighbors(Yte,return_distance=False)
        vals.append(R.overlap(hd,mp))
    json.dump(dict(policy=name,heldout_bary=vals),open(f'pbmc/ho_{name}.json','w'))
    print(name,np.round(np.mean(vals),3),flush=True)
