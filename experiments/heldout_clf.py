"""Held-out classification test: single vs multiscale t-SNE (t-SNE only).
Fit t-SNE on 80% of PBMC 3k, place the 20% held-out cells, classify them by kNN
(k=15) over the *training embedding* using coarse lineage labels.
Extensions: (bary) 30-NN barycentric in PCA space; (otsne) openTSNE .transform.
"""
import json, os, sys, time, numpy as np, openTSNE, warnings
warnings.filterwarnings('ignore')
from openTSNE import TSNE
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import pairwise_distances, balanced_accuracy_score, accuracy_score
import sweep as S

D = np.load('pbmc/pbmc3k_prep.npz', allow_pickle=True)
X, y = D['X'], D['coarse']; n = len(X)
rng = np.random.RandomState(123); perm = rng.permutation(n)
te = np.sort(perm[:n//5]); tr = np.sort(perm[n//5:])
Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
w_ = lambda ps: [p/sum(ps) for p in ps]
POL = {'S10': ([10],[1.]), 'S30': ([30],[1.]), 'S100': ([100],[1.]), 'S200': ([200],[1.]),
       'MS-U{10,100}': ([10,100],[.5,.5]), 'MS-E{10,100}': ([10,100], w_([10,100]))}
SEEDS = int(os.environ.get('SEEDS', 5))
# barycentric weights
d = pairwise_distances(Xte, Xtr, squared=True); idx = np.argsort(d,1)[:,:30]
dd = np.take_along_axis(d,idx,1); bw = np.exp(-dd/np.median(dd,1,keepdims=True)); bw /= bw.sum(1,keepdims=True)

def score(Ytr, Yte):
    clf = KNeighborsClassifier(15).fit(Ytr, ytr); p = clf.predict(Yte)
    return accuracy_score(yte,p), balanced_accuracy_score(yte,p)

def run(name):
    out = f'pbmc/clf_{name}.json'
    if os.path.exists(out): return
    perps, wts = POL[name]; aff = S.make_aff(Xtr, perps, wts, seed=0); rows = []
    for s in range(SEEDS):
        init = openTSNE.initialization.random(len(Xtr), random_state=s)
        emb = TSNE(n_jobs=2, random_state=s).fit(Xtr, affinities=aff, initialization=init)
        Ytr = np.asarray(emb)
        Yb = (Ytr[idx]*bw[:,:,None]).sum(1)
        a1, b1 = score(Ytr, Yb)
        try:
            Yo = np.asarray(emb.transform(Xte)); a2, b2 = score(Ytr, Yo)
            rng_ratio = float(np.abs(Yo).max()/np.abs(Ytr).max())
        except Exception as e:
            a2 = b2 = rng_ratio = float('nan')
        # train-set leave-one-out kNN acc in embedding (in-sample reference)
        from sklearn.model_selection import cross_val_score
        cv = float(cross_val_score(KNeighborsClassifier(15), Ytr, ytr, cv=5).mean())
        rows.append(dict(seed=s, bary_acc=a1, bary_bal=b1, ot_acc=a2, ot_bal=b2, ot_range_ratio=rng_ratio, train_cv_acc=cv))
    json.dump(dict(policy=name, rows=rows), open(out,'w'))
    print(name, {k: round(float(np.nanmean([r[k] for r in rows])),3) for k in rows[0] if k!='seed'}, flush=True)

if __name__ == '__main__':
    for nm in (sys.argv[1:] or POL): run(nm)
