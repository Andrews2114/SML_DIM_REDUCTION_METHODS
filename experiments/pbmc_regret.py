import sys, json, time, itertools, numpy as np, os, warnings
warnings.filterwarnings('ignore')
import openTSNE
from openTSNE import TSNE
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
import sweep as S, ms_tsne as M0

D = np.load('pbmc/pbmc3k_prep.npz', allow_pickle=True)
X, fine, coarse = D['X'], D['fine'], D['coarse']; n = len(X)
GRID = [5,10,20,30,50,100,200]
POL = {f'S{p}': ([p],[1.0]) for p in GRID}
w = lambda ps: list(np.array(ps)/np.sum(ps))               # equal-influence weights  alpha ~ Perp
POL['MS-U{10,100}'] = ([10,100],[.5,.5])
POL['MS-E{10,100}'] = ([10,100], w([10,100]))             # primary pre-registered multiscale policy
POL['MS-E{5,30,200}'] = ([5,30,200], w([5,30,200]))
SEEDS = 6; HSEEDS = 3
rng = np.random.RandomState(123); perm = rng.permutation(n); test_idx = np.sort(perm[:n//5]); train_idx = np.sort(perm[n//5:])

def knn_idx(A, k=15, self_ex=True):
    nn = NearestNeighbors(n_neighbors=k+1).fit(A); return nn.kneighbors(A, return_distance=False)[:,1:]
HD_NN = knn_idx(X)
cent_hd = np.array([X[fine==c].mean(0) for c in np.unique(fine)])
def cpd(Y):
    cm = np.array([Y[fine==c].mean(0) for c in np.unique(fine)])
    return float(spearmanr(pdist(cent_hd), pdist(cm))[0])
def overlap(a, b): return float(np.mean([len(set(x)&set(y))/len(x) for x,y in zip(a,b)]))

def fit_full(aff, seed):
    init = openTSNE.initialization.random(n, random_state=seed)
    return np.asarray(TSNE(n_jobs=2, random_state=seed, verbose=False).fit(X, affinities=aff, initialization=init))

def heldout(perps, weights, seed):
    Xtr, Xte = X[train_idx], X[test_idx]
    aff = S.make_aff(Xtr, perps, weights, seed=0)
    init = openTSNE.initialization.random(len(Xtr), random_state=seed)
    emb = TSNE(n_jobs=2, random_state=seed, verbose=False).fit(Xtr, affinities=aff, initialization=init)
    Yte = np.asarray(emb.transform(Xte)); Ytr = np.asarray(emb)
    hd = NearestNeighbors(n_neighbors=15).fit(Xtr).kneighbors(Xte, return_distance=False)
    mp = NearestNeighbors(n_neighbors=15).fit(Ytr).kneighbors(Yte, return_distance=False)
    return overlap(hd, mp)

def run(name):
    perps, weights = POL[name]; t0 = time.time()
    aff = S.make_aff(X, perps, weights, seed=0)
    Ys = [fit_full(aff, s) for s in range(SEEDS)]
    nb = [knn_idx(Y) for Y in Ys]
    stab = float(np.mean([overlap(nb[i], nb[j]) for i,j in itertools.combinations(range(SEEDS),2)]))
    recs = []
    for s, Y in enumerate(Ys):
        recs.append(dict(seed=s, trust=float(trustworthiness(X, Y, n_neighbors=15)), knnpres=overlap(HD_NN, nb[s]),
            acc_fine=M0.acc(Y, fine), acc_coarse=M0.acc(Y, coarse), cpd=cpd(Y), kl=S.kl_cost(aff.P, Y)))
    ho = [heldout(perps, weights, s) for s in range(HSEEDS)]
    out = dict(policy=name, perps=perps, weights=weights, stability=stab, recs=recs, heldout=ho, secs=time.time()-t0)
    json.dump(out, open(f'pbmc/res_{name}.json','w'), indent=1)
    np.save(f'pbmc/emb_{name}.npy', np.stack(Ys))
    m = lambda k: np.mean([r[k] for r in recs])
    print(f"{name:16s} trust {m('trust'):.3f} knn {m('knnpres'):.3f} accF {m('acc_fine'):.3f} accC {m('acc_coarse'):.3f} cpd {m('cpd'):.3f} stab {stab:.3f} held {np.mean(ho):.3f}  [{time.time()-t0:.0f}s]", flush=True)

if __name__=='__main__':
    for name in (sys.argv[1:] or list(POL)):
        if not os.path.exists(f'pbmc/res_{name}.json'): run(name)
