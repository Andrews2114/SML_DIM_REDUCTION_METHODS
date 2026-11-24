import sys, json, time, itertools, numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import pairwise_distances
from scipy.spatial import procrustes
from openTSNE import TSNE

# ---------------- datasets: return X (n,d), coarse labels or None, fine labels
def ds_toy(seed=0, G=3, M=4, d=20, n_per=125, sg=2.0, ss=0.6, sig=0.9):
    r = np.random.RandomState(seed)
    cg = sg*r.randn(G, d); X=[]; coarse=[]; fine=[]
    for g in range(G):
        for m in range(M):
            c = cg[g] + ss*r.randn(d)
            X.append(c + sig*r.randn(n_per, d)); coarse += [g]*n_per; fine += [g*M+m]*n_per
    return np.vstack(X), np.array(coarse), np.array(fine)

def ds_sim_scrna(seed=0, L=4, S=4, genes=2000, n_per=150):
    r = np.random.RandomState(seed)
    base = r.normal(-1.0, 1.0, genes)               # log mean expression
    lin = np.zeros((L, genes)); sub = np.zeros((L*S, genes))
    for l in range(L): lin[l, r.choice(genes, 120, replace=False)] = 1.5
    for k in range(L*S): sub[k, r.choice(genes, 40, replace=False)] = 0.8
    X=[]; coarse=[]; fine=[]
    for l in range(L):
        for s in range(S):
            k=l*S+s
            mu = np.exp(base + lin[l] + sub[k])
            lib = np.exp(r.normal(0, 0.3, (n_per,1)))
            rate = mu[None,:]*lib*3
            theta = 2.0
            lam = r.gamma(theta, rate/theta)
            X.append(r.poisson(lam)); coarse += [l]*n_per; fine += [k]*n_per
    C = np.vstack(X).astype(float)
    C = C/ C.sum(1, keepdims=True)*1e4
    C = np.log1p(C)
    hv = np.argsort(C.var(0))[::-1][:1000]
    C = C[:, hv]; C -= C.mean(0)
    Xp = PCA(50, random_state=0).fit_transform(C)
    return Xp, np.array(coarse), np.array(fine)

def ds_pbmc():
    import scanpy as sc
    a = sc.datasets.pbmc68k_reduced()
    X = PCA(50, random_state=0).fit_transform(np.asarray(a.X))
    bl = a.obs['bulk_labels'].astype(str).values
    def lineage(s):
        if 'Monocyte' in s or 'Dendritic' in s: return 'myeloid'
        if 'CD19' in s: return 'B'
        if 'NK' in s: return 'NK'
        if 'CD34' in s: return 'progenitor'
        return 'T'
    co = np.array([lineage(s) for s in bl])
    return X, np.unique(co, return_inverse=True)[1], np.unique(bl, return_inverse=True)[1]

def ds_digits():
    from sklearn.datasets import load_digits
    d = load_digits(); return d.data/16.0, None, d.target

DS = dict(toy=ds_toy, scrna=ds_sim_scrna, pbmc=ds_pbmc, digits=ds_digits)

# ---------------- mean JS between scales (conditional rows)
def cond_P(X, perp):
    from sklearn.manifold._utils import _binary_search_perplexity
    D = pairwise_distances(X, squared=True).astype(np.float32)
    return _binary_search_perplexity(D, perp, 0)
def mean_js(X, perps):
    Ps = [cond_P(X, p) for p in perps]; Pb = sum(Ps)/len(Ps)
    H = lambda P: -(P*np.log(np.clip(P,1e-300,None))).sum(1)
    return float((H(Pb) - sum(H(P) for P in Ps)/len(Ps)).mean())

# ---------------- metrics
def knn_sets(Y, k=15):
    nn = NearestNeighbors(n_neighbors=k+1).fit(Y); idx = nn.kneighbors(Y, return_distance=False)[:,1:]
    return idx
def overlap(a, b):
    return float(np.mean([len(set(x)&set(y))/len(x) for x,y in zip(a,b)]))
def acc(Y, lab, k=10):
    if lab is None: return np.nan
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(KNeighborsClassifier(k), Y, lab, cv=cv); return float((p==lab).mean())

METHODS = {'S10':[10], 'S30':[30], 'S100':[100], 'M10+100':[10,100], 'M10+30+100':[10,30,100]}

def run(name, seeds=4):
    X, co, fi = DS[name]()
    n = len(X); out = {'dataset':name, 'n':n, 'js':{}, 'methods':{}}
    for m, ps in METHODS.items():
        if len(ps)>1: out['js'][m] = mean_js(X, ps)
    embs = {}
    for m, ps in METHODS.items():
        Ys=[]; t0=time.time()
        for s in range(seeds):
            tsne = TSNE(perplexity=(ps if len(ps)>1 else ps[0]), initialization='random', random_state=s, n_jobs=2, verbose=False)
            Ys.append(np.asarray(tsne.fit(X)))
        embs[m]=Ys
        nb = [knn_sets(Y) for Y in Ys]
        pairs = list(itertools.combinations(range(seeds),2))
        stab = np.mean([overlap(nb[i],nb[j]) for i,j in pairs])
        proc = np.mean([procrustes(Ys[i],Ys[j])[2] for i,j in pairs])
        tr = [trustworthiness(X, Y, n_neighbors=15) for Y in Ys]
        ac = [acc(Y, co) for Y in Ys]; af = [acc(Y, fi) for Y in Ys]
        out['methods'][m] = dict(trust=[float(np.mean(tr)), float(np.std(tr))],
            acc_coarse=[float(np.nanmean(ac)), float(np.nanstd(ac))], acc_fine=[float(np.mean(af)), float(np.std(af))],
            nbr_stability=float(stab), procrustes=float(proc), secs=time.time()-t0)
        print(name, m, json.dumps(out['methods'][m]), flush=True)
    json.dump(out, open(f'res_{name}.json','w'), indent=1)
    np.savez(f'emb_{name}.npz', co=(co if co is not None else np.zeros(n)), fi=fi,
             **{f'{m}__{s}':embs[m][s] for m in embs for s in range(seeds)})
    print('done', name, 'js', out['js'], flush=True)

if __name__=='__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 4)
