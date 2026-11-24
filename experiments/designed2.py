import sys, json, itertools, numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
import sweep as S, ms_tsne as M0
from sklearn.manifold import trustworthiness

def make(seed=0, d=30, K=6, n_blob=100, delta=7.0, tiny_at=(1,2,3,4), n_tiny=8, off=3.0, sig_tiny=0.6):
    r=np.random.RandomState(seed); u=r.randn(d); u/=np.linalg.norm(u)
    X=[]; blob=[]; tiny=[]
    for k in range(K):
        X.append(k*delta*u + r.randn(n_blob,d)); blob+=[k]*n_blob; tiny+=[-1]*n_blob
    for t,k in enumerate(tiny_at):
        v=r.randn(d); v-=v.dot(u)*u; v/=np.linalg.norm(v)
        X.append(k*delta*u+off*v + sig_tiny*r.randn(n_tiny,d)); blob+=[-1]*n_tiny; tiny+=[t]*n_tiny
    return np.vstack(X), np.array(blob), np.array(tiny), K

def metrics(Y, blob, tiny, K):
    cen=np.array([Y[blob==k].mean(0) for k in range(K)])
    iu=np.triu_indices(K,1)
    D=np.linalg.norm(cen[:,None]-cen[None],axis=2)[iu]
    true=np.abs(np.arange(K)[:,None]-np.arange(K)[None])[iu]
    order=spearmanr(D,true)[0]
    nn=NearestNeighbors(n_neighbors=6).fit(Y); idx=nn.kneighbors(Y,return_distance=False)[:,1:]
    tm=tiny>=0
    tp=float(np.mean([(tiny[idx[i]]==tiny[i]).mean() for i in np.where(tm)[0]]))
    bm=blob>=0
    bp=float(np.mean([(blob[idx[i]]==blob[i]).mean() for i in np.where(bm)[0]]))
    return order, tp, bp

def main(seeds=8, perps=(5,150), alphas=(0,0.03,0.25,0.5,0.75,1.0)):
    X,blob,tiny,K=make(); n=len(X)
    P1=S.make_aff(X,perps,[1,0]).P; P2=S.make_aff(X,perps,[0,1]).P
    res=[]
    def run(label, aff, a=None):
        recs=[]; Ys=[]
        for s in range(seeds):
            Y=S.embed(X,aff,s); Ys.append(Y)
            o,tp,bp=metrics(Y,blob,tiny,K)
            c1=S.kl_cost(P1,Y); c2=S.kl_cost(P2,Y)
            recs.append(dict(order=o,tiny=tp,blobp=bp,C1=c1,C2=c2))
        f=lambda k:(float(np.mean([r[k] for r in recs])),float(np.std([r[k] for r in recs])))
        res.append(dict(label=label,alpha=a,order=f('order'),tiny=f('tiny'),blobp=f('blobp'),C1=f('C1'),C2=f('C2')))
        print(label, {k:np.round(v[0],3) for k,v in res[-1].items() if isinstance(v,tuple)}, flush=True)
        np.save(f'des2_{label}.npy', np.stack(Ys))
    for pp in (5,30,150):
        run(f'S{pp}', S.make_aff(X,[pp],[1.0]))
    for a in alphas:
        run(f'a{a}', S.make_aff(X,perps,[a,1-a]), a)
    json.dump(res, open('designed2.json','w'), indent=1)
    np.savez('des2_labels.npz', blob=blob, tiny=tiny)
if __name__=='__main__': main()
