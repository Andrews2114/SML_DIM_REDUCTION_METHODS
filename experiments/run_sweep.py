import sys, json, time, itertools, numpy as np
import sweep as S, ms_tsne as M0
from sklearn.manifold import trustworthiness
from scipy.spatial import procrustes

def main(name, perps=(10,100), alphas=(0,0.09,0.25,0.5,0.75,1.0), seeds=5):
    X, co, fi = M0.DS[name](); n=len(X)
    # alpha = weight on the SMALL scale
    P1 = S.make_aff(X, perps, [1,0]).P; P2 = S.make_aff(X, perps, [0,1]).P
    out = dict(dataset=name, perps=list(perps), alphas=list(alphas), runs=[])
    for a in alphas:
        aff = S.make_aff(X, perps, [a, 1-a]); Ys=[]
        for s in range(seeds):
            Y = S.embed(X, aff, s); Ys.append(Y)
        nb=[S.overlap_sets(Y) for Y in Ys]
        pairs=list(itertools.combinations(range(seeds),2))
        stab=float(np.mean([M0.overlap(nb[i],nb[j]) for i,j in pairs]))
        recs=[]
        for s,Y in enumerate(Ys):
            c1=S.kl_cost(P1,Y); c2=S.kl_cost(P2,Y)
            recs.append(dict(seed=s, C1=c1, C2=c2, Cmix=a*c1+(1-a)*c2,
              trust=float(trustworthiness(X,Y,n_neighbors=15)),
              acc_c=M0.acc(Y,co), acc_f=M0.acc(Y,fi)))
        out['runs'].append(dict(alpha=a, stability=stab, recs=recs))
        r=recs; f=lambda k: np.nanmean([x[k] for x in r])
        print(name, 'alpha',a,'trust %.3f accF %.3f stab %.3f C1 %.3f C2 %.3f'%(f('trust'),f('acc_f'),stab,f('C1'),f('C2')), flush=True)
        np.save(f'sw_{name}_a{a}.npy', np.stack(Ys))
    json.dump(out, open(f'sweep_{name}.json','w'), indent=1)

if __name__=='__main__':
    main(sys.argv[1])
