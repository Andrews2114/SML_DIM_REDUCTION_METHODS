import json, glob, numpy as np, os
import pandas as pd
G=[5,10,20,30,50,100,200]
names=['S%d'%p for p in G]+['MS-U{10,100}','MS-E{10,100}','MS-E{5,30,200}']
R={n:json.load(open(f'pbmc/res_{n}.json')) for n in names}
HO={n:np.mean(json.load(open(f'pbmc/ho_{n}.json'))['heldout_bary']) if os.path.exists(f'pbmc/ho_{n}.json') else np.nan for n in names}
mets=['trust','knnpres','acc_fine','cpd']
rows=[]
for n in names:
    r=R[n]; row=dict(policy=n)
    for m in mets: row[m]=np.mean([x[m] for x in r['recs']]); row[m+'_sd']=np.std([x[m] for x in r['recs']])
    row['heldout']=HO[n]; row['stability']=r['stability']; row['kl']=np.mean([x['kl'] for x in r['recs']]); rows.append(row)
df=pd.DataFrame(rows).set_index('policy'); n_cells=2638
# label-free selection rules among singles
sing=df.loc[['S%d'%p for p in G]].copy(); sing['perp']=G
sing['bic']=2*sing['kl']+np.log(n_cells)*sing['perp']/n_cells
sel={'stability':sing['stability'].idxmax(),'BIC-like':sing['bic'].idxmin()}
if sing['heldout'].notna().all(): sel['held-out']=sing['heldout'].idxmax()
allm=mets+['heldout']
oracle={m:sing[m].max() for m in allm}
out=[]
def reg(p): return {m:oracle[m]-df.loc[p,m] for m in allm}
pol=[('default S30','S30')]+[(f'sel:{k}',v) for k,v in sel.items()]+[('MS-U{10,100}','MS-U{10,100}'),('MS-E{10,100} (primary)','MS-E{10,100}'),('MS-E{5,30,200}','MS-E{5,30,200}')]
print('oracle (best single) per metric:',{m:round(v,3) for m,v in oracle.items()})
print('label-free selections:',sel)
tab=pd.DataFrame({lab:reg(p) for lab,p in pol}).T
tab['stability']=[df.loc[p,'stability'] for _,p in pol]
print('\nREGRET vs best single (lower=better; stability shown raw, higher=better)'); print(tab.round(3).to_string())
df.round(4).to_csv('pbmc/summary_by_policy.csv'); tab.round(4).to_csv('pbmc/regret_table.csv')
# paired seed differences MS-E{10,100} - S30
print('\nPAIRED per-seed differences (primary multiscale minus default S30): mean, se, z')
prim='MS-E{10,100}'
survive_counts=0; z2=0
for m in ['trust','knnpres','acc_fine','cpd']:
    d=np.array([a[m]-b[m] for a,b in zip(R[prim]['recs'],R['S30']['recs'])]); se=d.std(ddof=1)/np.sqrt(len(d)); z=d.mean()/se
    better=d.mean()>0; survive_counts+=better; z2+= (better and z>2)
    print(f'  {m:9s} {d.mean():+.4f} se {se:.4f} z {z:+.1f}')
dho=np.array(json.load(open(f'pbmc/ho_{prim}.json'))['heldout_bary'])-np.array(json.load(open('pbmc/ho_S30.json'))['heldout_bary']) if os.path.exists(f'pbmc/ho_{prim}.json') else None
if dho is not None:
    se=dho.std(ddof=1)/np.sqrt(len(dho)); z=dho.mean()/se if se>0 else np.nan; better=dho.mean()>0; survive_counts+=better; z2+=(better and z>2)
    print(f'  heldout   {dho.mean():+.4f} se {se:.4f} z {z:+.1f}')
print('\nmetrics where primary multiscale beats S30:',int(survive_counts),'of 5; with z>2:',int(z2))
# vs best label-free selection (majority of metrics)
for k,v in sel.items():
    wins=sum(df.loc[prim,m]>=df.loc[v,m] for m in allm if not np.isnan(df.loc[v,m]))
    print(f'  vs sel:{k} ({v}): primary >= on {wins} of {len(allm)} metrics')
