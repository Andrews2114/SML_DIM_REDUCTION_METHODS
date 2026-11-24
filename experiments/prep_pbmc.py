import numpy as np, scanpy as sc, warnings, json
warnings.filterwarnings('ignore')
a = sc.read_h5ad('../data/pbmc3k.h5ad')
print('raw', a.shape)
sc.pp.filter_cells(a, min_genes=200); sc.pp.filter_genes(a, min_cells=3)
a.var['mt'] = np.asarray(a.var_names.str.startswith('MT-'), dtype=bool)
sc.pp.calculate_qc_metrics(a, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
a = a[(a.obs.n_genes_by_counts < 2500) & (a.obs.pct_counts_mt < 5)].copy()
print('after QC', a.shape)
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
a.raw = a
sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor='seurat')
b = a[:, a.var.highly_variable].copy()
sc.pp.scale(b, max_value=10); sc.tl.pca(b, n_comps=50, random_state=0)
sc.pp.neighbors(b, n_neighbors=15, n_pcs=50, random_state=0)
sc.tl.leiden(b, resolution=1.5, random_state=0, flavor='igraph', n_iterations=2, directed=False)
b.obs['leiden'] = b.obs['leiden'].astype(str)
# marker-based lineage per cluster (cluster-level mean scores on log-normalised data)
markers = {'T/NK':['CD3D','CD3E','IL7R','CD8A','NKG7','GNLY','CCL5'],
           'B':['MS4A1','CD79A','CD79B'],
           'Myeloid':['CD14','LYZ','FCGR3A','FCER1A','CST3','S100A8','S100A9'],
           'Megakaryocyte':['PPBP','PF4']}
for k,g in markers.items():
    g = [x for x in g if x in a.var_names]
    sc.tl.score_genes(a, g, score_name='sc_'+k, use_raw=False)
sc_cols = ['sc_'+k for k in markers]
a.obs['leiden'] = b.obs['leiden'].reindex(a.obs_names)
cl = a.obs.groupby('leiden')[sc_cols].mean()
lin = cl.idxmax(axis=1).str.replace('sc_','')
print(cl.round(2)); print(lin.to_dict()); print(a.obs['leiden'].value_counts().sort_index().to_dict())
coarse_lab = a.obs['leiden'].map(lin)
X = b.obsm['X_pca'].astype(np.float64)
np.savez('pbmc/pbmc3k_prep.npz', X=X, fine=b.obs['leiden'].astype(int).values,
         coarse=np.unique(coarse_lab.values, return_inverse=True)[1],
         coarse_names=np.unique(coarse_lab.values))
print('coarse counts', coarse_lab.value_counts().to_dict(), 'X', X.shape)
