"""Embedding-quality metrics. X: (n, D) original data, Z: (n, d) embedding."""
import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def trust_continuity(X, Z, k=10):
    """Trustworthiness: are neighbours in Z also neighbours in X? (no false neighbours)
    Continuity: are neighbours in X still neighbours in Z? (no tears). Same formula, roles swapped."""
    return trustworthiness(X, Z, n_neighbors=k), trustworthiness(Z, X, n_neighbors=k)


def _rank_matrix(D):
    n = D.shape[0]
    order = np.argsort(D, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]
    return ranks                      # ranks[i, j] = rank of j among i's neighbours (self = 0)


def rnx_curve(X, Z):
    """Lee & Verleysen co-ranking. Returns (K, R_NX(K), AUC). Memory O(n^2): use n <= ~3000."""
    n = len(X)
    Rh = _rank_matrix(squareform(pdist(X)))
    Rl = _rank_matrix(squareform(pdist(Z)))
    off = ~np.eye(n, dtype=bool)
    rh, rl = Rh[off], Rl[off]                                   # values in 1..n-1
    Q = np.bincount((rh - 1) * (n - 1) + (rl - 1), minlength=(n - 1) ** 2).reshape(n - 1, n - 1)
    both = np.diag(Q.cumsum(0).cumsum(1))[: n - 2]              # pairs with both ranks <= K
    K = np.arange(1, n - 1)
    qnx = both / (K * n)                                        # fraction of K-neighbourhood kept
    rnx = ((n - 1) * qnx - K) / (n - 1 - K)                     # 0 = random, 1 = perfect
    auc = np.sum(rnx / K) / np.sum(1.0 / K)                     # log-scale weighting: favours local
    return K, rnx, auc


def dist_corr(X, Z):
    """Spearman correlation of pairwise distances: a *global* structure score."""
    return spearmanr(pdist(X), pdist(Z)).correlation


def intrinsic_r2(Z, T, k=10, cv=5):
    """How smoothly do the known intrinsic coordinates T (n, m) vary over the embedding?
    Cross-validated kNN-regression R^2 from Z to each column of T, averaged."""
    T = T.reshape(len(Z), -1)
    pred = cross_val_predict(KNeighborsRegressor(k), Z, T, cv=cv)
    return float(np.mean([r2_score(T[:, j], pred[:, j]) for j in range(T.shape[1])]))


def knn_acc(Z, y, k=10, cv=5):
    """Cross-validated kNN accuracy in the embedding (labels used for evaluation only)."""
    return float((cross_val_predict(KNeighborsClassifier(k), Z, y, cv=cv) == y).mean())


def procrustes_disparity(Z1, Z2):
    """Seed-to-seed instability: 0 = identical up to translation/rotation/scale."""
    return procrustes(Z1, Z2)[2]
