import sys, json, time, itertools, numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from scipy.spatial import procrustes
import openTSNE
from openTSNE import TSNE, affinity
from openTSNE.affinity import Multiscale
import ms_tsne as M0

class WMS(Multiscale):
    weights = None
    @staticmethod
    def _calculate_P(neighbors, distances, perplexities, symmetrize=True, normalization="pair-wise",
                     n_reference_samples=None, n_jobs=1):
        Ps = [affinity.joint_probabilities_nn(neighbors, distances, [p], symmetrize=symmetrize,
              normalization=normalization, n_reference_samples=n_reference_samples, n_jobs=n_jobs)
              for p in perplexities]
        P = sum(w*Pi for w, Pi in zip(WMS.weights, Ps)); P = P / P.sum(); return P

def make_aff(X, perps, weights, seed=0):
    WMS.weights = list(weights)
    return WMS(X, perplexities=list(perps), n_jobs=2, random_state=seed)

def embed(X, aff, seed):
    init = openTSNE.initialization.random(X.shape[0], random_state=seed)
    t = TSNE(n_jobs=2, random_state=seed, verbose=False)
    return np.asarray(t.fit(X, affinities=aff, initialization=init))

def kl_cost(P, Y):
    P = sp.coo_matrix(P); d2 = pairwise_distances(Y, squared=True)
    w = 1.0/(1.0+d2); np.fill_diagonal(w, 0.0); Z = w.sum()
    q = w[P.row, P.col]/Z; p = P.data
    return float(np.sum(p*np.log(p/q)))

def overlap_sets(Y, k=15): return M0.knn_sets(Y, k)
