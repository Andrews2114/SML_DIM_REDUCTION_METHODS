import numpy as np
import pytest

from methods import get_method
from data import make_dataset
from metrics import dist_corr, rnx_curve, trust_continuity


def test_identity_embedding_is_perfect():
    X = make_dataset("swiss_roll", n=300).X
    tr, co = trust_continuity(X, X, k=10)
    _, rnx, auc = rnx_curve(X, X)
    assert tr == pytest.approx(1.0) and co == pytest.approx(1.0)
    assert auc == pytest.approx(1.0, abs=1e-6)
    assert dist_corr(X, X) == pytest.approx(1.0)


def test_random_embedding_is_near_chance():
    rng = np.random.default_rng(0)
    X = make_dataset("swiss_roll", n=400).X
    Z = rng.normal(size=(400, 2))
    _, _, auc = rnx_curve(X, Z)
    assert abs(auc) < 0.1                       # R_NX = 0 means "no better than random"


def test_pca_beats_random_on_swiss_roll():
    ds = make_dataset("swiss_roll", n=400)
    Z = get_method("pca").fit_transform(ds.X)
    _, _, auc = rnx_curve(ds.X, Z)
    assert auc > 0.1


def test_tsne_has_no_out_of_sample_transform():
    m = get_method("tsne")
    with pytest.raises(NotImplementedError):
        m.transform(np.zeros((2, 3)))


def test_linear_autoencoder_spans_pca_subspace():
    torch = pytest.importorskip("torch")
    from scipy.linalg import subspace_angles
    from sklearn.decomposition import PCA
    X = make_dataset("swiss_roll", n=1000, D=20, noise=0.05).X
    ae = get_method("ae", n_components=3, hidden=(), epochs=400, patience=400, lr=5e-3, val_frac=0.0)
    ae.fit(X)
    W = ae.enc[0].weight.detach().numpy()                     # (3, D) encoder weights, standardized space
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    V = PCA(3).fit(Xs).components_                            # (3, D)
    angles = subspace_angles(W.T, V.T)
    assert np.max(angles) < 0.3                               # radians; ~0 when subspaces coincide
