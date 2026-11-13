"""Synthetic datasets with known ground truth.

Every dataset is a low-dimensional structure (known intrinsic coordinates T and/or
labels y) lifted into D dimensions by a random rotation plus isotropic noise, so that
"high-dimensional" methods have something non-trivial to do.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_s_curve, make_swiss_roll


@dataclass
class Dataset:
    name: str
    X: np.ndarray                      # (n, D) observed high-dimensional data
    color: np.ndarray                  # (n,) values used only for plotting
    T: Optional[np.ndarray] = None     # (n, k) intrinsic coordinates, if they exist
    y: Optional[np.ndarray] = None     # (n,) class labels, if they exist


def lift(X_low, D, noise, rng):
    """Embed (n, d) data in R^D with a random isometry and add isotropic Gaussian noise."""
    d = X_low.shape[1]
    Q, _ = np.linalg.qr(rng.normal(size=(D, d)))     # (D, d), orthonormal columns
    return X_low @ Q.T + noise * rng.normal(size=(X_low.shape[0], D))


def make_dataset(name, n=1000, D=50, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    if name == "swiss_roll":
        X, t = make_swiss_roll(n, noise=0.0, random_state=seed)
        return Dataset(name, lift(X, D, noise, rng), t, T=np.c_[t, X[:, 1]])
    if name == "s_curve":
        X, t = make_s_curve(n, noise=0.0, random_state=seed)
        return Dataset(name, lift(X, D, noise, rng), t, T=np.c_[t, X[:, 1]])
    if name == "circles":
        X, y = make_circles(n, factor=0.4, noise=0.03, random_state=seed)
        return Dataset(name, lift(X, D, noise, rng), y.astype(float), y=y)
    if name == "blobs_varied":
        # clusters with very different spreads: sizes/distances in an embedding are
        # NOT expected to be meaningful for t-SNE/UMAP -> tests what can be trusted
        X, y = make_blobs(n, n_features=5, centers=3, cluster_std=[0.5, 1.5, 3.0],
                          random_state=seed)
        return Dataset(name, lift(X, D, noise, rng), y.astype(float), y=y)
    if name == "noise":
        # no structure at all: any "cluster" in the embedding is an artifact
        X = rng.normal(size=(n, 10))
        return Dataset(name, lift(X, D, noise, rng), X[:, 0])
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")


DATASETS = ["swiss_roll", "s_curve", "circles", "blobs_varied", "noise"]
