"""Uniform interface: get_method(name, seed=..., **params) returns an object with
fit_transform(X) and transform(X). transform raises NotImplementedError for transductive
methods (t-SNE), which matters for the out-of-sample experiment (E4)."""
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap


class _SK:
    def __init__(self, model, has_transform=True):
        self.model, self.has_transform = model, has_transform

    def fit_transform(self, X):
        return self.model.fit_transform(X)

    def transform(self, X):
        if not self.has_transform:
            raise NotImplementedError(f"{type(self.model).__name__} has no out-of-sample transform")
        return self.model.transform(X)


def get_method(name, seed=0, n_components=2, **params):
    if name == "pca":
        return _SK(PCA(n_components, random_state=seed, **params))
    if name == "kpca":
        return _SK(KernelPCA(n_components, **{"kernel": "rbf", "random_state": seed, **params}))
    if name == "isomap":
        return _SK(Isomap(n_components=n_components, **{"n_neighbors": 10, **params}))
    if name == "tsne":
        p = {"perplexity": 30, "init": "pca", **params}
        return _SK(TSNE(n_components, random_state=seed, **p), has_transform=False)
    if name == "umap":
        import umap
        p = {"n_neighbors": 15, "min_dist": 0.1, **params}
        return _SK(umap.UMAP(n_components=n_components, random_state=seed, **p))
    if name == "ae":
        from .autoencoder import AEReducer
        return AEReducer(n_components=n_components, seed=seed, **params)
    raise ValueError(f"unknown method {name!r}; choose from {METHODS}")


METHODS = ["pca", "kpca", "isomap", "tsne", "umap", "ae"]
