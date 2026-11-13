"""Autoencoder reducer (PyTorch) exposing the regularization knobs studied in E2/E4:
bottleneck size, width/depth, weight decay, dropout, input noise (denoising) and
early stopping on a held-out validation split."""
import copy

import numpy as np
import torch
from torch import nn


def _mlp(sizes, act, dropout):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:                       # no activation after the last layer
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class AEReducer:
    def __init__(self, n_components=2, hidden=(128, 64), activation="gelu", lr=1e-3,
                 weight_decay=0.0, dropout=0.0, noise_std=0.0, epochs=300,
                 batch_size=128, val_frac=0.1, patience=30, seed=0, device="cpu"):
        """hidden=() gives a *linear* autoencoder (one Linear layer each way): used to
        check that a linear AE spans the PCA subspace."""
        self.d, self.hidden, self.lr, self.wd = n_components, tuple(hidden), lr, weight_decay
        self.dropout, self.noise_std, self.epochs = dropout, noise_std, epochs
        self.bs, self.val_frac, self.patience = batch_size, val_frac, patience
        self.seed, self.device = seed, device
        self.act = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        self.history = {"train": [], "val": []}

    def _build(self, D):
        self.enc = _mlp([D, *self.hidden, self.d], self.act, self.dropout).to(self.device)
        self.dec = _mlp([self.d, *self.hidden[::-1], D], self.act, self.dropout).to(self.device)

    def _prep(self, X):
        return torch.tensor((X - self.mu) / self.sd, dtype=torch.float32, device=self.device)

    def fit(self, X):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = self._prep(X)
        perm = torch.tensor(rng.permutation(len(Xs)), device=self.device)
        n_val = int(self.val_frac * len(Xs))
        Xval, Xtr = Xs[perm[:n_val]], Xs[perm[n_val:]]

        self._build(Xs.shape[1])
        params = list(self.enc.parameters()) + list(self.dec.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = np.inf, None, 0

        for _ in range(self.epochs):
            self.enc.train(); self.dec.train()
            idx = torch.randperm(len(Xtr), device=self.device)
            tot = 0.0
            for i in range(0, len(Xtr), self.bs):
                xb = Xtr[idx[i:i + self.bs]]
                xin = xb + self.noise_std * torch.randn_like(xb) if self.noise_std > 0 else xb
                loss = ((self.dec(self.enc(xin)) - xb) ** 2).mean()   # target = CLEAN input
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(xb)
            self.history["train"].append(tot / len(Xtr))
            if n_val:
                self.enc.eval(); self.dec.eval()
                with torch.no_grad():
                    v = ((self.dec(self.enc(Xval)) - Xval) ** 2).mean().item()
                self.history["val"].append(v)
                if v < best - 1e-6:
                    best, bad = v, 0
                    best_state = (copy.deepcopy(self.enc.state_dict()),
                                  copy.deepcopy(self.dec.state_dict()))
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.enc.load_state_dict(best_state[0]); self.dec.load_state_dict(best_state[1])
        return self

    def transform(self, X):
        self.enc.eval()
        with torch.no_grad():
            return self.enc(self._prep(X)).cpu().numpy()

    def reconstruct(self, X):
        self.enc.eval(); self.dec.eval()
        with torch.no_grad():
            return self.dec(self.enc(self._prep(X))).cpu().numpy() * self.sd + self.mu

    def fit_transform(self, X):
        return self.fit(X).transform(X)
