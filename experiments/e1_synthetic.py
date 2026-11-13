"""E1: synthetic sanity check. Every method x dataset x seed, default hyperparameters.

Usage:  python experiments/e1_synthetic.py --n 1000 --seeds 3
Outputs (results/): e1_metrics.csv, e1_summary.csv, e1_grid.png, e1_rnx.png
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dimred import DATASETS, get_method, make_dataset
from dimred.metrics import (dist_corr, intrinsic_r2, knn_acc, procrustes_disparity,
                            rnx_curve, trust_continuity)

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=1000)
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--methods", nargs="+", default=["pca", "kpca", "isomap", "tsne", "umap", "ae"])
p.add_argument("--datasets", nargs="+", default=DATASETS)
p.add_argument("--out", default="results")
args = p.parse_args()
out = Path(args.out); out.mkdir(exist_ok=True)

rows, embeds, curves = [], {}, {}
for dname in args.datasets:
    ds = make_dataset(dname, n=args.n, seed=0)          # same data for all seeds: seeds = method randomness
    for mname in args.methods:
        for seed in range(args.seeds):
            try:
                m = get_method(mname, seed=seed)
            except ImportError as e:
                print(f"[skip] {mname}: {e}"); break
            t0 = time.time()
            Z = m.fit_transform(ds.X)
            dt = time.time() - t0
            embeds[(dname, mname, seed)] = Z
            tr, co = trust_continuity(ds.X, Z, k=10)
            K, rnx, auc = rnx_curve(ds.X, Z)
            if seed == 0: curves[(dname, mname)] = (K, rnx)
            row = dict(dataset=dname, method=mname, seed=seed, time_s=dt, trust=tr, cont=co,
                       rnx_auc=auc, dist_corr=dist_corr(ds.X, Z))
            if ds.T is not None: row["task"] = intrinsic_r2(Z, ds.T); row["task_name"] = "intrinsic_R2"
            elif ds.y is not None: row["task"] = knn_acc(Z, ds.y); row["task_name"] = "knn_acc"
            if seed > 0: row["procrustes_vs_seed0"] = procrustes_disparity(embeds[(dname, mname, 0)], Z)
            rows.append(row)
            print(f"{dname:13s} {mname:7s} seed={seed} trust={tr:.3f} cont={co:.3f} "
                  f"auc={auc:.3f} t={dt:.1f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(out / "e1_metrics.csv", index=False)
cols = ["trust", "cont", "rnx_auc", "dist_corr", "task", "procrustes_vs_seed0", "time_s"]
summ = df.groupby(["dataset", "method"])[[c for c in cols if c in df]].agg(["mean", "std"]).round(3)
summ.to_csv(out / "e1_summary.csv")
print("\n", summ.to_string())

# --- figure 1: embeddings (seed 0), coloured by intrinsic coordinate / label ---
methods = [m for m in args.methods if any(k[1] == m for k in embeds)]
fig, axs = plt.subplots(len(args.datasets), len(methods), figsize=(2.6 * len(methods), 2.4 * len(args.datasets)), squeeze=False)
for i, d in enumerate(args.datasets):
    ds = make_dataset(d, n=args.n, seed=0)
    for j, m in enumerate(methods):
        ax = axs[i, j]
        Z = embeds[(d, m, 0)]
        ax.scatter(Z[:, 0], Z[:, 1], c=ds.color, s=4, cmap="viridis", linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(m)
        if j == 0: ax.set_ylabel(d, fontsize=8)
plt.tight_layout(); plt.savefig(out / "e1_grid.png", dpi=150); plt.close()

# --- figure 2: R_NX(K) curves (log K axis): left = local, right = global ---
fig, axs = plt.subplots(1, len(args.datasets), figsize=(3.2 * len(args.datasets), 2.8), squeeze=False)
for j, d in enumerate(args.datasets):
    for m in methods:
        K, rnx = curves[(d, m)]
        axs[0, j].semilogx(K, rnx, label=m)
    axs[0, j].set_title(d, fontsize=9); axs[0, j].set_xlabel("K"); axs[0, j].set_ylim(-0.1, 1)
axs[0, 0].set_ylabel("$R_{NX}(K)$"); axs[0, -1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(out / "e1_rnx.png", dpi=150); plt.close()
print(f"\nSaved to {out}/")
