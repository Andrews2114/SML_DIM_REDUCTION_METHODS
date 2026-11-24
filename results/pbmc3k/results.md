# Multiscale t-SNE on PBMC 3k: does the idea survive?

**Verdict (against the rules in PREREGISTERED.md): it does NOT survive as a better policy; at most PARTIAL.**
The primary multiscale policy (perplexities 10 and 100, weights proportional to perplexity) beat the default perplexity 30 on 2 of 5 metrics
(global geometry and held-out preservation) and lost significantly on the other 3 (trustworthiness, kNN preservation, fine-cluster accuracy). Rule (a) needed >= 3 of 5.

## What the numbers show (n = 2,638 cells, 6 seeds; means)
| policy | trust | kNN pres | fine acc | centroid dist corr | held-out | stability |
|---|---|---|---|---|---|---|
| S30 (default) | 0.895 | 0.211 | 0.875 | 0.560 | 0.106 | 0.477 |
| S100 | 0.890 | 0.153 | 0.831 | 0.651 | 0.137 | 0.574 |
| MS-U{10,100} (uniform) | 0.890 | 0.202 | 0.877 | 0.570 | 0.099 | 0.472 |
| MS-E{10,100} (primary) | 0.892 | 0.166 | 0.850 | 0.630 | 0.133 | 0.534 |
| MS-E{5,30,200} | 0.887 | 0.140 | 0.827 | 0.637 | 0.137 | 0.641 |

* Single perplexity trades local fidelity (best near 20-30) against global fidelity, held-out smoothness and stability (best at 100-200).
* Every multiscale policy lands ON that single-perplexity trade-off curve (see pbmc_regret_curves.png), not above it.
* The effective-perplexity formula e^{JS} * prod(Perp^alpha) predicts where: uniform {10,100} -> 37.7 (single perplexities with matching kNN preservation / centroid correlation: 38 / 34);
  primary {10,100} -> 85.8 (matched: 66-86).

## Caveats (read these)
* One dataset. Fine labels are Leiden clusters computed on the same PCA space (circular, favours local metrics); coarse labels use marker genes.
* The three label-free selection rules (max stability, BIC-like 2*KL + ln(n)*Perp/n as I remember it, max held-out preservation) ALL picked the largest grid value (200),
  because those quantities increase with perplexity. They are not informative here; the BIC formula should be checked against the paper.
* PROTOCOL CHANGE after seeing results: the pre-registered held-out metric used openTSNE's transform, which returned invalid positions for all policies
  (points scattered far outside the training map, overlap ~ chance). It was replaced by a barycentric out-of-sample extension (same for every policy).
* Grid up to 200 only; N/100 (~26) from the single-cell recipe is not a large-perplexity regime at this n.

## Run
From `experiments/`: `mkdir pbmc`, `python prep_pbmc.py` (reads ../data/pbmc3k.h5ad; needs scanpy, leidenalg, igraph),
then `python pbmc_regret.py`, `python pbmc_heldout.py`, `python pbmc_analyse.py`. Needs openTSNE; also sweep.py and ms_tsne.py in the same folder.
