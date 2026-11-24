# Pre-registered decision rules (written before running the PBMC 3k experiment)
Data: PBMC 3k (2,638 cells after QC), PCA-50; fine labels = Leiden (res 1.5, 12 clusters); coarse = marker-based lineage (4).
Primary multiscale policy (fixed in advance, no label tuning): MS-E{10,100}, weights alpha ∝ perplexity.
Exploratory: MS-U{10,100} (uniform) and MS-E{5,30,200}.
Baselines: default S30; oracle (best single perplexity per metric, uses labels: upper bound only); label-free selection rules
 (max seed-stability; BIC-like 2*KL + ln(n)*Perp/n; max held-out neighbour preservation).
Metrics: trustworthiness@15, kNN preservation@15, fine-label kNN accuracy, cluster-centroid distance correlation (CPD), held-out kNN preservation, seed stability.
Regret(policy, metric) = oracle best single value - policy value.
SURVIVES if MS-E{10,100} (a) beats default S30 on >= 3 of the 5 non-stability metrics AND (b) is not worse than the best label-free
selection rule on a majority of metrics AND (c) paired per-seed differences vs S30 exceed 2 standard errors on >= 2 metrics.
PARTIAL if it improves only stability or held-out preservation. Otherwise: does not survive as a better policy.
