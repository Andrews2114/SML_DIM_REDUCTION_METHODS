# Nonlinear dimensionality reduction: t-SNE vs UMAP vs autoencoders

Course project. Question: *what structure does each method preserve, how sensitive is it to
hyperparameters / initialization / seed, and how well does it generalize to unseen data?*

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q                                   # sanity tests (metrics + methods)
python experiments/e1_synthetic.py --n 1000 --seeds 3
```

## Layout
- `src/dimred/data.py`: synthetic datasets with known ground truth (lifted to 50-D with noise)
- `src/dimred/methods.py`: `get_method(name, seed, **params)` -> `fit_transform` / `transform`
- `src/dimred/autoencoder.py`: PyTorch AE (weight decay, dropout, denoising, early stopping)
- `src/dimred/metrics.py`: trustworthiness, continuity, R_NX(K) AUC, distance correlation,
  intrinsic-coordinate R^2, kNN accuracy, Procrustes (seed instability)
- `experiments/`: one script per experiment (E1 = synthetic sanity check); outputs to `results/`

## Experiments
E1 synthetic sanity, E2 hyperparameter sensitivity, E3 quantitative quality, E4 out-of-sample
generalization, E5 downstream use, E6 initialization ablation, E7 scalability.
