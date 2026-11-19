"""Interactive comparison of dimensionality-reduction methods (PCA, kPCA, Isomap, t-SNE, UMAP, autoencoder).

Run from the repo root:   streamlit run app.py
Reuses the project's own modules in src/ (data, methods, metrics, autoencoder); nothing there is modified.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))   # flat imports, as in tests/

from data import DATASETS, make_dataset          # noqa: E402
from methods import get_method                   # noqa: E402
import metrics as M                              # noqa: E402

try:
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))   # tiny MLPs: more threads only add overhead
    HAS_TORCH = True
except Exception:                                # pragma: no cover
    HAS_TORCH = False
try:
    import umap  # noqa: F401
    HAS_UMAP = True
except Exception:                                # pragma: no cover
    HAS_UMAP = False

# ----------------------------------------------------------------------------- look & feel
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
ACCENT = {"pca": "#64748b", "kpca": "#0d9488", "isomap": "#d97706",
          "tsne": "#7c3aed", "umap": "#db2777", "ae": "#2563eb"}
NAME = {"pca": "PCA", "kpca": "Kernel PCA", "isomap": "Isomap", "tsne": "t-SNE", "umap": "UMAP",
        "ae": "Autoencoder"}
CLASS_COLORS = ["#2563eb", "#d97706", "#0d9488", "#db2777", "#7c3aed", "#65a30d", "#dc2626", "#0891b2",
                "#a16207", "#475569"]

st.set_page_config(page_title="Dimensionality reduction", page_icon="◐", layout="wide")
st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1350px; }}
  h1 {{ font-weight: 650; letter-spacing: -0.02em; margin-bottom: 0; }}
  .sub {{ color: {MUTED}; margin: 0 0 1.2rem 0; font-size: 1.02rem; }}
  .card {{ border: 1px solid {GRID}; border-radius: 12px; padding: 0.7rem 1rem; background: #fff; }}
  .card .k {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; }}
  .card .v {{ font-size: 1.35rem; font-weight: 650; color: {INK}; }}
  .card .m {{ font-size: 0.85rem; }}
  [data-testid="stSidebar"] {{ background: #f8fafc; }}
  div.stButton > button[kind="primary"] {{ border-radius: 10px; font-weight: 600; padding: .55rem 1.6rem; }}
</style>""", unsafe_allow_html=True)


def style_fig(fig, height=360, title=None, color=INK):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=8, t=44 if title else 12, b=8),
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=12),
        title=dict(text=title, x=0.02, font=dict(size=15, color=color)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_data(name, n, D, noise, seed):
    """Returns X, color (for plotting), T (intrinsic coords or None), y (labels or None)."""
    if name == "digits (64-D)":
        from sklearn.datasets import load_digits
        d = load_digits()
        idx = np.random.default_rng(seed).permutation(len(d.data))[: min(n, len(d.data))]
        return d.data[idx].astype(float), d.target[idx], None, d.target[idx]
    ds = make_dataset(name, n=n, D=D, noise=noise, seed=seed)
    return ds.X, ds.color, getattr(ds, "T", None), getattr(ds, "y", None)


def median_sqdist(X, cap=1000):
    return float(np.median(pdist(X[:cap], "sqeuclidean")))


@st.cache_data(show_spinner=False)
def run_method(name, params_key, X, seed):
    params = dict(params_key)
    if "hidden" in params:
        params["hidden"] = tuple(params["hidden"])
    t0 = time.time()
    m = get_method(name, seed=seed, n_components=2, **params)
    Z = m.fit_transform(X)
    return Z, time.time() - t0, getattr(m, "history", None)


def evaluate(X, Z, T, y, k, rnx):
    tr, co = M.trust_continuity(X, Z, k=k)
    out = {"Trustworthiness": tr, "Continuity": co, "Distance corr.": M.dist_corr(X, Z)}
    if rnx:
        out["R_NX AUC"] = M.rnx_curve(X, Z)[2]
    if T is not None:
        out["Task R²"] = M.intrinsic_r2(Z, T)
    elif y is not None:
        out["kNN acc."] = M.knn_acc(Z, y)
    return out


def scatter(Z, color, discrete, accent, title):
    fig = go.Figure()
    if discrete:
        labels = np.asarray(color)
        for i, c in enumerate(sorted(set(labels.tolist()))):
            s = labels == c
            fig.add_scatter(x=Z[s, 0], y=Z[s, 1], mode="markers", name=str(c),
                            marker=dict(size=5, color=CLASS_COLORS[i % 10], opacity=0.85))
    else:
        fig.add_scatter(x=Z[:, 0], y=Z[:, 1], mode="markers", showlegend=False,
                        marker=dict(size=5, color=color, colorscale="Viridis", opacity=0.9))
    style_fig(fig, 340, title, accent)
    fig.update_layout(showlegend=discrete, legend=dict(orientation="h", y=-0.05, font=dict(size=10)))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return fig


# ----------------------------------------------------------------------------- sidebar
st.sidebar.markdown("### Data")
ds_name = st.sidebar.selectbox("Dataset", [*DATASETS, "digits (64-D)"])
n = st.sidebar.slider("Samples", 200, 3000, 1000, 100)
if ds_name == "digits (64-D)":
    D, noise = 64, 0.0
    st.sidebar.caption("sklearn digits (8×8 images), a small MNIST stand-in. Max 1797 samples.")
else:
    D = st.sidebar.slider("Ambient dimension", 3, 100, 50)
    noise = st.sidebar.slider("Noise", 0.0, 1.0, 0.1, 0.05)
seed = int(st.sidebar.number_input("Seed", 0, 10_000, 0))
X, color, T, y = load_data(ds_name, n, D, noise, seed)
n = len(X)
st.sidebar.markdown("### Evaluation")
k_nn = st.sidebar.slider("Neighbourhood size k", 5, 50, 10, help="Used by trustworthiness and continuity.")
do_rnx = st.sidebar.toggle("R_NX AUC", value=n <= 2000, help="O(n²); slow above ~2000 points.")

# ----------------------------------------------------------------------------- header + config
st.markdown("# Dimensionality reduction, side by side")
st.markdown(f'<p class="sub">{NAME["pca"]}, kernel PCA, Isomap, t-SNE, UMAP and autoencoders on '
            f'<b>{ds_name.replace("_", " ")}</b> · {n} points in {X.shape[1]}-D → 2-D</p>', unsafe_allow_html=True)

avail = [m for m in NAME if (m != "ae" or HAS_TORCH) and (m != "umap" or HAS_UMAP)]
if not HAS_TORCH:
    st.warning("PyTorch not found: autoencoder disabled.")
if not HAS_UMAP:
    st.warning("umap-learn not found: UMAP disabled.")
chosen = st.multiselect("Methods", avail, default=[m for m in ["pca", "kpca", "isomap", "tsne", "umap", "ae"]
                                                   if m in avail], format_func=NAME.get)

cfg = {}
with st.expander("Method settings", expanded=False):
    cols = st.columns(max(len(chosen), 1))
    for col, m in zip(cols, chosen):
        with col:
            st.markdown(f"**{NAME[m]}**")
            if m == "pca":
                cfg[m] = {}
                st.caption("No parameters.")
            elif m == "kpca":
                kern = st.selectbox("Kernel", ["rbf", "poly", "cosine", "sigmoid"], key="k_kern")
                p = {"kernel": kern}
                if kern == "rbf":
                    s = st.slider("γ scale (log₁₀)", -2.0, 2.0, 0.0, 0.25, key="k_g",
                                  help="γ = 10^s / (2 · median squared distance). Small → PCA; large → K ≈ I.")
                    p["gamma"] = float(10 ** s / (2 * median_sqdist(X)))
                cfg[m] = p
            elif m == "isomap":
                cfg[m] = {"n_neighbors": st.slider("Neighbours", 3, 60, 12, key="iso_k")}
            elif m == "tsne":
                cfg[m] = {"perplexity": st.slider("Perplexity", 5, 100, 30, key="ts_p")}
            elif m == "umap":
                cfg[m] = {"n_neighbors": st.slider("n_neighbors", 3, 100, 15, key="um_k"),
                          "min_dist": st.slider("min_dist", 0.0, 1.0, 0.1, 0.05, key="um_d")}
            elif m == "ae":
                linear = st.toggle("Linear autoencoder", value=False, key="ae_lin",
                                   help="No hidden layers or nonlinearity: spans the PCA subspace (Baldi–Hornik).")
                hid = () if linear else tuple(int(h) for h in st.text_input(
                    "Hidden layers", "128,64", key="ae_h", help="Comma separated, encoder side.").split(",")
                    if h.strip())
                p = {"hidden": hid, "epochs": st.slider("Max epochs", 20, 400, 100, 10, key="ae_e"),
                     "batch_size": st.select_slider("Batch size", [64, 128, 256, 512], 256, key="ae_b"),
                     "lr": st.select_slider("Learning rate", [3e-4, 1e-3, 3e-3, 1e-2], 3e-3, key="ae_lr"),
                     "patience": 12,
                     "noise_std": st.slider("Denoising noise σ", 0.0, 1.0, 0.0, 0.05, key="ae_n"),
                     "weight_decay": st.select_slider("Weight decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2], 0.0,
                                                      key="ae_w")}
                if not linear:
                    p["activation"] = st.selectbox("Activation", ["gelu", "relu", "tanh"], key="ae_a")
                cfg[m] = p

run = st.button("Run comparison", type="primary")
if run and chosen:
    res, prog = {}, st.progress(0.0)
    for i, m in enumerate(chosen):
        prog.progress(i / len(chosen), text=f"Fitting {NAME[m]}…")
        try:
            Z, dt, hist = run_method(m, tuple(sorted(cfg[m].items())), X, seed)
            ev = evaluate(X, Z, T, y, k_nn, do_rnx)
            ev["Time (s)"] = dt
            res[m] = (Z, ev, hist)
        except Exception as e:
            st.error(f"{NAME[m]} failed: {type(e).__name__}: {e}")
    prog.empty()
    st.session_state.update(res=res, color=color, discrete=(y is not None and T is None),
                            rnx=do_rnx, X=X, ae_linear=cfg.get("ae", {}).get("hidden") == ())

res = st.session_state.get("res")
if res and len(st.session_state["color"]) == n:
    discrete, Xr = st.session_state["discrete"], st.session_state["X"]
    labels = dict(NAME)
    if st.session_state.get("ae_linear"):
        labels["ae"] = "Linear autoencoder"

    # ---- embeddings
    for row in range(0, len(res), 3):
        batch = list(res.items())[row: row + 3]
        for col, (m, (Z, ev, _)) in zip(st.columns(3), batch):
            col.plotly_chart(scatter(Z, st.session_state["color"], discrete, ACCENT[m], labels[m]),
                             width="stretch", key=f"sc_{m}")

    # ---- best-per-metric cards
    table = pd.DataFrame({labels[m]: ev for m, (_, ev, _) in res.items()}).T
    quality = [c for c in table.columns if c != "Time (s)"]
    cards = st.columns(len(quality) + 1)
    inv = {v: k for k, v in labels.items()}
    for col, c in zip(cards, quality):
        best = table[c].idxmax()
        col.markdown(f'<div class="card"><div class="k">Best · {c}</div><div class="v">{table[c].max():.3f}</div>'
                     f'<div class="m" style="color:{ACCENT[inv[best]]}">● {best}</div></div>',
                     unsafe_allow_html=True)
    fastest = table["Time (s)"].idxmin()
    cards[-1].markdown(f'<div class="card"><div class="k">Fastest</div><div class="v">'
                       f'{table["Time (s)"].min():.2f}s</div><div class="m" style="color:'
                       f'{ACCENT[inv[fastest]]}">● {fastest}</div></div>', unsafe_allow_html=True)

    # ---- table
    st.markdown("&nbsp;")
    st.dataframe(table.style.format("{:.3f}").background_gradient(axis=0, cmap="Blues", subset=quality),
                 width="stretch")
    with st.expander("What the metrics mean"):
        st.markdown(
            "- **Trustworthiness**: points close in the embedding were close originally (no false neighbours).\n"
            "- **Continuity**: points close originally stay close in the embedding (no tearing).\n"
            "- **Distance corr.**: Spearman correlation of pairwise distances (global structure).\n"
            "- **R_NX AUC**: rank-based neighbourhood quality averaged over all scales (log-weighted).\n"
            "- **Task R² / kNN acc.**: how well the 2-D embedding predicts the intrinsic coordinates (manifold "
            "datasets) or class labels (labelled datasets).")

    # ---- R_NX + AE loss
    left, right = st.columns(2)
    if st.session_state["rnx"]:
        fig = go.Figure()
        for m in res:
            K, r, _ = M.rnx_curve(Xr, res[m][0])
            fig.add_scatter(x=K, y=r, name=labels[m], mode="lines", line=dict(color=ACCENT[m], width=2.5))
        style_fig(fig, 300, "R_NX(K): quality by neighbourhood size")
        fig.update_xaxes(type="log", title="K", gridcolor=GRID)
        fig.update_yaxes(title=None, gridcolor=GRID)
        fig.update_layout(legend=dict(orientation="h", y=-0.28))
        left.plotly_chart(fig, width="stretch", key="rnx")
    h = res.get("ae", (None, None, None))[2]
    if h and h["train"]:
        fig = go.Figure()
        fig.add_scatter(y=h["train"], name="train", line=dict(color=ACCENT["ae"], width=2.5))
        if h["val"]:
            fig.add_scatter(y=h["val"], name="validation", line=dict(color=MUTED, width=2, dash="dot"))
        style_fig(fig, 300, "Autoencoder loss (standardised MSE)")
        fig.update_yaxes(type="log", gridcolor=GRID)
        fig.update_xaxes(title="epoch", gridcolor=GRID)
        fig.update_layout(legend=dict(orientation="h", y=-0.28))
        right.plotly_chart(fig, width="stretch", key="ae_loss")
elif not run:
    st.info("Pick a dataset and methods, then press **Run comparison**.")