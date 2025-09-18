#!/usr/bin/env python3
"""Dual encoder visualization utilities.

Produces per-language visuals under assets/analysis/<lang>/dual_vis:
- synonym and antonym TSNE/UMAP scatterplots
- cosine similarity heatmaps for syn and ant spaces (sample subset)
- decision-boundary visualization using concatenated embeddings and a linear classifier
- clustering quality metrics (silhouette, davies-bouldin) per space

This is intentionally conservative (subsamples large embeddings) to avoid OOM.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import argparse
import logging
try:
    from umap import UMAP
except Exception:
    UMAP = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dual_vis')

PROJECT_ROOT = Path('.')
ASSETS = PROJECT_ROOT / 'assets' / 'analysis'
TRAINED_DUAL = PROJECT_ROOT / 'models' / 'trained' / 'dual_encoder' / 'analysis'


def _load_embeddings(lang: str):
    emb_root = TRAINED_DUAL / 'embeddings'
    syn_f = emb_root / f"{lang}_dual_syn_embeddings.npy"
    ant_f = emb_root / f"{lang}_dual_ant_embeddings.npy"
    labels_f = emb_root / f"{lang}_labels.npy"
    if not syn_f.exists() and not ant_f.exists():
        return None
    syn = np.load(syn_f) if syn_f.exists() else None
    ant = np.load(ant_f) if ant_f.exists() else None
    labs = np.load(labels_f) if labels_f.exists() else None
    return syn, ant, labs


def _subsample(X, cap=1000, seed=42):
    if X is None:
        return None
    n = X.shape[0]
    if n <= cap:
        return X, np.arange(n)
    idx = np.random.RandomState(seed).choice(n, cap, replace=False)
    return X[idx], idx


def plot_space_scatter(X, labs, title, outp, use_umap=True):
    try:
        pca = PCA(n_components=min(50, X.shape[1]))
        Xp = pca.fit_transform(X)
    except Exception:
        Xp = X
    try:
        ts = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        Xt = ts.fit_transform(Xp)
        fig, ax = plt.subplots(figsize=(6,5))
        if labs is not None:
            sns.scatterplot(x=Xt[:,0], y=Xt[:,1], hue=labs, palette='tab10', s=10, ax=ax, legend=False)
        else:
            ax.scatter(Xt[:,0], Xt[:,1], s=10, c='C2')
        ax.set_title(title + ' (t-SNE)')
        fig.savefig(outp.with_suffix('.tsne.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"t-SNE failed for {title}: {e}")

    if use_umap and UMAP is not None:
        try:
            um = UMAP(n_components=2, random_state=42)
            Xu = um.fit_transform(Xp)
            fig, ax = plt.subplots(figsize=(6,5))
            if labs is not None:
                sns.scatterplot(x=Xu[:,0], y=Xu[:,1], hue=labs, palette='tab10', s=10, ax=ax, legend=False)
            else:
                ax.scatter(Xu[:,0], Xu[:,1], s=10, c='C2')
            ax.set_title(title + ' (UMAP)')
            fig.savefig(outp.with_suffix('.umap.png'), dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"UMAP failed for {title}: {e}")


def plot_similarity_heatmap(X, outp, cap=200):
    # compute cosine similarities on a small subset and plot heatmap
    if X is None:
        return
    Xs, idx = _subsample(X, cap=cap)
    try:
        sim = cosine_similarity(Xs)
        # clip for better visualization
        sim = np.clip(sim, -1, 1)
        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(sim, vmin=-1, vmax=1, cmap='vlag', ax=ax)
        ax.set_title('Cosine similarity (subset)')
        fig.savefig(outp, dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to compute similarity heatmap: {e}")


def decision_boundary_visualization(X_syn, X_ant, labs, outp):
    """Train a simple classifier on combined syn+ant features and visualize decision boundary in 2D PCA space."""
    if X_syn is None or X_ant is None:
        return
    # concatenate per-sample syn and ant embeddings along features
    try:
        n = min(X_syn.shape[0], X_ant.shape[0])
        Xc = np.hstack([X_syn[:n], X_ant[:n]])
        if labs is not None:
            y = labs[:n]
        else:
            # fallback to kmeans labels for supervision
            y = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(Xc)

        # standardize
        sc = StandardScaler()
        Xs = sc.fit_transform(Xc)

        # reduce to 2D via PCA for plotting decision boundary
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(Xs)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X2, y)

        # plot scatter and decision boundary
        x_min, x_max = X2[:,0].min() - .5, X2[:,0].max() + .5
        y_min, y_max = X2[:,1].min() - .5, X2[:,1].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(6,5))
        ax.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
        sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=y, palette='tab10', s=12, ax=ax, legend=False)
        ax.set_title('Decision boundary in PCA space (concatenated syn+ant)')
        fig.savefig(outp, dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed decision boundary visualization: {e}")


def clustering_quality(X, label, outp_prefix):
    res = {}
    if X is None:
        return res
    try:
        # KMeans cluster to 2 clusters (visualization) and compute silhouette/davies
        k = 2
        labs = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        if X.shape[0] > k:
            sil = silhouette_score(X, labs)
            db = davies_bouldin_score(X, labs)
        else:
            sil = float('nan')
            db = float('nan')
        res['silhouette'] = float(sil)
        res['davies_bouldin'] = float(db)
        # also save the cluster scatter
        try:
            plot_space_scatter(X, labs, outp_prefix.name + ' clustering', outp_prefix, use_umap=True)
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Failed clustering quality: {e}")
    return res


def analyze_language(lang: str, out_root: Path):
    syn, ant, labs = _load_embeddings(lang)
    out_dir = out_root / lang / 'dual_vis'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {'language': lang}

    # Synonym space scatter & similarity
    if syn is not None:
        Xs, idxs = _subsample(syn, cap=2000)
        plot_space_scatter(Xs, labs[idxs] if labs is not None and len(labs) >= Xs.shape[0] else None,
                           f"{lang} - Dual SYN", out_dir / f"{lang}_dual_syn")
        plot_similarity_heatmap(Xs, out_dir / f"{lang}_dual_syn_similarity.png", cap=200)
        q_syn = clustering_quality(Xs, 'syn', out_dir / f"{lang}_dual_syn_cluster")
        summary.update({f"syn_{k}": v for k, v in q_syn.items()})

    # Antonym space scatter & similarity
    if ant is not None:
        Xa, idxs_a = _subsample(ant, cap=2000)
        plot_space_scatter(Xa, labs[idxs_a] if labs is not None and len(labs) >= Xa.shape[0] else None,
                           f"{lang} - Dual ANT", out_dir / f"{lang}_dual_ant")
        plot_similarity_heatmap(Xa, out_dir / f"{lang}_dual_ant_similarity.png", cap=200)
        q_ant = clustering_quality(Xa, 'ant', out_dir / f"{lang}_dual_ant_cluster")
        summary.update({f"ant_{k}": v for k, v in q_ant.items()})

    # Decision boundary on concatenated spaces
    if syn is not None and ant is not None:
        # use smaller subset to keep visualization fast
        s_cap = min(2000, syn.shape[0], ant.shape[0])
        Xs_full = syn[:s_cap]
        Xa_full = ant[:s_cap]
        decision_boundary_visualization(Xs_full, Xa_full, labs[:s_cap] if labs is not None else None,
                                        out_dir / f"{lang}_decision_boundary.png")

    # save summary CSV
    df = pd.DataFrame([summary])
    df.to_csv(out_dir / f"{lang}_dual_vis_summary.csv", index=False)
    logger.info(f"Wrote dual visualizations for {lang} to {out_dir}")


def main(languages=None):
    languages = languages or sorted([p.name for p in (PROJECT_ROOT / 'assets' / 'analysis').iterdir() if p.is_dir()])
    out_root = PROJECT_ROOT / 'assets' / 'analysis'
    for lang in languages:
        try:
            analyze_language(lang, out_root)
        except Exception as e:
            logger.warning(f"Failed to analyze {lang}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', nargs='+', help='Languages to analyze')
    args = parser.parse_args()
    langs = args.languages if args.languages else None
    main(langs)
