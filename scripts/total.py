#!/usr/bin/env python3
"""Aggregate analysis outputs and trained-model analysis into final tables and plots.

Scans:
 - assets/analysis/<lang>/ for baseline reports and predictions
 - models/trained/bert/analysis and models/trained/dual_encoder/analysis for per-language trained analysis (tables, embeddings, plots)

Produces:
 - assets/analysis/final_summary.csv (absolute paths inside)
 - assets/analysis/final_summary.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import json
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
try:
    from umap import UMAP
except Exception:
    UMAP = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('total')

# Hard-coded absolute roots (explicitly set to user's runtime paths)
PROJECT_ROOT = Path('/home/scratch/samyak/temp/multilingual_antonym_detection')
ASSETS_ANALYSIS = Path('/home/scratch/samyak/temp/multilingual_antonym_detection/assets/analysis')
TRAINED_BERT_ANALYSIS = Path('/home/scratch/samyak/temp/multilingual_antonym_detection/models/trained/bert/analysis')
TRAINED_DUAL_ANALYSIS = Path('/home/scratch/samyak/temp/multilingual_antonym_detection/models/trained/dual_encoder/analysis')
OUT_CSV = ASSETS_ANALYSIS / 'final_summary.csv'
OUT_PNG = ASSETS_ANALYSIS / 'final_summary.png'

# Default ordered languages for grids (8 languages)
DEFAULT_LANGS = ['english', 'dutch', 'french', 'italian', 'russian', 'spanish', 'portuguese', 'german']

# Log paths for diagnostics
logger.info(f'Using PROJECT_ROOT={PROJECT_ROOT}')
logger.info(f'ASSETS_ANALYSIS={ASSETS_ANALYSIS}')
logger.info(f'TRAINED_BERT_ANALYSIS={TRAINED_BERT_ANALYSIS}')
logger.info(f'TRAINED_DUAL_ANALYSIS={TRAINED_DUAL_ANALYSIS}')


def _pseudo_labels_for_visualization(X):
    """Create 2-cluster pseudo-labels (KMeans on a small PCA projection) for visualization.

    Returns None on failure.
    """
    try:
        try:
            p_small = PCA(n_components=min(10, X.shape[1]))
            Xs = p_small.fit_transform(X)
        except Exception:
            Xs = X
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labs = km.fit_predict(Xs)
        return labs
    except Exception:
        return None


def read_metric_from_trained(lang_dir: Path, prefix: str):
    """Try to read an accuracy or metric file under the trained analysis dir for a language.
    Common files: <lang>_confusion_matrix.txt, tables/history.csv, plots/ ...
    Returns a dict of metrics found.
    """
    res = {}
    # look for tables/history*.csv
    try:
        tables_dir = lang_dir / 'tables'
        if tables_dir.exists():
            for f in tables_dir.glob('*history*.csv'):
                try:
                    df = pd.read_csv(f)
                    # find accuracy-like columns
                    for col in ['test_acc', 'val_acc', 'accuracy', 'test_accuracy']:
                        if col in df.columns:
                            res['final_acc'] = float(df[col].dropna().iloc[-1])
                            break
                except Exception:
                    continue
    except Exception:
        pass

    # also try top-level confusion matrix file
    cm_file = lang_dir.parent / f"{lang_dir.name}_confusion_matrix.txt"
    if cm_file.exists():
        # can't parse easily; mark existence
        res['has_confusion_matrix'] = True
    return res


def aggregate(languages=None, write_csv: bool = True):
    rows = []
    languages = languages or sorted([p.name for p in ASSETS_ANALYSIS.iterdir() if p.is_dir()])
    for lang in languages:
        a_dir = ASSETS_ANALYSIS / lang
        row = {
            'language': lang,
            'analysis_dir': str(a_dir.resolve()),
            'bert_pred_csv': None,
            'dual_pred_csv': None,
            'baseline_report': None,
            'trained_bert_metrics': None,
            'trained_dual_metrics': None,
        }
        # assets/analysis files
        bert_pred = a_dir / f"{lang}_bert_predictions.csv"
        dual_pred = a_dir / f"{lang}_dual_predictions.csv"
        baseline_report = a_dir / f"{lang}_baseline_probe_report.txt"
        if bert_pred.exists():
            row['bert_pred_csv'] = str(bert_pred.resolve())
            try:
                df = pd.read_csv(bert_pred)
                if 'label' in df.columns and 'pred' in df.columns:
                    # compute macro F1 as the canonical metric
                    try:
                        row['bert_f1'] = float(f1_score(df['label'], df['pred'], average='macro'))
                    except Exception:
                        # fallback to accuracy if F1 can't be computed
                        row['bert_f1'] = float((df['label'] == df['pred']).mean())
            except Exception:
                pass
        if dual_pred.exists():
            row['dual_pred_csv'] = str(dual_pred.resolve())
            try:
                df2 = pd.read_csv(dual_pred)
                if 'label' in df2.columns and 'pred' in df2.columns:
                    try:
                        row['dual_f1'] = float(f1_score(df2['label'], df2['pred'], average='macro'))
                    except Exception:
                        row['dual_f1'] = float((df2['label'] == df2['pred']).mean())
            except Exception:
                pass
        if baseline_report.exists():
            row['baseline_report'] = str(baseline_report.resolve())
            # try to read F1 or accuracy from file
            try:
                txt = baseline_report.read_text(encoding='utf-8')
                for line in txt.splitlines():
                    if 'F1:' in line:
                        try:
                            row['baseline_f1'] = float(line.split('F1:')[-1].strip())
                        except Exception:
                            pass
                    if 'Accuracy:' in line and 'baseline_f1' not in row:
                        try:
                            row['baseline_f1'] = float(line.split('Accuracy:')[-1].strip())
                        except Exception:
                            pass
            except Exception:
                pass

        # trained model analysis
        bert_trained_dir = TRAINED_BERT_ANALYSIS / lang
        dual_trained_dir = TRAINED_DUAL_ANALYSIS / lang
        if bert_trained_dir.exists():
            row['trained_bert_metrics'] = read_metric_from_trained(bert_trained_dir, 'bert')
            # also check for prediction CSVs under trained analysis
            try:
                predf = bert_trained_dir / f"{lang}_bert_predictions.csv"
                if predf.exists():
                    row['bert_pred_csv'] = str(predf.resolve())
                    try:
                        dfp = pd.read_csv(predf)
                        if 'label' in dfp.columns and 'pred' in dfp.columns:
                            try:
                                row['bert_f1'] = float(f1_score(dfp['label'], dfp['pred'], average='macro'))
                            except Exception:
                                row['bert_f1'] = float((dfp['label'] == dfp['pred']).mean())
                    except Exception:
                        logger.warning(f'Could not read bert predictions CSV for {lang} at {predf}')
            except Exception:
                pass
        if dual_trained_dir.exists():
            row['trained_dual_metrics'] = read_metric_from_trained(dual_trained_dir, 'dual')
            try:
                predf2 = dual_trained_dir / f"{lang}_dual_predictions.csv"
                if predf2.exists():
                    row['dual_pred_csv'] = str(predf2.resolve())
                    try:
                        dfp2 = pd.read_csv(predf2)
                        if 'label' in dfp2.columns and 'pred' in dfp2.columns:
                            try:
                                row['dual_f1'] = float(f1_score(dfp2['label'], dfp2['pred'], average='macro'))
                            except Exception:
                                row['dual_f1'] = float((dfp2['label'] == dfp2['pred']).mean())
                    except Exception:
                        logger.warning(f'Could not read dual predictions CSV for {lang} at {predf2}')
            except Exception:
                pass

        rows.append(row)

    df = pd.DataFrame(rows)
    if write_csv:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_CSV, index=False)
        logger.info(f"Wrote aggregate CSV to {OUT_CSV}")

    # Create a simple bar chart comparing bert_acc, dual_acc, baseline_acc if present
    try:
        plt.figure(figsize=(10,6))
        idx = np.arange(len(df))
        width = 0.25
        # prefer F1 if present, fallback to accuracy-like columns
        bert_vals = df.get('bert_f1', df.get('bert_acc', pd.Series([np.nan]*len(df)))).fillna(0)
        dual_vals = df.get('dual_f1', df.get('dual_acc', pd.Series([np.nan]*len(df)))).fillna(0)
        base_vals = df.get('baseline_f1', df.get('baseline_acc', pd.Series([np.nan]*len(df)))).fillna(0)
        plt.bar(idx - width, bert_vals, width, label='BERT (F1)')
        plt.bar(idx, dual_vals, width, label='Dual (F1)')
        plt.bar(idx + width, base_vals, width, label='Baseline (F1)')
        plt.xticks(idx, df['language'], rotation=45)
        plt.ylabel('Macro-F1')
        plt.title('Model comparison across languages (macro-F1 preferred)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_PNG)
        logger.info(f"Wrote aggregate plot to {OUT_PNG}")
    except Exception as e:
        logger.warning(f"Failed to create comparison plot: {e}")

    return df


def make_comparison_plot(df: pd.DataFrame, out_png: Path):
    """Create and save the comparison bar chart (Macro-F1 preferred)."""
    try:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10,6))
        idx = np.arange(len(df))
        width = 0.25
        bert_vals = df.get('bert_f1', df.get('bert_acc', pd.Series([np.nan]*len(df)))).fillna(0)
        dual_vals = df.get('dual_f1', df.get('dual_acc', pd.Series([np.nan]*len(df)))).fillna(0)
        base_vals = df.get('baseline_f1', df.get('baseline_acc', pd.Series([np.nan]*len(df)))).fillna(0)
        plt.bar(idx - width, bert_vals, width, label='BERT (F1)')
        plt.bar(idx, dual_vals, width, label='Dual (F1)')
        plt.bar(idx + width, base_vals, width, label='Baseline (F1)')
        plt.xticks(idx, df['language'], rotation=45)
        plt.ylabel('Macro-F1')
        plt.title('Model comparison across languages (macro-F1 preferred)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png)
        logger.info(f"Wrote aggregate plot to {out_png}")
    except Exception as e:
        logger.warning(f"Failed to create comparison plot: {e}")


def _collect_embeddings(root: Path, pattern: str):
    """Collect embeddings matching pattern under root/embeddings.
    Returns list of (lang, embeddings, labels)"""
    out = []
    emb_root = root / 'embeddings'
    if not emb_root.exists():
        return out
    for f in sorted(emb_root.glob(pattern)):
        name = f.name
        # infer language from filename: e.g., german_bert_cls_embeddings.npy -> german
        lang = name.split('_')[0]
        try:
            embs = np.load(f)
        except Exception:
            continue
        # try to find labels file
        # possible labels names: {lang}_bert_labels.npy or {lang}_labels.npy
        labels = None
        for lab_pattern in [f"{lang}_bert_labels.npy", f"{lang}_labels.npy", f"{lang}_bert_labels.npy"]:
            labf = emb_root / lab_pattern
            if labf.exists():
                try:
                    labels = np.load(labf)
                except Exception:
                    labels = None
                break
        out.append((lang, embs, labels))
    return out


def plot_combined_embeddings(project_root: Path, out_dir: Path, max_points=5000, per_lang_cap=1000):
    """Create combined t-SNE and UMAP plots for BERT CLS and Dual syn/ant embeddings.
    Writes PNGs into out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bert_root = project_root / 'models' / 'trained' / 'bert' / 'analysis'
    dual_root = project_root / 'models' / 'trained' / 'dual_encoder' / 'analysis'

    # Collect bert cls embeddings
    bert_embs = _collect_embeddings(bert_root, '*_bert_cls_embeddings.npy')
    if bert_embs:
        # assemble arrays and labels
        X_parts = []
        y_lang = []
        y_label = []
        for lang, embs, labels in bert_embs:
            if embs is None:
                continue
            n = embs.shape[0]
            cap = min(per_lang_cap, n)
            if n > cap:
                idx = np.random.RandomState(42).choice(n, cap, replace=False)
                embs_s = embs[idx]
                labs_s = labels[idx] if labels is not None else np.full(len(idx), -1)
            else:
                embs_s = embs
                labs_s = labels if labels is not None else np.full(n, -1)
            X_parts.append(embs_s)
            y_lang += [lang] * embs_s.shape[0]
            y_label += [int(l) if (labs_s is not None and len(labs_s)>0) else -1 for l in labs_s]

        if X_parts:
            X = np.vstack(X_parts)
            # optional PCA prior to TSNE for speed
            try:
                pca = PCA(n_components=min(50, X.shape[1]))
                Xp = pca.fit_transform(X)
            except Exception:
                Xp = X

            # t-SNE
            try:
                ts = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
                Xt = ts.fit_transform(Xp)
                plt.figure(figsize=(10,8))
                sns.scatterplot(x=Xt[:,0], y=Xt[:,1], hue=y_lang, palette='tab10', s=6, legend='full')
                plt.title('t-SNE of BERT CLS embeddings (all languages)')
                plt.savefig(out_dir / 'combined_tsne_bert.png', dpi=150)
                plt.close()
            except Exception as e:
                logger.warning(f"t-SNE failed for BERT combined: {e}")

            # UMAP
            try:
                um = UMAP(n_components=2, random_state=42)
                Xu = um.fit_transform(Xp)
                plt.figure(figsize=(10,8))
                sns.scatterplot(x=Xu[:,0], y=Xu[:,1], hue=y_lang, palette='tab10', s=6, legend='full')
                plt.title('UMAP of BERT CLS embeddings (all languages)')
                plt.savefig(out_dir / 'combined_umap_bert.png', dpi=150)
                plt.close()
            except Exception as e:
                logger.warning(f"UMAP failed for BERT combined: {e}")

    # Collect dual encoder syn embeddings
    dual_syn = _collect_embeddings(dual_root, '*_dual_syn_embeddings.npy')
    dual_ant = _collect_embeddings(dual_root, '*_dual_ant_embeddings.npy')
    if dual_syn:
        X_parts = []
        y_lang = []
        for lang, embs, labels in dual_syn:
            if embs is None:
                continue
            n = embs.shape[0]
            cap = min(per_lang_cap, n)
            if n > cap:
                idx = np.random.RandomState(42).choice(n, cap, replace=False)
                embs_s = embs[idx]
            else:
                embs_s = embs
            X_parts.append(embs_s)
            y_lang += [lang] * embs_s.shape[0]
        if X_parts:
            X = np.vstack(X_parts)
            try:
                pca = PCA(n_components=min(50, X.shape[1]))
                Xp = pca.fit_transform(X)
            except Exception:
                Xp = X
            try:
                ts = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
                Xt = ts.fit_transform(Xp)
                plt.figure(figsize=(10,8))
                sns.scatterplot(x=Xt[:,0], y=Xt[:,1], hue=y_lang, palette='tab10', s=6, legend='full')
                plt.title('t-SNE of Dual-Encoder SYN embeddings (all languages)')
                plt.savefig(out_dir / 'combined_tsne_dual_syn.png', dpi=150)
                plt.close()
            except Exception as e:
                logger.warning(f"t-SNE failed for Dual syn combined: {e}")
            try:
                um = UMAP(n_components=2, random_state=42)
                Xu = um.fit_transform(Xp)
                plt.figure(figsize=(10,8))
                sns.scatterplot(x=Xu[:,0], y=Xu[:,1], hue=y_lang, palette='tab10', s=6, legend='full')
                plt.title('UMAP of Dual-Encoder SYN embeddings (all languages)')
                plt.savefig(out_dir / 'combined_umap_dual_syn.png', dpi=150)
                plt.close()
            except Exception as e:
                logger.warning(f"UMAP failed for Dual syn combined: {e}")

    # Optionally create a combined plot for dual ant embeddings as well
    if dual_ant:
        X_parts = []
        y_lang = []
        for lang, embs, labels in dual_ant:
            if embs is None:
                continue
            n = embs.shape[0]
            cap = min(per_lang_cap, n)
            if n > cap:
                idx = np.random.RandomState(42).choice(n, cap, replace=False)
                embs_s = embs[idx]
            else:
                embs_s = embs
            X_parts.append(embs_s)
            y_lang += [lang] * embs_s.shape[0]
        if X_parts:
            X = np.vstack(X_parts)
            try:
                pca = PCA(n_components=min(50, X.shape[1]))
                Xp = pca.fit_transform(X)
            except Exception:
                Xp = X
            try:
                ts = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
                Xt = ts.fit_transform(Xp)
                plt.figure(figsize=(10,8))
                sns.scatterplot(x=Xt[:,0], y=Xt[:,1], hue=y_lang, palette='tab10', s=6, legend='full')
                plt.title('t-SNE of Dual-Encoder ANT embeddings (all languages)')
                plt.savefig(out_dir / 'combined_tsne_dual_ant.png', dpi=150)
                plt.close()
            except Exception as e:
                logger.warning(f"t-SNE failed for Dual ant combined: {e}")
            try:
                um = UMAP(n_components=2, random_state=42)
                Xu = um.fit_transform(Xp)
                plt.figure(figsize=(10,8))
                sns.scatterplot(x=Xu[:,0], y=Xu[:,1], hue=y_lang, palette='tab10', s=6, legend='full')
                plt.title('UMAP of Dual-Encoder ANT embeddings (all languages)')
                plt.savefig(out_dir / 'combined_umap_dual_ant.png', dpi=150)
                plt.close()
            except Exception as e:
                logger.warning(f"UMAP failed for Dual ant combined: {e}")


def collect_per_language_plots(languages=None):
    """Search analysis folders for existing TSNE/UMAP plot PNGs and return a mapping.

    Returns dict: lang -> {plot_key: Path}
    plot_key examples: 'bert_tsne', 'bert_umap', 'dual_syn_tsne', 'dual_syn_umap', 'dual_ant_tsne', 'dual_ant_umap'
    """
    out = {}
    # candidate roots to search
    candidate_roots = [ASSETS_ANALYSIS, TRAINED_BERT_ANALYSIS, TRAINED_DUAL_ANALYSIS]
    # languages list
    langs = languages or sorted({p.name for p in ASSETS_ANALYSIS.iterdir() if p.is_dir()})
    for lang in langs:
        found = {}
        patterns = {
            'bert_tsne': ['*bert_tsne*.png', '*bert_cls_tsne*.png'],
            'bert_umap': ['*bert_umap*.png', '*bert_cls_umap*.png'],
            'dual_syn_tsne': ['*dual_syn_tsne*.png', '*dual_syn_tsne*.png', '*dual_syn_tsne.png'],
            'dual_syn_umap': ['*dual_syn_umap*.png'],
            'dual_ant_tsne': ['*dual_ant_tsne*.png'],
            'dual_ant_umap': ['*dual_ant_umap*.png']
        }
        for root in candidate_roots:
            # check lang-specific dirs and their 'plots' subdirs
            candidates = [root / lang, root / lang / 'plots', root / lang / 'analysis' / 'plots']
            for c in candidates:
                if not c.exists():
                    continue
                for key, globs in patterns.items():
                    if key in found:
                        continue
                    for g in globs:
                        for f in c.glob(g):
                            found[key] = f
                            break
                        if key in found:
                            break
        # If some plots missing, try to generate them from embeddings into assets/analysis/<lang>/plots
        missing_keys = [k for k in ['bert_tsne','bert_umap','dual_syn_tsne','dual_syn_umap','dual_ant_tsne','dual_ant_umap'] if k not in found]
        if missing_keys:
            gen_dir = ASSETS_ANALYSIS / lang / 'plots'
            gen = _generate_plots_from_embeddings(lang, gen_dir)
            for k, v in gen.items():
                if k not in found:
                    found[k] = v
        out[lang] = found
    return out


def _generate_plots_from_embeddings(lang: str, out_plots_dir: Path, per_lang_cap: int = 1000):
    """Try to generate TSNE/UMAP plots from saved .npy embeddings for a language.

    Saves plots under out_plots_dir (creates dir). Returns dict of generated plot paths.
    """
    generated = {}
    out_plots_dir.mkdir(parents=True, exist_ok=True)
    bert_emb_root = PROJECT_ROOT / 'models' / 'trained' / 'bert' / 'analysis' / 'embeddings'
    dual_emb_root = PROJECT_ROOT / 'models' / 'trained' / 'dual_encoder' / 'analysis' / 'embeddings'

    # Helper to create KMeans pseudo-labels when labels missing
    def _pseudo_labels_for_visualization(X):
        try:
            # small PCA to reduce noise
            try:
                p_small = PCA(n_components=min(10, X.shape[1]))
                Xs = p_small.fit_transform(X)
            except Exception:
                Xs = X
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            labs = km.fit_predict(Xs)
            return labs
        except Exception:
            return None

    # BERT CLS embeddings
    bert_file = bert_emb_root / f"{lang}_bert_cls_embeddings.npy"
    bert_labels = bert_emb_root / f"{lang}_bert_labels.npy"
    if bert_file.exists():
        try:
            embs = np.load(bert_file)
            labels = np.load(bert_labels) if bert_labels.exists() else None
            n = embs.shape[0]
            cap = min(per_lang_cap, n)
            idx = np.random.RandomState(42).choice(n, cap, replace=False) if n > cap else np.arange(n)
            X = embs[idx]
            # only use labels if the labels array length matches embeddings length
            if labels is not None and len(labels) == n:
                y = labels[idx]
            else:
                y = None

            # PCA preproj
            try:
                pca = PCA(n_components=min(50, X.shape[1]))
                Xp = pca.fit_transform(X)
            except Exception:
                Xp = X

            # t-SNE
            try:
                ts = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
                Xt = ts.fit_transform(Xp)
                fig, ax = plt.subplots(figsize=(6,5))
                if y is not None:
                    sns.scatterplot(x=Xt[:,0], y=Xt[:,1], hue=y, palette='coolwarm', s=10, ax=ax, legend=False)
                else:
                    ax.scatter(Xt[:,0], Xt[:,1], s=10, c='C0')
                ax.set_title(f"{lang} - BERT CLS t-SNE")
                p = out_plots_dir / f"{lang}_bert_tsne.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                generated['bert_tsne'] = p
            except Exception as e:
                logger.warning(f"Failed to compute per-language t-SNE for {lang}: {e}")

            # UMAP
            if UMAP is not None:
                try:
                    um = UMAP(n_components=2, random_state=42)
                    Xu = um.fit_transform(Xp)
                    fig, ax = plt.subplots(figsize=(6,5))
                    if y is not None:
                        sns.scatterplot(x=Xu[:,0], y=Xu[:,1], hue=y, palette='coolwarm', s=10, ax=ax, legend=False)
                    else:
                        ax.scatter(Xu[:,0], Xu[:,1], s=10, c='C0')
                    ax.set_title(f"{lang} - BERT CLS UMAP")
                    p = out_plots_dir / f"{lang}_bert_umap.png"
                    fig.savefig(p, dpi=150)
                    plt.close(fig)
                    generated['bert_umap'] = p
                except Exception as e:
                    logger.warning(f"Failed to compute per-language UMAP for {lang}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load BERT embeddings for {lang}: {e}")

    # Dual syn/ant embeddings
    dual_syn_file = dual_emb_root / f"{lang}_dual_syn_embeddings.npy"
    dual_ant_file = dual_emb_root / f"{lang}_dual_ant_embeddings.npy"
    dual_labels = dual_emb_root / f"{lang}_labels.npy"

    if dual_syn_file.exists():
        try:
            embs = np.load(dual_syn_file)
            n = embs.shape[0]
            cap = min(per_lang_cap, n)
            idx = np.random.RandomState(42).choice(n, cap, replace=False) if n > cap else np.arange(n)
            X = embs[idx]
            # load labels safely only if lengths match
            if dual_labels.exists():
                labs = np.load(dual_labels)
                if len(labs) == n:
                    y = labs[idx]
                else:
                    y = None
            else:
                y = None

            try:
                pca = PCA(n_components=min(50, X.shape[1]))
                Xp = pca.fit_transform(X)
            except Exception:
                Xp = X

            try:
                ts = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
                Xt = ts.fit_transform(Xp)
                fig, ax = plt.subplots(figsize=(6,5))
                # If labels missing, create pseudo-labels via KMeans for visualization
                if y is None:
                    y = _pseudo_labels_for_visualization(X)
                if y is not None:
                    sns.scatterplot(x=Xt[:,0], y=Xt[:,1], hue=y, palette='tab10', s=10, ax=ax, legend=False)
                else:
                    ax.scatter(Xt[:,0], Xt[:,1], s=10, c='C1')
                ax.set_title(f"{lang} - Dual Syn t-SNE")
                p = out_plots_dir / f"{lang}_dual_syn_tsne.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                generated['dual_syn_tsne'] = p
            except Exception as e:
                logger.warning(f"Failed to compute dual syn t-SNE for {lang}: {e}")

            if UMAP is not None:
                try:
                    um = UMAP(n_components=2, random_state=42)
                    Xu = um.fit_transform(Xp)
                    fig, ax = plt.subplots(figsize=(6,5))
                    if y is None:
                        y = _pseudo_labels_for_visualization(X)
                    if y is not None:
                        sns.scatterplot(x=Xu[:,0], y=Xu[:,1], hue=y, palette='tab10', s=10, ax=ax, legend=False)
                    else:
                        ax.scatter(Xu[:,0], Xu[:,1], s=10, c='C1')
                    ax.set_title(f"{lang} - Dual Syn UMAP")
                    p = out_plots_dir / f"{lang}_dual_syn_umap.png"
                    fig.savefig(p, dpi=150)
                    plt.close(fig)
                    generated['dual_syn_umap'] = p
                except Exception as e:
                    logger.warning(f"Failed to compute dual syn UMAP for {lang}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load dual syn embeddings for {lang}: {e}")

    if dual_ant_file.exists():
        try:
            embs = np.load(dual_ant_file)
            n = embs.shape[0]
            cap = min(per_lang_cap, n)
            idx = np.random.RandomState(42).choice(n, cap, replace=False) if n > cap else np.arange(n)
            X = embs[idx]
            if dual_labels.exists():
                labs = np.load(dual_labels)
                if len(labs) == n:
                    y = labs[idx]
                else:
                    y = None
            else:
                y = None

            try:
                pca = PCA(n_components=min(50, X.shape[1]))
                Xp = pca.fit_transform(X)
            except Exception:
                Xp = X

            try:
                ts = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
                Xt = ts.fit_transform(Xp)
                fig, ax = plt.subplots(figsize=(6,5))
                if y is None:
                    y = _pseudo_labels_for_visualization(X)
                if y is not None:
                    sns.scatterplot(x=Xt[:,0], y=Xt[:,1], hue=y, palette='tab10', s=10, ax=ax, legend=False)
                else:
                    ax.scatter(Xt[:,0], Xt[:,1], s=10, c='C1')
                ax.set_title(f"{lang} - Dual Ant t-SNE")
                p = out_plots_dir / f"{lang}_dual_ant_tsne.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                generated['dual_ant_tsne'] = p
            except Exception as e:
                logger.warning(f"Failed to compute dual ant t-SNE for {lang}: {e}")

            if UMAP is not None:
                try:
                    um = UMAP(n_components=2, random_state=42)
                    Xu = um.fit_transform(Xp)
                    fig, ax = plt.subplots(figsize=(6,5))
                    if y is None:
                        y = _pseudo_labels_for_visualization(X)
                    if y is not None:
                        sns.scatterplot(x=Xu[:,0], y=Xu[:,1], hue=y, palette='tab10', s=10, ax=ax, legend=False)
                    else:
                        ax.scatter(Xu[:,0], Xu[:,1], s=10, c='C1')
                    ax.set_title(f"{lang} - Dual Ant UMAP")
                    p = out_plots_dir / f"{lang}_dual_ant_umap.png"
                    fig.savefig(p, dpi=150)
                    plt.close(fig)
                    generated['dual_ant_umap'] = p
                except Exception as e:
                    logger.warning(f"Failed to compute dual ant UMAP for {lang}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load dual ant embeddings for {lang}: {e}")

    return generated


def plot_individual_and_grid(plot_map: dict, out_dir: Path):
    """Given per-language plot paths, save individual copies and a grid image comparing languages.

    plot_map: dict(lang -> dict(plot_key -> Path))
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.image as mpimg

    # Determine columns (plot types) order
    plot_keys = ['bert_tsne', 'bert_umap', 'dual_syn_tsne', 'dual_syn_umap', 'dual_ant_tsne', 'dual_ant_umap']

    langs = list(plot_map.keys())
    n_rows = len(langs)
    n_cols = len(plot_keys)

    # Save individual images (copy) and prepare grid
    for lang, pm in plot_map.items():
        lang_dir = out_dir / 'individual' / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        for key in plot_keys:
            p = pm.get(key)
            if p and p.exists():
                # copy to individual folder
                dst = lang_dir / f"{lang}_{key}.png"
                try:
                    img = mpimg.imread(p)
                    plt.imsave(dst, img)
                except Exception:
                    try:
                        # fallback to file copy
                        from shutil import copyfile
                        copyfile(str(p), str(dst))
                    except Exception:
                        pass

    # Create grid figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    # Ensure axes is a 2D array for consistent indexing
    axes = np.atleast_2d(axes)
    for i, lang in enumerate(langs):
        pm = plot_map[lang]
        for j, key in enumerate(plot_keys):
            ax = axes[i, j]
            p = pm.get(key)
            ax.axis('off')
            if p and p.exists():
                try:
                    img = mpimg.imread(p)
                    # Force the image to fill the subplot area to avoid tiny thumbnails
                    ax.imshow(img, aspect='auto', extent=[0, 1, 0, 1])
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_title(f"{lang} - {key}", fontsize=8)
                    ax.axis('off')
                except Exception:
                    ax.text(0.5, 0.5, 'failed to load', ha='center')
            else:
                ax.text(0.5, 0.5, 'missing', ha='center')

    plt.tight_layout()
    grid_path = out_dir / 'per_language_grid.png'
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved per-language grid of TSNE/UMAP plots to {grid_path}")


def plot_selected_grids(plot_map: dict, out_dir: Path, languages: list):
    """Create four 3x3 grids (TSNE bert, TSNE dual, UMAP bert, UMAP dual).

    languages: list of 8 languages in desired order. The grid is 3x3: the first 8 cells
    are the languages in order row-major, and the 9th cell is an index listing the languages.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.image as mpimg

    # Define mapping from grid name to plot_key in plot_map
    grids = {
        'tsne_bert': 'bert_tsne',
        'tsne_dual': 'dual_syn_tsne',
        'umap_bert': 'bert_umap',
        'umap_dual': 'dual_syn_umap'
    }

    # enforce exactly 8 languages
    langs = list(languages)[:8]
    while len(langs) < 8:
        langs.append('missing')

    for grid_name, key in grids.items():
        # Try to plot directly from embeddings for consistent axes and scaling
        coords = {}
        has_any = False
        # for bert grids we expect BERT CLS embeddings; for dual we use dual_syn embeddings
        for lang in langs:
            coords[lang] = None
            if 'bert' in grid_name:
                embf = PROJECT_ROOT / 'models' / 'trained' / 'bert' / 'analysis' / 'embeddings' / f"{lang}_bert_cls_embeddings.npy"
                labf = PROJECT_ROOT / 'models' / 'trained' / 'bert' / 'analysis' / 'embeddings' / f"{lang}_bert_labels.npy"
            else:
                embf = PROJECT_ROOT / 'models' / 'trained' / 'dual_encoder' / 'analysis' / 'embeddings' / f"{lang}_dual_syn_embeddings.npy"
                labf = PROJECT_ROOT / 'models' / 'trained' / 'dual_encoder' / 'analysis' / 'embeddings' / f"{lang}_labels.npy"

            if embf.exists():
                try:
                    embs = np.load(embf)
                    n = embs.shape[0]
                    cap = min(1000, n)
                    idx = np.random.RandomState(42).choice(n, cap, replace=False) if n > cap else np.arange(n)
                    X = embs[idx]
                    labs = None
                    if labf.exists():
                        try:
                            lab_all = np.load(labf)
                            if lab_all.shape[0] >= n:
                                labs = lab_all[idx]
                        except Exception:
                            labs = None

                    # PCA then TSNE/UMAP as requested
                    try:
                        pca = PCA(n_components=min(50, X.shape[1]))
                        Xp = pca.fit_transform(X)
                    except Exception:
                        Xp = X

                    if grid_name.startswith('tsne'):
                        try:
                            ts = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
                            X2 = ts.fit_transform(Xp)
                        except Exception as e:
                            logger.warning(f"t-SNE failed for {lang} in grid {grid_name}: {e}")
                            X2 = None
                    else:
                        if UMAP is None:
                            X2 = None
                        else:
                            try:
                                um = UMAP(n_components=2, random_state=42)
                                X2 = um.fit_transform(Xp)
                            except Exception as e:
                                logger.warning(f"UMAP failed for {lang} in grid {grid_name}: {e}")
                                X2 = None

                    if X2 is not None:
                        # if no labels are present for a dual grid, create pseudo-labels so the plot shows two colors
                        if labs is None and 'bert' not in grid_name:
                            try:
                                # labs correspond to X (pre-projection); compute on X
                                labs = _pseudo_labels_for_visualization(X)
                            except Exception:
                                labs = None
                        coords[lang] = (X2, labs)
                        has_any = True
                except Exception as e:
                    logger.warning(f"Failed to load embeddings for grid {grid_name} lang {lang}: {e}")

        # If we have embeddings for at least one language, compute global bounds and plot
        fig, axes = plt.subplots(3, 3, figsize=(9,9))
        axes = axes.reshape(3, 3)
        if has_any:
            # collect min/max
            xs = []
            ys = []
            for lang in langs:
                v = coords.get(lang)
                if v is None:
                    continue
                X2, _ = v
                xs.append(X2[:,0])
                ys.append(X2[:,1])
            if xs:
                allx = np.concatenate(xs)
                ally = np.concatenate(ys)
                xmin, xmax = np.percentile(allx, [1,99])
                ymin, ymax = np.percentile(ally, [1,99])
            else:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1

            for idx in range(9):
                r = idx // 3
                c = idx % 3
                ax = axes[r, c]
                ax.axis('off')
                if idx < 8:
                    lang = langs[idx]
                    v = coords.get(lang)
                    if v is None:
                        ax.text(0.5, 0.5, 'missing', ha='center')
                    else:
                        X2, labs = v
                        ax.scatter(X2[:,0], X2[:,1], c=(labs if labs is not None else 'C0'), s=8)
                        ax.set_xlim(xmin, xmax)
                        ax.set_ylim(ymin, ymax)
                        ax.set_title(lang, fontsize=9)
                        ax.axis('off')
                else:
                    txt = '\n'.join([f"{i+1}. {l}" for i, l in enumerate(langs)])
                    ax.text(0.02, 0.98, txt, va='top', ha='left', fontsize=10)

        else:
            # fallback: copy existing PNGs into grid cells
            for idx in range(9):
                r = idx // 3
                c = idx % 3
                ax = axes[r, c]
                ax.axis('off')
                if idx < 8:
                    lang = langs[idx]
                    pm = plot_map.get(lang, {})
                    p = pm.get(key)
                    if p and p.exists():
                        try:
                            img = mpimg.imread(p)
                            ax.imshow(img, aspect='auto', extent=[0,1,0,1])
                            ax.set_xlim(0,1)
                            ax.set_ylim(0,1)
                            ax.set_title(f"{lang}", fontsize=9)
                            ax.axis('off')
                        except Exception:
                            ax.text(0.5, 0.5, 'failed to load', ha='center')
                    else:
                        ax.text(0.5, 0.5, 'missing', ha='center')
                else:
                    txt = '\n'.join([f"{i+1}. {l}" for i, l in enumerate(langs)])
                    ax.text(0.02, 0.98, txt, va='top', ha='left', fontsize=10)

        plt.tight_layout()
        outp = out_dir / f'grid_{grid_name}.png'
        fig.savefig(outp, dpi=150)
        plt.close(fig)
        logger.info(f"Saved grid {grid_name} to {outp}")


def _build_focused_grids(plot_map: dict, out_dir: Path):
    """Build four focused grids: BERT t-SNE, Dual t-SNE, BERT UMAP, Dual UMAP.

    Each grid will have one column per language and will be saved under out_dir/grids.
    """
    grid_root = out_dir / 'grids'
    grid_root.mkdir(parents=True, exist_ok=True)
    import matplotlib.image as mpimg

    # Define mapping for the four grids
    grids = {
        'bert_tsne_grid': 'bert_tsne',
        'dual_tsne_grid': 'dual_syn_tsne',
        'bert_umap_grid': 'bert_umap',
        'dual_umap_grid': 'dual_syn_umap'
    }

    langs = list(plot_map.keys())
    for grid_name, key in grids.items():
        n_cols = len(langs)
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
        axes = np.atleast_1d(axes)
        for i, lang in enumerate(langs):
            ax = axes[i]
            p = plot_map[lang].get(key)
            ax.axis('off')
            if p and p.exists():
                try:
                    img = mpimg.imread(p)
                    ax.imshow(img, aspect='auto', extent=[0,1,0,1])
                except Exception:
                    ax.text(0.5, 0.5, 'failed', ha='center')
            else:
                ax.text(0.5, 0.5, 'missing', ha='center')
            ax.set_title(lang, fontsize=9)
        plt.tight_layout()
        outp = grid_root / f"{grid_name}.png"
        fig.savefig(outp, dpi=150)
        plt.close(fig)
        logger.info(f"Saved focused grid {outp}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', nargs='+', help='Optional list of languages to aggregate')
    parser.add_argument('--recompile', action='store_true', help='Recompute combined embeddings (t-SNE/UMAP) and recompile final plots')
    parser.add_argument('--grid', dest='grid', action='store_true', help='Build per-language grid of TSNE/UMAP plots')
    parser.add_argument('--no-grid', dest='grid', action='store_false', help='Do not build per-language grid of plots')
    parser.set_defaults(grid=True)
    args = parser.parse_args()

    # If recompile is requested, do the heavy embedding recompute but DO NOT overwrite final_summary.csv
    if args.recompile:
        out_dir = ASSETS_ANALYSIS / 'combined_embeddings'
        logger.info('Recomputing combined embeddings (t-SNE/UMAP) for all languages...')
        plot_combined_embeddings(PROJECT_ROOT, out_dir, max_points=5000, per_lang_cap=1000)

        # Load existing final_summary.csv if present to avoid overwriting it; otherwise aggregate in-memory without writing
        if OUT_CSV.exists():
            try:
                df = pd.read_csv(OUT_CSV)
                logger.info(f"Loaded existing final summary from {OUT_CSV} (no overwrite)")
            except Exception:
                df = aggregate(args.languages, write_csv=False)
        else:
            df = aggregate(args.languages, write_csv=False)

        # Recreate the comparison plot from the loaded/aggregated dataframe
        make_comparison_plot(df, OUT_PNG)
    else:
        # Normal behavior: aggregate and write final_summary.csv
        df = aggregate(args.languages, write_csv=True)

    print(df)

    # By default, build a per-language grid of available TSNE/UMAP plots
    if args.grid:
        out_dir = ASSETS_ANALYSIS / 'combined_embeddings'
        logger.info('Collecting per-language TSNE/UMAP plots and building grid...')
        # Use only the explicit language list to avoid noise from other folders
        languages_list = args.languages if args.languages else DEFAULT_LANGS
        # Ensure lower-case language names and preserve order
        languages_list = [l.lower() for l in languages_list]
        plot_map = collect_per_language_plots(languages_list)
        # Save individual copies and a full table-grid for the requested languages
        try:
            plot_individual_and_grid(plot_map, out_dir)
        except Exception as e:
            logger.warning(f'plot_individual_and_grid failed: {e}')

        # Create the four requested 3x3 grids (t-SNE bert, t-SNE dual, UMAP bert, UMAP dual)
        try:
            plot_selected_grids(plot_map, out_dir, languages_list)
        except Exception as e:
            logger.warning(f'plot_selected_grids failed: {e}')
