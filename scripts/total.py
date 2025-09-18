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
from umap import UMAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('total')

# Hard-coded absolute roots (explicitly set to user's runtime paths)
PROJECT_ROOT = Path('/home/scratch/samyak/temp/multilingual_antonym_detection')
ASSETS_ANALYSIS = Path('/home/scratch/samyak/temp/multilingual_antonym_detection/assets/analysis')
TRAINED_BERT_ANALYSIS = Path('/home/scratch/samyak/temp/multilingual_antonym_detection/models/trained/bert/analysis')
TRAINED_DUAL_ANALYSIS = Path('/home/scratch/samyak/temp/multilingual_antonym_detection/models/trained/dual_encoder/analysis')
OUT_CSV = ASSETS_ANALYSIS / 'final_summary.csv'
OUT_PNG = ASSETS_ANALYSIS / 'final_summary.png'

# Log paths for diagnostics
logger.info(f'Using PROJECT_ROOT={PROJECT_ROOT}')
logger.info(f'ASSETS_ANALYSIS={ASSETS_ANALYSIS}')
logger.info(f'TRAINED_BERT_ANALYSIS={TRAINED_BERT_ANALYSIS}')
logger.info(f'TRAINED_DUAL_ANALYSIS={TRAINED_DUAL_ANALYSIS}')


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


def aggregate(languages=None):
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', nargs='+', help='Optional list of languages to aggregate')
    parser.add_argument('--recompile', action='store_true', help='Recompute combined embeddings (t-SNE/UMAP) and recompile final plots')
    args = parser.parse_args()
    df = aggregate(args.languages)
    print(df)
    if args.recompile:
        # recompute combined embeddings for all languages and regenerate combined plots
        out_dir = ASSETS_ANALYSIS / 'combined_embeddings'
        logger.info('Recomputing combined embeddings (t-SNE/UMAP) for all languages...')
        plot_combined_embeddings(PROJECT_ROOT, out_dir, max_points=5000, per_lang_cap=1000)
        # recompile the comparison plot using the possibly-updated df
        make_comparison_plot(df, OUT_PNG)
