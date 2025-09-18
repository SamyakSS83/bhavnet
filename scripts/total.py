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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('total')

# Default absolute roots (adjust if your project root differs)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ANALYSIS = PROJECT_ROOT / 'assets' / 'analysis'
TRAINED_BERT_ANALYSIS = PROJECT_ROOT / 'models' / 'trained' / 'bert' / 'analysis'
TRAINED_DUAL_ANALYSIS = PROJECT_ROOT / 'models' / 'trained' / 'dual_encoder' / 'analysis'
OUT_CSV = ASSETS_ANALYSIS / 'final_summary.csv'
OUT_PNG = ASSETS_ANALYSIS / 'final_summary.png'


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
                    row['bert_acc'] = float((df['label'] == df['pred']).mean())
            except Exception:
                pass
        if dual_pred.exists():
            row['dual_pred_csv'] = str(dual_pred.resolve())
            try:
                df2 = pd.read_csv(dual_pred)
                if 'label' in df2.columns and 'pred' in df2.columns:
                    row['dual_acc'] = float((df2['label'] == df2['pred']).mean())
            except Exception:
                pass
        if baseline_report.exists():
            row['baseline_report'] = str(baseline_report.resolve())
            # try to read accuracy from file
            try:
                txt = baseline_report.read_text(encoding='utf-8')
                for line in txt.splitlines():
                    if 'Accuracy:' in line:
                        try:
                            row['baseline_acc'] = float(line.split('Accuracy:')[-1].strip())
                        except Exception:
                            pass
            except Exception:
                pass

        # trained model analysis
        bert_trained_dir = TRAINED_BERT_ANALYSIS / lang
        dual_trained_dir = TRAINED_DUAL_ANALYSIS / lang
        if bert_trained_dir.exists():
            row['trained_bert_metrics'] = read_metric_from_trained(bert_trained_dir, 'bert')
        if dual_trained_dir.exists():
            row['trained_dual_metrics'] = read_metric_from_trained(dual_trained_dir, 'dual')

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
        bert_vals = df.get('bert_acc', pd.Series([np.nan]*len(df))).fillna(0)
        dual_vals = df.get('dual_acc', pd.Series([np.nan]*len(df))).fillna(0)
        base_vals = df.get('baseline_acc', pd.Series([np.nan]*len(df))).fillna(0)
        plt.bar(idx - width, bert_vals, width, label='BERT')
        plt.bar(idx, dual_vals, width, label='Dual')
        plt.bar(idx + width, base_vals, width, label='Baseline')
        plt.xticks(idx, df['language'], rotation=45)
        plt.ylabel('Accuracy')
        plt.title('Model comparison across languages')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_PNG)
        logger.info(f"Wrote aggregate plot to {OUT_PNG}")
    except Exception as e:
        logger.warning(f"Failed to create comparison plot: {e}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', nargs='+', help='Optional list of languages to aggregate')
    args = parser.parse_args()
    df = aggregate(args.languages)
    print(df)
