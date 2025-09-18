#!/usr/bin/env python3
"""Error analysis tool comparing BERT vs Dual-Encoder predictions.

Generates per-language and aggregated analysis across multiple bases:
 - by true label (antonym vs not)
 - by normalized edit-distance bins
 - by character-Jaccard bins
 - by absolute length-difference bins

Outputs:
 - assets/analysis/<lang>/<lang>_error_analysis.csv  (per-sample with features and category)
 - assets/analysis/error_analysis_summary.csv (aggregated across languages)
 - PNGs under assets/analysis/error_analysis/ for visual summaries

Usage:
    python3 scripts/error_analysis.py --languages english dutch ...

This script only depends on pandas/numpy/matplotlib/seaborn (no external Levenshtein lib).
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math
import csv

DEFAULT_LANGS = ['english', 'dutch', 'french', 'italian', 'russian', 'spanish', 'portuguese', 'german']
ASSETS_ROOT = Path('assets/analysis')


# simple Levenshtein distance implementation
def edit_distance(a: str, b: str) -> int:
    if a is None: a = ''
    if b is None: b = ''
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = [list(range(lb+1))] + [[i+1] + [0]*lb for i in range(la)]
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[la][lb]


def char_jaccard(a: str, b: str) -> float:
    sa = set(a or '')
    sb = set(b or '')
    if not sa and not sb:
        return 1.0
    inter = sa.intersection(sb)
    uni = sa.union(sb)
    return len(inter) / len(uni) if uni else 0.0


def safe_read_preds(lang: str):
    """Try to locate bert and dual prediction CSVs for a language.
    Return (bert_df, dual_df) or (None,None) if missing.
    """
    a_dir = ASSETS_ROOT / lang
    candidates = [a_dir / f"{lang}_bert_predictions.csv",
                  a_dir / f"{lang}_baseline_probe_predictions.csv", # sometimes bert baseline saved here
                  Path('models/trained/bert/analysis') / lang / f"{lang}_bert_predictions.csv"]
    bert_df = None
    for c in candidates:
        if c.exists():
            try:
                bert_df = pd.read_csv(c)
                break
            except Exception:
                continue
    dual_candidates = [a_dir / f"{lang}_dual_predictions.csv",
                       Path('models/trained/dual_encoder/analysis') / lang / f"{lang}_dual_predictions.csv"]
    dual_df = None
    for c in dual_candidates:
        if c.exists():
            try:
                dual_df = pd.read_csv(c)
                break
            except Exception:
                continue
    return bert_df, dual_df


def merge_preds(bert_df: pd.DataFrame, dual_df: pd.DataFrame) -> pd.DataFrame:
    """Merge on word1, word2 (best-effort). If merge fails, align by index.
    Expects columns: word1, word2, label, pred
    """
    # normalize column names
    def norm(df):
        df = df.copy()
        for c in ['word1','word2','label','pred']:
            if c not in df.columns:
                # try variants
                for alt in df.columns:
                    if alt.lower() == c:
                        df.rename(columns={alt: c}, inplace=True)
                        break
        return df
    b = norm(bert_df) if bert_df is not None else None
    d = norm(dual_df) if dual_df is not None else None
    if b is None and d is None:
        return None
    if b is None:
        b = pd.DataFrame(columns=['word1','word2','label','pred'])
    if d is None:
        d = pd.DataFrame(columns=['word1','word2','label','pred'])

    # try merge on word1+word2
    if 'word1' in b.columns and 'word2' in b.columns and 'word1' in d.columns and 'word2' in d.columns:
        # create keys
        b['_key'] = b['word1'].astype(str) + '||' + b['word2'].astype(str)
        d['_key'] = d['word1'].astype(str) + '||' + d['word2'].astype(str)
        merged = pd.merge(b, d, on='_key', suffixes=('_bert','_dual'), how='outer')
        # if label columns differ, prefer bert label if available else dual
        if 'label_bert' not in merged.columns and 'label' in merged.columns:
            merged.rename(columns={'label':'label_bert'}, inplace=True)
        if 'label_dual' not in merged.columns and 'label' in merged.columns:
            merged.rename(columns={'label':'label_dual'}, inplace=True)
        # unify label
        def pick_label(row):
            for c in ['label_bert','label_dual','label_bert_x','label_x','label_y','label_dual_y']:
                if c in row and not pd.isna(row[c]):
                    return int(row[c])
            return np.nan
        merged['label'] = merged.apply(pick_label, axis=1)
        # ensure word columns
        merged['word1'] = merged.get('word1_bert').fillna(merged.get('word1_dual'))
        merged['word2'] = merged.get('word2_bert').fillna(merged.get('word2_dual'))
        # preds
        merged['pred_bert'] = merged.get('pred_bert') if 'pred_bert' in merged.columns else merged.get('pred')
        merged['pred_dual'] = merged.get('pred_dual') if 'pred_dual' in merged.columns else merged.get('pred')
        # final cleanup
        res = merged[['word1','word2','label','pred_bert','pred_dual']].copy()
        res = res.rename(columns={'pred_bert':'bert_pred','pred_dual':'dual_pred'})
        return res
    else:
        # fallback: align by index
        n = max(len(b), len(d))
        b2 = b.reset_index(drop=True)
        d2 = d.reset_index(drop=True)
        # pad
        for col in ['word1','word2','label','pred']:
            if col not in b2.columns:
                b2[col] = [np.nan]*len(b2)
            if col not in d2.columns:
                d2[col] = [np.nan]*len(d2)
        b2 = b2.reindex(range(n)).reset_index(drop=True)
        d2 = d2.reindex(range(n)).reset_index(drop=True)
        res = pd.DataFrame({
            'word1': b2['word1'].fillna(d2['word1']),
            'word2': b2['word2'].fillna(d2['word2']),
            'label': b2['label'].fillna(d2['label']).astype(float).replace({np.nan: None}),
            'bert_pred': b2['pred'],
            'dual_pred': d2['pred']
        })
        return res


def analyze_language(lang: str, out_root: Path):
    bert_df, dual_df = safe_read_preds(lang)
    if bert_df is None and dual_df is None:
        print(f'No predictions found for {lang}, skipping')
        return None
    merged = merge_preds(bert_df, dual_df)
    if merged is None or merged.empty:
        print(f'Empty merged predictions for {lang}, skipping')
        return None

    # normalize columns
    merged['word1'] = merged['word1'].astype(str)
    merged['word2'] = merged['word2'].astype(str)
    merged['label'] = merged['label'].apply(lambda x: int(x) if (not pd.isna(x)) else np.nan)
    merged['bert_pred'] = merged['bert_pred'].apply(lambda x: int(x) if (not pd.isna(x)) else np.nan)
    merged['dual_pred'] = merged['dual_pred'].apply(lambda x: int(x) if (not pd.isna(x)) else np.nan)

    # compute features
    ED = []
    EDN = []
    J = []
    LD = []
    for w1, w2 in zip(merged['word1'], merged['word2']):
        e = edit_distance(w1, w2)
        ED.append(e)
        maxlen = max(1, len(w1), len(w2))
        EDN.append(e / maxlen)
        J.append(char_jaccard(w1, w2))
        LD.append(abs(len(w1) - len(w2)))
    merged['edit_distance'] = ED
    merged['edit_distance_norm'] = EDN
    merged['char_jaccard'] = J
    merged['len_diff'] = LD

    # correctness categories
    merged['bert_correct'] = merged.apply(lambda r: (not pd.isna(r['bert_pred'])) and (r['bert_pred'] == r['label']), axis=1)
    merged['dual_correct'] = merged.apply(lambda r: (not pd.isna(r['dual_pred'])) and (r['dual_pred'] == r['label']), axis=1)

    def cat(r):
        if r['bert_correct'] and r['dual_correct']:
            return 'both'
        if r['bert_correct'] and not r['dual_correct']:
            return 'bert_only'
        if (not r['bert_correct']) and r['dual_correct']:
            return 'dual_only'
        return 'neither'
    merged['category'] = merged.apply(cat, axis=1)

    # bins for analyses
    merged['ed_bin'] = pd.cut(merged['edit_distance_norm'], bins=[-0.01,0.0,0.2,0.4,0.6,0.8,1.01], labels=['0','(0,0.2]','(0.2,0.4]','(0.4,0.6]','(0.6,0.8]','(0.8,1.0]'])
    merged['jaccard_bin'] = pd.cut(merged['char_jaccard'], bins=[-0.01,0.0,0.2,0.4,0.6,0.8,1.01], labels=['0','(0,0.2]','(0.2,0.4]','(0.4,0.6]','(0.6,0.8]','(0.8,1.0]'])
    merged['len_diff_bin'] = pd.cut(merged['len_diff'], bins=[-0.01,0,1,2,5,1000], labels=['0','1','2','3-5','6+'])

    # save per-sample enriched CSV
    out_lang_dir = out_root / lang
    out_lang_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_lang_dir / f'{lang}_error_analysis.csv'
    merged.to_csv(per_sample_path, index=False)

    print(f'Wrote per-sample error analysis to {per_sample_path}')

    # produce summary tables for bases
    summaries = []
    def summarize(groupby_col, label):
        g = merged.groupby(groupby_col)
        rows = []
        for key, sub in g:
            n = len(sub)
            bert_acc = (sub['bert_correct'].sum() / n) if n>0 else np.nan
            dual_acc = (sub['dual_correct'].sum() / n) if n>0 else np.nan
            dual_only = (sub['category'] == 'dual_only').sum()
            bert_only = (sub['category'] == 'bert_only').sum()
            both = (sub['category'] == 'both').sum()
            neither = (sub['category'] == 'neither').sum()
            rows.append({'language': lang, 'basis': label, 'bin': str(key), 'n': n, 'bert_acc': bert_acc, 'dual_acc': dual_acc, 'dual_only': dual_only, 'bert_only': bert_only, 'both': both, 'neither': neither})
        return pd.DataFrame(rows)

    summaries.append(summarize('label', 'label'))
    summaries.append(summarize('ed_bin', 'edit_distance_norm'))
    summaries.append(summarize('jaccard_bin', 'char_jaccard'))
    summaries.append(summarize('len_diff_bin', 'len_diff'))

    summary_df = pd.concat(summaries, ignore_index=True)
    summary_path = out_lang_dir / f'{lang}_error_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'Wrote per-language summary to {summary_path}')

    # produce a few plots per language
    plt.figure(figsize=(8,4))
    sns.barplot(data=summary_df[summary_df['basis']=='edit_distance_norm'], x='bin', y='dual_acc', color='C0')
    plt.title(f'{lang} - Dual accuracy by edit-distance bin')
    plt.tight_layout()
    plt.savefig(out_lang_dir / f'{lang}_dual_acc_by_ed_bin.png', dpi=150)
    plt.close()

    # stacked category by bin (example for jaccard)
    cattab = merged.pivot_table(index='jaccard_bin', columns='category', values='word1', aggfunc='count').fillna(0)
    if not cattab.empty:
        cattab = cattab[['both','dual_only','bert_only','neither']] if set(['both','dual_only','bert_only','neither']).issubset(cattab.columns) else cattab
        cattab.plot(kind='bar', stacked=True, figsize=(8,4))
        plt.title(f'{lang} - category distribution by char-jaccard bin')
        plt.tight_layout()
        plt.savefig(out_lang_dir / f'{lang}_category_by_jaccard.png', dpi=150)
        plt.close()

    return summary_df


def aggregate_all(summaries, out_root: Path):
    # concat and write global summary
    all_df = pd.concat(summaries, ignore_index=True)
    outp = out_root / 'error_analysis_summary.csv'
    all_df.to_csv(outp, index=False)
    print(f'Wrote aggregate summary to {outp}')

    # high-level plot: bert vs dual acc for each (language,basis,bin)
    # Create independent scatter plots per basis (don't overlay)
    bases = sorted(all_df['basis'].dropna().unique())
    for basis in bases:
        sub = all_df[all_df['basis'] == basis]
        if sub.empty:
            continue
        plt.figure(figsize=(6,6))
        sns.scatterplot(data=sub, x='bert_acc', y='dual_acc')
        plt.plot([0,1],[0,1], linestyle='--', color='gray')
        plt.xlabel('BERT acc')
        plt.ylabel('Dual acc')
        plt.title(f'BERT vs Dual accuracy — basis: {basis}')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.tight_layout()
        fname = out_root / f'bert_vs_dual_by_bin_{basis}.png'
        # sanitize filename
        fname = Path(str(fname).replace(' ', '_').replace('/', '_'))
        plt.savefig(fname, dpi=150)
        plt.close()

    return all_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', nargs='+', help='Languages to analyze', default=None)
    args = parser.parse_args()
    langs = [l.lower() for l in (args.languages if args.languages else DEFAULT_LANGS)]
    out_root = ASSETS_ROOT / 'error_analysis'
    out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for lang in langs:
        s = analyze_language(lang, out_root)
        if s is not None:
            summaries.append(s)
    if summaries:
        agg = aggregate_all(summaries, out_root)
    else:
        print('No summaries generated')
