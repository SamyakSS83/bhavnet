#!/usr/bin/env python3
"""Quick scatter plot script: BERT vs Dual metrics from assets/analysis/final_summary.csv

Usage:
    python3 scripts/plot_scatter.py --csv assets/analysis/final_summary.csv --out assets/analysis/bert_vs_dual_scatter.png

This script reads `final_summary.csv`, selects best available metric for BERT/DUAL (prefers bert_f1/dual_f1, falls back to accuracy-like columns), and plots a scatter with language labels and an identity line.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


def main(csv_path: Path, out_png: Path):
    df = pd.read_csv(csv_path)
    # prefer F1 if present, else look for accuracy-like columns
    def pick_metric(row, pref='bert'):
        for col in [f'{pref}_f1', f'{pref}_acc', f'{pref}_accuracy']:
            if col in df.columns and not pd.isna(row.get(col)):
                return float(row[col])
        return None

    x = []
    y = []
    labels = []
    for _, r in df.iterrows():
        bx = pick_metric(r, 'bert')
        dy = pick_metric(r, 'dual')
        if bx is None or dy is None:
            continue
        x.append(bx)
        y.append(dy)
        labels.append(r['language'])

    if not x:
        print('No data points found in CSV for plotting')
        return

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=60)
    # identity line
    mmin = min(min(x), min(y))
    mmax = max(max(x), max(y))
    plt.plot([mmin, mmax], [mmin, mmax], linestyle='--', color='gray')
    for xi, yi, lab in zip(x, y, labels):
        plt.text(xi, yi, lab, fontsize=9, ha='center', va='bottom')
    plt.xlabel('BERT Accuracy / F1')
    plt.ylabel('Dual Encoder Accuracy / F1')
    plt.title('BERT vs Dual Encoder Accuracy')
    plt.xlim(mmin - 0.05, mmax + 0.05)
    plt.ylim(mmin - 0.05, mmax + 0.05)
    plt.grid(True, linestyle=':')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f'Wrote scatter to {out_png}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('assets/analysis/final_summary.csv'))
    parser.add_argument('--out', type=Path, default=Path('assets/analysis/bert_vs_dual_scatter.png'))
    args = parser.parse_args()
    main(args.csv, args.out)
