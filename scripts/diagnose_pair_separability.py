"""Diagnostic: train a lightweight linear probe on pair features built from precomputed NPZ embeddings.

Usage:
    python scripts/diagnose_pair_separability.py --features path/to/lang_embeddings.npz --data_dir datasets/<lang> --language german

Outputs train/val/test accuracy and macro-F1.
"""
import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def load_npz(npz_path: Path):
    arr = np.load(npz_path, allow_pickle=True)
    X = arr['embeddings']
    words = arr['words'].tolist()
    w2i = {w: i for i, w in enumerate(words)}
    return X, words, w2i


def build_pair_features(X, pairs):
    # pairs: list of (i,j)
    feats = []
    for i, j in pairs:
        vi = X[i]
        vj = X[j]
        absdiff = np.abs(vi - vj)
        had = vi * vj
        feats.append(np.concatenate([absdiff, had], axis=0))
    return np.vstack(feats) if len(feats) else np.zeros((0, X.shape[1] * 2))


def read_pairs(file_path: Path, w2i):
    pairs = []
    labels = []
    if not file_path.exists():
        return pairs, labels
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            w1, w2, lab = parts[0], parts[1], parts[2]
            if w1 in w2i and w2 in w2i:
                pairs.append((w2i[w1], w2i[w2]))
                labels.append(int(lab))
    return pairs, labels


def summarize(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    print(f"{name} accuracy={acc:.4f} macroF1={f1m:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--data_dir', default='datasets')
    ap.add_argument('--language', required=True)
    ap.add_argument('--C', type=float, default=1.0, help='Inverse regularization for LogisticRegression')
    ap.add_argument('--max_iter', type=int, default=200)
    args = ap.parse_args()

    npz_path = Path(args.features)
    X, words, w2i = load_npz(npz_path)

    data_dir = Path(args.data_dir) / args.language
    tr_pairs, tr_labels = read_pairs(data_dir / 'train.txt', w2i)
    va_pairs, va_labels = read_pairs(data_dir / 'val.txt', w2i)
    te_pairs, te_labels = read_pairs(data_dir / 'test.txt', w2i)

    print(f"Loaded nodes={len(words)} train={len(tr_pairs)} val={len(va_pairs)} test={len(te_pairs)}")

    X_tr = build_pair_features(X, tr_pairs)
    X_va = build_pair_features(X, va_pairs)
    X_te = build_pair_features(X, te_pairs)

    y_tr = np.array(tr_labels, dtype=np.int64)
    y_va = np.array(va_labels, dtype=np.int64)
    y_te = np.array(te_labels, dtype=np.int64)

    # Standardize
    scaler = StandardScaler()
    if X_tr.shape[0] > 0:
        scaler.fit(X_tr)
        X_tr = scaler.transform(X_tr)
    if X_va.shape[0] > 0:
        X_va = scaler.transform(X_va)
    if X_te.shape[0] > 0:
        X_te = scaler.transform(X_te)

    # Use balanced class weight to address imbalance
    clf = LogisticRegression(C=args.C, class_weight='balanced', max_iter=args.max_iter, solver='liblinear')

    if X_tr.shape[0] == 0:
        print('No training pairs found; aborting')
        return

    clf.fit(X_tr, y_tr)

    y_tr_pred = clf.predict(X_tr)
    summarize('train', y_tr, y_tr_pred)

    if X_va.shape[0] > 0:
        y_va_pred = clf.predict(X_va)
        summarize('val', y_va, y_va_pred)

    if X_te.shape[0] > 0:
        y_te_pred = clf.predict(X_te)
        summarize('test', y_te, y_te_pred)


if __name__ == '__main__':
    main()
