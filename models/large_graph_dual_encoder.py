import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_word_features(npz_path: Path) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    arr = np.load(npz_path, allow_pickle=True)
    X = arr['embeddings']
    words = arr['words'].tolist()
    w2i = {w: i for i, w in enumerate(words)}
    return X, words, w2i


def read_pairs(file_path: Path, w2i: Dict[str, int]) -> Tuple[List[Tuple[int, int]], List[int]]:
    edges = []
    labels = []
    if not file_path.exists():
        return edges, labels
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            w1, w2, lab = parts[0], parts[1], parts[2]
            if w1 in w2i and w2 in w2i:
                i, j = w2i[w1], w2i[w2]
                edges.append((i, j))
                labels.append(int(lab))
    return edges, labels


def build_graph(x: np.ndarray, train_edges: List[Tuple[int, int]], undirected: bool = True) -> Data:
    # Use only training edges for message passing to avoid leakage
    if len(train_edges) == 0:
        raise ValueError("No training edges found to build the graph.")
    src = [i for i, j in train_edges]
    dst = [j for i, j in train_edges]
    if undirected:
        src = src + dst
        dst = dst + src[:len(dst)]  # original src
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index)


class LargeGraphDualEncoder(nn.Module):
    """
    Large-graph GNN: node features -> GNN -> dual projections (syn/ant).
    Edge classification uses pair features from both spaces.
    """
    def __init__(self, in_dim=768, hidden=256, gnn_heads=2, gnn_layers=2, dropout=0.2):
        super().__init__()
        conv_in = in_dim
        self.gnn_layers = nn.ModuleList()
        for li in range(gnn_layers):
            conv = TransformerConv(conv_in, hidden, heads=gnn_heads, dropout=dropout)
            self.gnn_layers.append(conv)
            conv_in = hidden * gnn_heads
        self.post = nn.Linear(conv_in, hidden)

        self.syn_proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.ant_proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)
        )

        # Edge scorer over pair features from both spaces
        pair_in = hidden * 4  # [syn abs diff | syn hadamard | ant abs diff | ant hadamard]
        self.edge_mlp = nn.Sequential(
            nn.Linear(pair_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, data: Data, edges: torch.Tensor):
        x, ei = data.x, data.edge_index
        h = x
        for conv in self.gnn_layers:
            h = conv(h, ei)
            h = F.relu(h)
        h = self.post(h)
        h = F.relu(h)

        z_syn = self.syn_proj(h)
        z_ant = self.ant_proj(h)

        i, j = edges[0], edges[1]
        vi_syn, vj_syn = z_syn[i], z_syn[j]
        vi_ant, vj_ant = z_ant[i], z_ant[j]

        syn_abs = torch.abs(vi_syn - vj_syn)
        syn_hadamard = vi_syn * vj_syn
        ant_abs = torch.abs(vi_ant - vj_ant)
        ant_hadamard = vi_ant * vj_ant
        pair = torch.cat([syn_abs, syn_hadamard, ant_abs, ant_hadamard], dim=-1)
        logit = self.edge_mlp(pair).squeeze(-1)
        return logit


def _edge_tensor(edge_list: List[Tuple[int, int]], device) -> torch.Tensor:
    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    ei = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    return ei


def train_language(npz_path: Path, data_dir: Path, out_dir: Path, hidden=256, layers=2, heads=2, dropout=0.2, lr=1e-3, epochs=10, language="lang"):
    X, words, w2i = load_word_features(npz_path)
    train_edges, train_labels = read_pairs(data_dir / 'train.txt', w2i)
    val_edges, val_labels = read_pairs(data_dir / 'val.txt', w2i)
    test_edges, test_labels = read_pairs(data_dir / 'test.txt', w2i)

    logger.info(f"{language}: nodes={len(words)}, train_edges={len(train_edges)}, val_edges={len(val_edges)}, test_edges={len(test_edges)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = build_graph(X, train_edges, undirected=True)
    data = data.to(device)

    model = LargeGraphDualEncoder(in_dim=X.shape[1], hidden=hidden, gnn_heads=heads, gnn_layers=layers, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    crit = nn.BCEWithLogitsLoss()

    tr_e = _edge_tensor(train_edges, device)
    tr_y = torch.tensor(train_labels, dtype=torch.float, device=device)
    va_e = _edge_tensor(val_edges, device)
    va_y = torch.tensor(val_labels, dtype=torch.float, device=device) if len(val_labels) else None
    te_e = _edge_tensor(test_edges, device)
    te_y = torch.tensor(test_labels, dtype=torch.float, device=device) if len(test_labels) else None

    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_path = out_dir / f"best_{language}_largegraph_dual.pt"

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(data, tr_e)
        loss = crit(logits, tr_y)
        loss.backward()
        opt.step()

        # simple eval on val
        model.eval()
        with torch.no_grad():
            if va_y is not None and va_e.numel() > 0:
                v_logits = model(data, va_e)
                v_pred = (torch.sigmoid(v_logits) > 0.5).long().cpu().numpy()
                v_true = va_y.long().cpu().numpy()
                f1 = f1_score(v_true, v_pred, average='macro')
            else:
                f1 = 0.0
        logger.info(f"[{language}] Epoch {ep}: loss={loss.item():.4f} val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)

    # Final test
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    with torch.no_grad():
        t_logits = model(data, te_e)
        t_probs = torch.sigmoid(t_logits).cpu().numpy()
        t_pred = (t_probs > 0.5).astype(int)
        t_true = te_y.long().cpu().numpy() if te_y is not None else np.zeros_like(t_pred)

    acc = accuracy_score(t_true, t_pred) if len(t_true) else 0.0
    try:
        f1m = f1_score(t_true, t_pred, average='macro') if len(t_true) else 0.0
    except Exception:
        f1m = 0.0
    logger.info(f"[{language}] TEST acc={acc:.4f} macroF1={f1m:.4f}")

    # Save per-edge predictions
    import pandas as pd
    rows = []
    for (i, j), y, p in zip(test_edges, t_true, t_probs):
        rows.append({"word1": words[i], "word2": words[j], "label": int(y), "prob_antonym": float(p)})
    df = pd.DataFrame(rows)
    out_csv = out_dir / f"{language}_largegraph_dual_predictions.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved predictions to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Train large-graph dual-encoder edge classifier using precomputed word features")
    ap.add_argument('--language', required=True)
    ap.add_argument('--features', required=True, help='Path to NPZ with embeddings and words')
    ap.add_argument('--data_dir', default='datasets')
    ap.add_argument('--output_dir', default='assets')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=10)
    args = ap.parse_args()

    npz_path = Path(args.features)
    data_dir = Path(args.data_dir) / args.language
    out_dir = Path(args.output_dir) / 'analysis' / 'large_graph'

    train_language(npz_path, data_dir, out_dir, hidden=args.hidden, layers=args.layers, heads=args.heads,
                   dropout=args.dropout, lr=args.lr, epochs=args.epochs, language=args.language)


if __name__ == '__main__':
    main()
