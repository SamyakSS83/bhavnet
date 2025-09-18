import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import NearestNeighbors

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


def _compute_knn_edges(
    x: np.ndarray,
    node_indices: np.ndarray,
    k: int,
    metric: str = "cosine",
) -> List[Tuple[int, int]]:
    if k <= 0:
        return []
    # Fit on selected nodes only
    feats = x[node_indices]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(node_indices)), metric=metric)
    nbrs.fit(feats)
    dists, inds = nbrs.kneighbors(feats, return_distance=True)
    edges = []
    for src_pos, neigh_list in enumerate(inds):
        src = int(node_indices[src_pos])
        for nb_pos in neigh_list[1:]:  # skip self (first)
            dst = int(node_indices[int(nb_pos)])
            if src != dst:
                edges.append((src, dst))
    return edges


def build_graph(
    x: np.ndarray,
    train_edges: List[Tuple[int, int]],
    undirected: bool = True,
    knn_k: int = 0,
    knn_metric: str = "cosine",
    knn_train_nodes_only: bool = True,
    x_alt: Optional[np.ndarray] = None,
) -> Data:
    # Use only training edges for message passing to avoid leakage
    if len(train_edges) == 0:
        raise ValueError("No training edges found to build the graph.")
    src = [i for i, j in train_edges]
    dst = [j for i, j in train_edges]

    # kNN augmentation on selected node set
    if knn_k and knn_k > 0:
        if knn_train_nodes_only:
            nodes = sorted(set(src) | set(dst))
        else:
            nodes = list(range(x.shape[0]))
        knn_edges = _compute_knn_edges(x, np.array(nodes, dtype=np.int64), knn_k, knn_metric)
        # Merge edges, avoid duplicates
        base_edges = set((int(a), int(b)) for a, b in zip(src, dst))
        for a, b in knn_edges:
            if (a, b) not in base_edges:
                src.append(a)
                dst.append(b)
                base_edges.add((a, b))

    if undirected:
        src0, dst0 = list(src), list(dst)
        src = src0 + dst0
        dst = dst0 + src0
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index)
    if x_alt is not None:
        data.x_alt = torch.tensor(x_alt, dtype=torch.float)
    return data


class GNNLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        out_channels = max(1, hidden // heads)
        self.hidden_out = out_channels * heads
        self.conv = TransformerConv(hidden, out_channels, heads=heads, dropout=dropout)
        self.proj = nn.Linear(self.hidden_out, hidden) if self.hidden_out != hidden else nn.Identity()
        self.ln = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)
        if self.hidden_out != x.size(-1):
            h = self.proj(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return self.ln(x + h)


class FeatureMixer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, in_dim_alt: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.has_alt = in_dim_alt is not None
        self.proj_main = nn.Linear(in_dim, hidden)
        if self.has_alt:
            self.proj_alt = nn.Linear(in_dim_alt, hidden)
            self.gate = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.Sigmoid()
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_alt: Optional[torch.Tensor] = None) -> torch.Tensor:
        z0 = self.proj_main(x)
        if self.has_alt and x_alt is not None:
            z1 = self.proj_alt(x_alt)
            g = self.gate(torch.cat([z0, z1], dim=-1))
            z = g * z0 + (1.0 - g) * z1
        else:
            z = z0
        return self.dropout(F.gelu(z))


class LargeGraphDualEncoder(nn.Module):
    """
    Large-graph GNN: optional feature mixer -> stacked residual TransformerConv + LayerNorm -> dual projections.
    Edge classification via bilinear scores in syn/ant spaces with gated fusion.
    """
    def __init__(
        self,
        in_dim: int = 768,
        hidden: int = 512,
        gnn_heads: int = 4,
        gnn_layers: int = 6,
        dropout: float = 0.2,
        in_dim_alt: Optional[int] = None,
    ):
        super().__init__()
        self.mixer = FeatureMixer(in_dim, hidden, in_dim_alt=in_dim_alt, dropout=dropout)
        self.gnn_layers = nn.ModuleList([GNNLayer(hidden, gnn_heads, dropout) for _ in range(gnn_layers)])

        # Dual projections
        self.syn_proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.ant_proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)
        )

        # Bilinear scorers
        self.W_syn = nn.Parameter(torch.empty(hidden, hidden))
        self.W_ant = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.xavier_uniform_(self.W_syn)
        nn.init.xavier_uniform_(self.W_ant)

        # Gate over pair features from both spaces
        pair_dim = hidden * 4  # [syn abs | syn had | ant abs | ant had]
        self.gate_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def node_encode(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        x_alt = getattr(data, 'x_alt', None)
        h = self.mixer(x, x_alt)
        ei = data.edge_index
        for layer in self.gnn_layers:
            h = layer(h, ei)
        z_syn = self.syn_proj(h)
        z_ant = self.ant_proj(h)
        return z_syn, z_ant

    def forward(self, data: Data, edges: torch.Tensor) -> torch.Tensor:
        z_syn, z_ant = self.node_encode(data)
        i, j = edges[0], edges[1]
        vi_syn, vj_syn = z_syn[i], z_syn[j]
        vi_ant, vj_ant = z_ant[i], z_ant[j]

        # Bilinear scores (efficient, O(E))
        s_syn = ((vi_syn @ self.W_syn) * vj_syn).sum(dim=-1)
        s_ant = ((vi_ant @ self.W_ant) * vj_ant).sum(dim=-1)

        # Pair features for gate
        syn_abs = torch.abs(vi_syn - vj_syn)
        syn_hadamard = vi_syn * vj_syn
        ant_abs = torch.abs(vi_ant - vj_ant)
        ant_hadamard = vi_ant * vj_ant
        pair = torch.cat([syn_abs, syn_hadamard, ant_abs, ant_hadamard], dim=-1)
        g = self.gate_mlp(pair).squeeze(-1)  # in [0,1]

        # Fuse: antonymy should rely more on ant score; reduce effect of syn (negatively)
        logit = g * s_ant + (1.0 - g) * (-s_syn)
        return logit


def _edge_tensor(edge_list: List[Tuple[int, int]], device) -> torch.Tensor:
    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    ei = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    return ei


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return (a_n * b_n).sum(dim=-1).clamp(-1.0, 1.0)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
):
    # Sigmoid + BCE
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy(prob, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    mod = (1 - p_t) ** gamma
    loss = (alpha * mod * ce)
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def train_language(
    npz_path: Path,
    data_dir: Path,
    out_dir: Path,
    hidden=512,
    layers=6,
    heads=4,
    dropout=0.2,
    lr=1e-3,
    epochs=10,
    language="lang",
    # Graph opts
    knn_k: int = 0,
    knn_metric: str = 'cosine',
    knn_train_nodes_only: bool = True,
    # Losses
    use_focal: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    lambda_syn: float = 0.0,
    lambda_ant: float = 0.0,
    margin_pos: float = 0.5,
    margin_neg: float = 0.0,
    ortho_lambda: float = 0.0,
    # Train tweaks
    edge_dropout: float = 0.0,
    cosine_lr: bool = False,
    # Optional alternate features
    npz_path_alt: Optional[Path] = None,
):
    X, words, w2i = load_word_features(npz_path)
    X_alt = None
    if npz_path_alt is not None:
        X_alt2, words2, _ = load_word_features(npz_path_alt)
        if words2 != words:
            logger.warning("features_alt words do not match main features; ignoring alt features")
        else:
            X_alt = X_alt2
    train_edges, train_labels = read_pairs(data_dir / 'train.txt', w2i)
    val_edges, val_labels = read_pairs(data_dir / 'val.txt', w2i)
    test_edges, test_labels = read_pairs(data_dir / 'test.txt', w2i)

    logger.info(f"{language}: nodes={len(words)}, train_edges={len(train_edges)}, val_edges={len(val_edges)}, test_edges={len(test_edges)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = build_graph(
        X,
        train_edges,
        undirected=True,
        knn_k=int(knn_k),
        knn_metric=str(knn_metric),
        knn_train_nodes_only=bool(knn_train_nodes_only),
        x_alt=X_alt,
    )
    data = data.to(device)

    model = LargeGraphDualEncoder(
        in_dim=X.shape[1], hidden=hidden, gnn_heads=heads, gnn_layers=layers, dropout=dropout,
        in_dim_alt=(X_alt.shape[1] if X_alt is not None else None)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    bce = nn.BCEWithLogitsLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs)) if cosine_lr else None

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
        # Edge dropout
        if edge_dropout and tr_e.numel() > 0:
            E = tr_e.shape[1]
            keep = max(1, int((1.0 - edge_dropout) * E))
            perm = torch.randperm(E, device=device)[:keep]
            tr_e_epoch = tr_e[:, perm]
            tr_y_epoch = tr_y[perm]
        else:
            tr_e_epoch = tr_e
            tr_y_epoch = tr_y

        logits = model(data, tr_e_epoch)
        main_loss = focal_bce_with_logits(logits, tr_y_epoch, alpha=focal_alpha, gamma=focal_gamma) if use_focal else bce(logits, tr_y_epoch)

        # Auxiliary metric losses
        aux_loss = torch.tensor(0.0, device=device)
        if (lambda_syn > 0 or lambda_ant > 0) and tr_e_epoch.numel() > 0:
            z_syn, z_ant = model.node_encode(data)
            i, j = tr_e_epoch[0], tr_e_epoch[1]
            y = tr_y_epoch
            cos_syn = _cosine(z_syn[i], z_syn[j])
            cos_ant = _cosine(z_ant[i], z_ant[j])
            # For synonyms (label 0)
            mask_neg = (y < 0.5)
            # For antonyms (label 1)
            mask_pos = (y >= 0.5)
            if lambda_syn > 0:
                loss_syn = torch.tensor(0.0, device=device)
                if mask_neg.any():
                    loss_syn = loss_syn + F.relu(margin_pos - cos_syn[mask_neg]).mean()
                if mask_pos.any():
                    loss_syn = loss_syn + F.relu(cos_syn[mask_pos] - margin_neg).mean()
                aux_loss = aux_loss + lambda_syn * loss_syn
            if lambda_ant > 0:
                loss_ant = torch.tensor(0.0, device=device)
                if mask_pos.any():
                    loss_ant = loss_ant + F.relu(margin_pos - cos_ant[mask_pos]).mean()
                if mask_neg.any():
                    loss_ant = loss_ant + F.relu(cos_ant[mask_neg] - margin_neg).mean()
                aux_loss = aux_loss + lambda_ant * loss_ant

        # Orthogonality regularizer
        ortho_loss = torch.tensor(0.0, device=device)
        if ortho_lambda > 0:
            if isinstance(model.syn_proj[0], nn.Linear) and isinstance(model.ant_proj[0], nn.Linear):
                W_s = model.syn_proj[0].weight  # [H,H]
                W_a = model.ant_proj[0].weight  # [H,H]
                M = W_s @ W_a.T
                ortho_loss = (M.pow(2).sum())
        loss = main_loss + aux_loss + ortho_lambda * ortho_loss
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

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
        logger.info(
            f"[{language}] Epoch {ep}: main={float(main_loss):.4f} aux={float(aux_loss):.4f} "
            f"ortho={float(ortho_loss):.4f} total={float(loss):.4f} val_f1={f1:.4f} lr={opt.param_groups[0]['lr']:.2e}"
        )
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
    ap.add_argument('--features_alt', default=None, help='Optional second NPZ of embeddings to mix')
    ap.add_argument('--data_dir', default='datasets')
    ap.add_argument('--output_dir', default='assets')
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=10)
    # Graph options
    ap.add_argument('--knn_k', type=int, default=0)
    ap.add_argument('--knn_metric', type=str, default='cosine')
    ap.add_argument('--knn_train_nodes_only', action='store_true')
    ap.add_argument('--no-knn_train_nodes_only', dest='knn_train_nodes_only', action='store_false')
    # Losses and regularization
    ap.add_argument('--focal', dest='use_focal', action='store_true')
    ap.add_argument('--no-focal', dest='use_focal', action='store_false')
    ap.add_argument('--focal_alpha', type=float, default=0.25)
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--lambda_syn', type=float, default=0.0)
    ap.add_argument('--lambda_ant', type=float, default=0.0)
    ap.add_argument('--margin_pos', type=float, default=0.5)
    ap.add_argument('--margin_neg', type=float, default=0.0)
    ap.add_argument('--ortho_lambda', type=float, default=0.0)
    ap.add_argument('--edge_dropout', type=float, default=0.0)
    ap.add_argument('--cosine_lr', action='store_true')
    ap.add_argument('--no-cosine_lr', dest='cosine_lr', action='store_false')
    ap.set_defaults(knn_train_nodes_only=True, use_focal=False, cosine_lr=False)
    args = ap.parse_args()

    npz_path = Path(args.features)
    npz_path_alt = Path(args.features_alt) if args.features_alt else None
    data_dir = Path(args.data_dir) / args.language
    out_dir = Path(args.output_dir) / 'analysis' / 'large_graph'

    train_language(
        npz_path,
        data_dir,
        out_dir,
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        language=args.language,
        knn_k=args.knn_k,
        knn_metric=args.knn_metric,
        knn_train_nodes_only=args.knn_train_nodes_only,
        use_focal=args.use_focal,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        lambda_syn=args.lambda_syn,
        lambda_ant=args.lambda_ant,
        margin_pos=args.margin_pos,
        margin_neg=args.margin_neg,
        ortho_lambda=args.ortho_lambda,
        edge_dropout=args.edge_dropout,
        cosine_lr=args.cosine_lr,
        npz_path_alt=npz_path_alt,
    )


if __name__ == '__main__':
    main()
