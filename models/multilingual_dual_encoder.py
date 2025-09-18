import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm
# Set matplotlib backend before importing pyplot to avoid segfaults
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml
import logging
import time
import csv
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WordPairGraphDataset(Dataset):
    """
    Graph dataset for word pairs using pre-trained language model embeddings.
    Each word pair becomes a 2-node graph with bidirectional edges.
    """
    def __init__(self, dataframe, bert_model, tokenizer, device):
        self.data = dataframe
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        word1 = str(row['word1'])
        word2 = str(row['word2'])
        label = int(row['label'])

        # Get embeddings from fine-tuned BERT
        with torch.no_grad():
            inputs1 = self.tokenizer(word1, return_tensors='pt').to(self.device)
            inputs2 = self.tokenizer(word2, return_tensors='pt').to(self.device)
            
            emb1 = self.bert_model(**inputs1).last_hidden_state[:, 0, :].squeeze(0).cpu()
            emb2 = self.bert_model(**inputs2).last_hidden_state[:, 0, :].squeeze(0).cpu()

        # Create graph with two nodes
        x = torch.stack([emb1, emb2])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = GraphData(x=x, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
        # Attach metadata so we can recover words during analysis
        data.word1 = word1
        data.word2 = word2
        return data

class DualEncoderGraphTransformer(nn.Module):
    """
    Dual Encoder Graph Transformer for antonym detection.
    Uses separate synonym and antonym projection branches.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout=0.2):
        super(DualEncoderGraphTransformer, self).__init__()
        
        # Synonym projection branch
        self.syn_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Antonym projection branch
        self.ant_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels * heads)
        
        # Graph transformer layers
        self.conv1 = TransformerConv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout
        )
        self.conv2 = TransformerConv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=1, 
            dropout=dropout
        )
        
        # Classification layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Dual projections
        x_syn = self.syn_proj(x)
        x_ant = self.ant_proj(x)
        
        # Store for margin loss computation
        self.x_syn = x_syn
        self.x_ant = x_ant
        
        # Feature fusion
        x_combined = torch.cat([x_syn, x_ant], dim=1)
        x_fused = self.fusion(x_combined)
        
        # Graph transformer layers
        x = self.conv1(x_fused, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Global pooling and classification
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        
        return self.lin2(x)

class MultilingualDualEncoderTrainer:
    """Trainer for the multilingual dual encoder graph transformer."""

    def __init__(self, config, language, output_dir, use_trained_bert: bool = True):
        self.config = config
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Analysis / assets directories
        self.analysis_dir = self.output_dir / 'analysis'
        self.plots_dir = self.analysis_dir / 'plots'
        self.tables_dir = self.analysis_dir / 'tables'
        self.svg_dir = self.analysis_dir / 'svg'
        self.emb_dir = self.analysis_dir / 'embeddings'
        for d in (self.analysis_dir, self.plots_dir, self.tables_dir, self.svg_dir, self.emb_dir):
            d.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Get configuration
        self.lang_config = config['languages'][language]
        self.training_config = config.get('training', {})
        self.use_trained_bert = use_trained_bert

        # Load pre-trained BERT model
        self.model_name = self.lang_config['bert_model']
        logger.info(f"Loading BERT model: {self.model_name}")
        # Explicit handling/log for English datasets
        if self.language.lower() in ('english', 'en'):
            logger.info("English language selected — will use legacy 'dataset/' for data and pre-trained BERT base (no fine-tuned weights by default)")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Decide whether to load fine-tuned BERT: for non-English, load if trained_bert_path present by default
        if self.language.lower() in ('english', 'en'):
            ft_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
            self.bert_model = self._extract_base_model(ft_model).to(self.device)
        else:
            # For non-English, prefer fine-tuned if available unless explicitly disabled
            trained_path = self.lang_config.get('trained_bert_path', None)
            if trained_path and Path(trained_path).exists() and self.use_trained_bert:
                try:
                    trained_bert_path = Path(trained_path)
                    logger.info(f"Loading fine-tuned BERT from {trained_bert_path}")
                    ft_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
                    ft_model.load_state_dict(torch.load(trained_bert_path, map_location=self.device))
                    self.bert_model = self._extract_base_model(ft_model).to(self.device)
                    logger.info("✓ Successfully loaded fine-tuned BERT model")
                except Exception as e:
                    logger.warning(f"Failed to load fine-tuned BERT from {trained_bert_path}: {e}")
                    logger.info("Falling back to pre-trained BERT")
                    ft_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
                    self.bert_model = self._extract_base_model(ft_model).to(self.device)
            else:
                logger.info("Using pre-trained BERT (not fine-tuned)")
                ft_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
                self.bert_model = self._extract_base_model(ft_model).to(self.device)

        self.bert_model.eval()
        
        # Initialize dual encoder model
        self.model = DualEncoderGraphTransformer(
            in_channels=768,  # BERT hidden size
            hidden_channels=self.training_config.get('hidden_dim', 256),
            out_channels=2,
            heads=2,
            dropout=self.training_config.get('dropout', 0.2)
        ).to(self.device)

    def _extract_base_model(self, ft_model):
        """Extract the underlying base transformer from a SequenceClassification model.

        Many HF models wrap the base model under attributes like `base_model`, `bert`, or `roberta`.
        Return the first matching attribute or the model itself as a fallback.
        """
        # Common attribute names that hold the base encoder
        for attr in ('base_model', 'bert', 'roberta', 'distilbert', 'transformer', 'model'):
            if hasattr(ft_model, attr):
                base = getattr(ft_model, attr)
                logger.info(f"Extracted base model using attribute '{attr}'")
                return base

        # Fallback: try calling the model and inspect output for last_hidden_state
        logger.warning("Could not find a conventional base_model attribute; using the full model as-is")
        return ft_model
        
    def test(self, test_loader, model_path):
        """Test the model and generate detailed results.

        Produces per-sample prediction CSV and returns metrics including accuracy and macro-F1.
        """
        # Load best model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        all_preds = []
        all_labels = []
        per_sample_rows = []

        logger.info("Evaluating dual encoder on test set...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                batch = batch.to(self.device)
                logits = self.model(batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = batch.y.cpu().numpy()

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

                # Recover word1/word2 from batched graphs when possible
                try:
                    if hasattr(batch, 'to_data_list'):
                        data_list = batch.to_data_list()
                        for d_idx, d in enumerate(data_list):
                            w1 = getattr(d, 'word1', None)
                            w2 = getattr(d, 'word2', None)
                            lbl = int(d.y.cpu().numpy()) if hasattr(d, 'y') else None
                            pred = int(preds[d_idx]) if d_idx < len(preds) else None
                            per_sample_rows.append({'word1': w1, 'word2': w2, 'label': lbl, 'pred': pred, 'score': None})
                    else:
                        for i in range(len(preds)):
                            per_sample_rows.append({'word1': None, 'word2': None, 'label': int(labels[i]), 'pred': int(preds[i]), 'score': None})
                except Exception:
                    for i in range(len(preds)):
                        per_sample_rows.append({'word1': None, 'word2': None, 'label': int(labels[i]), 'pred': int(preds[i]), 'score': None})

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        try:
            f1 = float(f1_score(all_labels, all_preds, average='macro')) if len(all_labels) > 0 else 0.0
        except Exception:
            f1 = 0.0

        report = classification_report(
            all_labels,
            all_preds,
            target_names=['Not Antonym', 'Antonym'],
            zero_division=0
        )

        logger.info(f"Dual Encoder Test Accuracy for {self.language}: {accuracy:.4f}")
        logger.info(f"Dual Encoder Test Macro-F1 for {self.language}: {f1:.4f}")
        logger.info(f"Classification Report:\n{report}")

        # Save confusion matrix as text to avoid plotting issues
        try:
            cm = confusion_matrix(all_labels, all_preds)
            cm_file = self.output_dir / f'{self.language}_dual_encoder_confusion_matrix.txt'
            with open(cm_file, 'w') as f:
                f.write(f"Dual Encoder Confusion Matrix for {self.language}:\n")
                f.write(f"{cm}\n")
                f.write(f"Classes: ['Not Antonym', 'Antonym']\n")
            logger.info(f"Saved confusion matrix data to {cm_file}")
        except Exception as e:
            logger.warning(f"Failed to compute/save confusion matrix: {e}")

        # Save per-sample predictions for analysis
        try:
            save_dir = self.analysis_dir / self.language
            save_dir.mkdir(parents=True, exist_ok=True)
            preds_df = pd.DataFrame(per_sample_rows)
            preds_csv = save_dir / f'{self.language}_dual_predictions.csv'
            preds_df.to_csv(preds_csv, index=False)
            logger.info(f"Saved per-sample Dual Encoder predictions to {preds_csv}")
        except Exception as e:
            logger.warning(f"Failed to save per-sample Dual Encoder predictions: {e}")

        return {
            'language': self.language,
            'accuracy': accuracy,
            'f1_macro': f1,
            'classification_report': report
        }
        
        train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = GeoDataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, test_loader
    
    def compute_margin_loss(self, model, data):
        """Compute margin-based loss for dual projections."""
        margin_syn = self.training_config.get('margin_syn', 0.8)
        margin_ant = self.training_config.get('margin_ant', 0.2)
        losses = []
        
        batch = data.batch.cpu().numpy()
        unique_graphs = np.unique(batch)
        
        for g in unique_graphs:
            idx = (data.batch == g).nonzero(as_tuple=False).view(-1)
            if idx.shape[0] != 2:
                continue
            
            graph_idx = g.item() if hasattr(g, 'item') else g
            label = data.y[graph_idx].item()
            
            # Compute similarities in both spaces
            sim_syn = torch.tanh(torch.dot(model.x_syn[idx[0]], model.x_syn[idx[1]]))
            sim_ant = torch.tanh(torch.dot(model.x_ant[idx[0]], model.x_ant[idx[1]]))
            
            if label == 0:  # non-antonym pair
                loss = torch.relu(margin_syn - sim_syn)
            else:  # antonym pair
                loss = torch.relu(sim_ant - margin_ant)
                
            losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=self.device)

    def load_data(self, data_root: str):
        """Load train/test data for the language. For English, prefer legacy dataset/ files.

        Returns (train_df, test_df)
        """
        data_root = Path(data_root)
        if self.language.lower() in ('english', 'en'):
            # Legacy layout under project_root/dataset/*.train/*.test
            legacy = Path(__file__).parent.parent / 'dataset'
            def read_legacy(suffix):
                frames = []
                for kind in ('adjective-pairs', 'noun-pairs', 'verb-pairs'):
                    f = legacy / f"{kind}.{suffix}"
                    if f.exists():
                        try:
                            df = pd.read_csv(f, sep='\t', header=None, names=['word1', 'word2', 'label'])
                            frames.append(df)
                        except Exception:
                            continue
                if frames:
                    return pd.concat(frames, ignore_index=True)
                return pd.DataFrame()

            train_df = read_legacy('train')
            test_df = read_legacy('test')
            return train_df, test_df
        else:
            lang_dir = data_root / self.language
            train_file = lang_dir / 'train.txt'
            test_file = lang_dir / 'test.txt'
            train_df = pd.read_csv(train_file, sep='\t', header=None, names=['word1', 'word2', 'label']) if train_file.exists() else pd.DataFrame()
            test_df = pd.read_csv(test_file, sep='\t', header=None, names=['word1', 'word2', 'label']) if test_file.exists() else pd.DataFrame()
            return train_df, test_df

    def create_dataloaders(self, train_df, test_df):
        """Create GeoDataLoader for train and test graphs."""
        batch_size = self.training_config.get('batch_size', 16)
        # Build datasets: WordPairGraphDataset maps pairs to 2-node graphs using bert_model
        train_dataset = WordPairGraphDataset(train_df, self.bert_model, self.tokenizer, self.device)
        test_dataset = WordPairGraphDataset(test_df, self.bert_model, self.tokenizer, self.device)

        train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = GeoDataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader
    
    def train(self, train_loader, test_loader):
        """Train the dual encoder model with early stopping."""
        num_epochs = self.training_config.get('num_epochs', self.training_config.get('epochs', 10))
        learning_rate = self.training_config.get('learning_rate', self.training_config.get('lr', 1e-4))
        margin_weight = self.training_config.get('margin_weight', 0.5)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_model_path = self.output_dir / f'best_{self.language}_dual_encoder_model.pt'

        # Early stopping parameters
        patience = 5
        no_improvement_count = 0

        logger.info(f"Starting dual encoder training for {num_epochs} epochs with early stopping (patience={patience})...")

        history = {'epoch': [], 'avg_loss': [], 'test_acc': [], 'epoch_time': []}
        total_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.model.train()
            total_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in train_pbar:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = self.model(batch)
                
                # Classification loss
                ce_loss = criterion(out, batch.y)
                
                # Margin loss
                margin_loss = self.compute_margin_loss(self.model, batch)
                
                # Combined loss
                loss = ce_loss + margin_weight * margin_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                train_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'ce_loss': f"{ce_loss.item():.4f}",
                    'margin_loss': f"{margin_loss.item():.4f}"
                })
            
            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            history['epoch'].append(epoch + 1)
            history['avg_loss'].append(avg_loss)
            history['epoch_time'].append(epoch_time)
            
            # Evaluate on test set
            test_acc = self._evaluate(test_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}, "
                f"Loss: {avg_loss:.4f}, "
                f"Test Accuracy: {test_acc:.4f}"
            )

            history['test_acc'].append(test_acc)
            
            # Save best model and check for early stopping
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"Saved new best model with test accuracy: {test_acc:.4f}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                logger.info(f"No improvement for {no_improvement_count} epochs")

                if no_improvement_count >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # Training finished — save analysis artifacts
        total_time = time.time() - total_start_time
        logger.info(f"Training finished in {total_time:.1f}s, best_acc={best_acc:.4f}")

        try:
            # Save history CSV
            hist_df = pd.DataFrame(history)
            hist_csv = self.tables_dir / f'{self.language}_training_history.csv'
            hist_df.to_csv(hist_csv, index=False)
            logger.info(f"Saved training history to {hist_csv}")

            # Plot training curves
            self._plot_training_curves(history)

            # Collect embeddings on test set and save
            self._collect_and_save_embeddings(test_loader)

            # Save a simple architecture SVG
            self._save_architecture_svg()
        except Exception as e:
            logger.warning(f"Failed to save analysis artifacts: {e}")

        return best_model_path
    
    def _evaluate(self, data_loader):
        """Evaluate model on a dataset."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = batch.y.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        return accuracy_score(all_labels, all_preds)

    # ---------------------- Analysis helpers ----------------------
    def _plot_training_curves(self, history: dict):
        """Plot and save training loss and accuracy curves."""
        try:
            df = pd.DataFrame(history)
            plt.figure(figsize=(8, 4))
            plt.plot(df['epoch'], df['avg_loss'], label='Avg Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{self.language} - Training Loss')
            plt.grid(True)
            save_path = self.plots_dir / f'{self.language}_loss_curve.png'
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved loss curve to {save_path}")

            plt.figure(figsize=(8, 4))
            plt.plot(df['epoch'], df['test_acc'], label='Test Acc', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'{self.language} - Test Accuracy')
            plt.grid(True)
            save_path = self.plots_dir / f'{self.language}_acc_curve.png'
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved accuracy curve to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to plot training curves: {e}")

    def _collect_and_save_embeddings(self, data_loader):
        """Collect BERT embeddings (CLS) and final dual-space embeddings for analysis and t-SNE."""
        try:
            self.bert_model.eval()
            self.model.eval()
            bert_embs = []
            dual_syn = []
            dual_ant = []
            labels = []

            with torch.no_grad():
                for batch in tqdm(data_loader, desc='Collecting embeddings'):
                    batch = batch.to(self.device)
                    # Obtain node embeddings from bert_model applied earlier in dataset; but dataset returns graph x
                    # We will use model.x_syn and model.x_ant after forward pass
                    logits = self.model(batch)
                    # model.x_syn and x_ant are stored per forward call
                    x_syn = self.model.x_syn.cpu().numpy()
                    x_ant = self.model.x_ant.cpu().numpy()
                    # batch.x contains BERT CLS embeddings per node
                    bert_nodes = batch.x.cpu().numpy()

                    # Each graph has two nodes; flatten
                    for i in range(bert_nodes.shape[0]):
                        bert_embs.append(bert_nodes[i])
                    for i in range(x_syn.shape[0]):
                        dual_syn.append(x_syn[i])
                    for i in range(x_ant.shape[0]):
                        dual_ant.append(x_ant[i])

                    labels.extend(batch.y.cpu().numpy().tolist())

            # Save embeddings as npy
            bert_np = np.asarray(bert_embs)
            syn_np = np.asarray(dual_syn)
            ant_np = np.asarray(dual_ant)
            np.save(self.emb_dir / f'{self.language}_bert_embeddings.npy', bert_np)
            np.save(self.emb_dir / f'{self.language}_dual_syn_embeddings.npy', syn_np)
            np.save(self.emb_dir / f'{self.language}_dual_ant_embeddings.npy', ant_np)
            np.save(self.emb_dir / f'{self.language}_labels.npy', np.asarray(labels))
            logger.info(f"Saved embeddings to {self.emb_dir}")

            # t-SNE on BERT vs dual syn (sample if large)
            try:
                sample_n = min(2000, bert_np.shape[0])
                idx = np.linspace(0, bert_np.shape[0]-1, sample_n, dtype=int)
                tsne = TSNE(n_components=2, random_state=42)
                bert_2d = tsne.fit_transform(bert_np[idx])
                syn_2d = tsne.fit_transform(syn_np[idx])

                plt.figure(figsize=(8, 4))
                plt.scatter(bert_2d[:,0], bert_2d[:,1], c=np.asarray(labels)[idx], cmap='coolwarm', s=5)
                plt.title(f'{self.language} - BERT CLS t-SNE')
                plt.savefig(self.plots_dir / f'{self.language}_bert_tsne.png')
                plt.close()

                plt.figure(figsize=(8, 4))
                plt.scatter(syn_2d[:,0], syn_2d[:,1], c=np.asarray(labels)[idx], cmap='coolwarm', s=5)
                plt.title(f'{self.language} - Dual Syn t-SNE')
                plt.savefig(self.plots_dir / f'{self.language}_dual_syn_tsne.png')
                plt.close()
                logger.info(f"Saved t-SNE plots to {self.plots_dir}")
                # UMAP for dual syn
                try:
                    from umap import UMAP
                    um = UMAP(n_components=2, random_state=42)
                    syn_u = um.fit_transform(syn_np[idx])
                    plt.figure(figsize=(8,4))
                    plt.scatter(syn_u[:,0], syn_u[:,1], c=np.asarray(labels)[idx], cmap='coolwarm', s=5)
                    plt.title(f'{self.language} - Dual Syn UMAP')
                    plt.savefig(self.plots_dir / f'{self.language}_dual_syn_umap.png')
                    plt.close()
                except Exception as e:
                    logger.warning(f"Failed to compute Dual syn UMAP: {e}")
            except Exception as e:
                logger.warning(f"Failed to compute t-SNE: {e}")

        except Exception as e:
            logger.warning(f"Failed to collect embeddings: {e}")

    def _save_architecture_svg(self):
        """Save a minimal architecture SVG describing the model components."""
        try:
            svg_content = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='800' height='200'>
  <rect x='10' y='10' width='180' height='60' fill='#e3f2fd' stroke='#1565c0' />
  <text x='20' y='40' font-size='12' fill='#1565c0'>BERT CLS Embeddings</text>
  <rect x='220' y='10' width='180' height='60' fill='#fff3e0' stroke='#ef6c00' />
  <text x='230' y='40' font-size='12' fill='#ef6c00'>Dual Projections (syn/ant)</text>
  <rect x='430' y='10' width='180' height='60' fill='#e8f5e9' stroke='#2e7d32' />
  <text x='440' y='40' font-size='12' fill='#2e7d32'>Graph Transformer</text>
  <rect x='640' y='10' width='140' height='60' fill='#f3e5f5' stroke='#6a1b9a' />
  <text x='650' y='40' font-size='12' fill='#6a1b9a'>Classifier</text>
  <line x1='190' y1='40' x2='220' y2='40' stroke='#333' />
  <line x1='400' y1='40' x2='430' y2='40' stroke='#333' />
  <line x1='610' y1='40' x2='640' y2='40' stroke='#333' />
</svg>
"""
            svg_path = self.svg_dir / f'{self.language}_architecture.svg'
            with open(svg_path, 'w') as f:
                f.write(svg_content)
            logger.info(f"Saved architecture SVG to {svg_path}")
        except Exception as e:
            logger.warning(f"Failed to save architecture SVG: {e}")
    
    
    def _plot_confusion_matrix(self, cm, title):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=["Not Antonym", "Antonym"], 
            yticklabels=["Not Antonym", "Antonym"]
        )
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        
        save_path = self.output_dir / f'{self.language}_dual_encoder_confusion_matrix.png'
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Train multilingual dual encoder for antonym detection')
    parser.add_argument('--language', type=str, required=True,
                      help='Language to train on (e.g., german, spanish, french)')
    parser.add_argument('--config', type=str, 
                      default='config/language_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--data_dir', type=str,
                      default='datasets',
                      help='Directory containing language datasets')
    parser.add_argument('--output_dir', type=str,
                      default='assets',
                      help='Directory to save model outputs')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if language is supported
    if args.language not in config['languages']:
        logger.error(f"Language '{args.language}' not supported.")
        logger.info(f"Supported languages: {list(config['languages'].keys())}")
        return
    
    # Initialize trainer
    trainer = MultilingualDualEncoderTrainer(config, args.language, args.output_dir)
    
    # Load data
    train_data, test_data = trainer.load_data(args.data_dir)
    
    # Create dataloaders
    train_loader, test_loader = trainer.create_dataloaders(train_data, test_data)
    
    # Train model
    best_model_path = trainer.train(train_loader, test_loader)
    
    # Test model
    results = trainer.test(test_loader, best_model_path)
    
    logger.info("Dual encoder training completed successfully!")

if __name__ == "__main__":
    main()
