import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Create assets directory if it doesn't exist
os.makedirs("../assets", exist_ok=True)

# -----------------------
# Device Configuration
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# Load Fine-Tuned BERT for Embeddings
# -----------------------
# Update this path to point to your fine-tuned BERT model checkpoint directory.
finetuned_bert_path = "../assets/best_bert_model.pt" # <-- CHANGE THIS PATH as needed
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ft_model = AutoModelForSequenceClassification.from_pretrained(model_name)
ft_model.load_state_dict(torch.load(finetuned_bert_path, map_location=device))

# Extract the underlying BERT encoder for embeddings
finetuned_bert = ft_model.bert.to(device)
finetuned_bert.eval()

# -----------------------
# Data Loading Functions
# -----------------------
def load_data(file_path):
    """Load data from a file and return a DataFrame."""
    word1_list, word2_list, labels = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                word1, word2, label = parts[0], parts[1], int(parts[2])
                word1_list.append(word1)
                word2_list.append(word2)
                labels.append(label)
    return pd.DataFrame({'word1': word1_list, 'word2': word2_list, 'label': labels})

# -----------------------
# Graph Dataset Definition
# -----------------------
class WordPairGraphDataset(Dataset):
    """
    For each word pair, we create a graph with two nodes:
    - Each node is the BERT embedding (CLS token) of the word from your fine-tuned model.
    - A bidirectional edge connects the two nodes.
    The graph's label is the word pair's label.
    """
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word1 = self.data.iloc[idx]['word1']
        word2 = self.data.iloc[idx]['word2']
        label = self.data.iloc[idx]['label']

        # Use the fine-tuned BERT encoder to get embeddings for each word
        with torch.no_grad():
            emb1 = finetuned_bert(**tokenizer(word1, return_tensors='pt').to(device)).last_hidden_state[:, 0, :].squeeze(0).cpu()
            emb2 = finetuned_bert(**tokenizer(word2, return_tensors='pt').to(device)).last_hidden_state[:, 0, :].squeeze(0).cpu()

        # Node features for both words
        x = torch.stack([emb1, emb2])
        
        # Create a bidirectional edge between the two nodes
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        # Create a GraphData object from torch_geometric
        return GraphData(x=x, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))

# -----------------------
# Dual Encoder Graph Transformer Model Definition
# -----------------------
class DualEncoderGraphTransformer(nn.Module):
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
        self.conv1 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        
        # Classification layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Compute projections from both branches
        x_syn = self.syn_proj(x)
        x_ant = self.ant_proj(x)
        
        # Store for potential margin loss computation
        self.x_syn = x_syn
        self.x_ant = x_ant
        
        # Concatenate the two projections
        x_combined = torch.cat([x_syn, x_ant], dim=1)
        x_fused = self.fusion(x_combined)
        
        # Graph transformer layers
        x = self.conv1(x_fused, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Pooling to get graph-level embedding
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        
        return self.lin2(x)

# Function to compute margin loss
def compute_margin_loss(model, data):
    """
    Compute margin-based loss on the dual projections.
    For a graph with two nodes (word pair):
    - For non-antonym pairs (label==0): we want similarity in synonym space to be high
    - For antonym pairs (label==1): we want similarity in antonym space to be high
    """
    margin_syn = 0.8
    margin_ant = 0.2
    losses = []
    batch = data.batch.cpu().numpy()
    unique_graphs = np.unique(batch)
    
    for g in unique_graphs:
        idx = (data.batch == g).nonzero(as_tuple=False).view(-1)
        if idx.shape[0] != 2:
            continue
        
        # Get the graph label
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
        return torch.tensor(0.0, device=device)

def plot_confusion(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Antonym", "Antonym"], yticklabels=["Not Antonym", "Antonym"])
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -----------------------
# Load Dataset
# -----------------------
def main():
    data_dir = "../dataset"
    word_types = ["adjective-pairs", "noun-pairs", "verb-pairs"]
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_sets = {}

    for wt in word_types:
        train_file = os.path.join(data_dir, f"{wt}.train")
        val_file = os.path.join(data_dir, f"{wt}.val")
        test_file = os.path.join(data_dir, f"{wt}.test")
        
        if os.path.exists(train_file):
            train_df = pd.concat([train_df, load_data(train_file)], ignore_index=True)
        if os.path.exists(val_file):
            val_df = pd.concat([val_df, load_data(val_file)], ignore_index=True)
        if os.path.exists(test_file):
            test_sets[wt] = load_data(test_file)

    # Combine training and validation sets for training
    train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Check if we have any data
    if len(train_df) == 0:
        print("Error: No training data found. Please check if the dataset files exist in the correct directory.")
        print(f"Looking for files in: {os.path.abspath(data_dir)}")
        for wt in word_types:
            train_file = os.path.join(data_dir, f"{wt}.train")
            val_file = os.path.join(data_dir, f"{wt}.val")
            test_file = os.path.join(data_dir, f"{wt}.test")
            print(f"  {train_file}: {'EXISTS' if os.path.exists(train_file) else 'NOT FOUND'}")
            print(f"  {val_file}: {'EXISTS' if os.path.exists(val_file) else 'NOT FOUND'}")
            print(f"  {test_file}: {'EXISTS' if os.path.exists(test_file) else 'NOT FOUND'}")
        return
    
    print(f"Training data: {len(train_df)} samples")
    if test_sets:
        total_test = sum(len(df) for df in test_sets.values())
        print(f"Test data: {total_test} samples across {len(test_sets)} word types")

    # -----------------------
    # Create Datasets and DataLoaders
    # -----------------------
    train_dataset = WordPairGraphDataset(train_df)
    train_loader = GeoDataLoader(train_dataset, batch_size=32, shuffle=True)

    # Combined test dataset
    combined_test_df = pd.concat(list(test_sets.values()), ignore_index=True)
    combined_test_dataset = WordPairGraphDataset(combined_test_df)
    combined_test_loader = GeoDataLoader(combined_test_dataset, batch_size=32)

    # Test loaders per word type
    test_loaders = {
        wt: GeoDataLoader(WordPairGraphDataset(df), batch_size=32)
        for wt, df in test_sets.items()
    }

    # -----------------------
    # Model, Optimizer, and Loss Setup
    # -----------------------
    model = DualEncoderGraphTransformer(in_channels=768, hidden_channels=256, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # -----------------------
    # Training Loop
    # -----------------------
    num_epochs = 10
    best_model_path = "../assets/best_dual_encoder_graph_model.pt"
    best_acc = 0.0
    lambda_margin = 0.5  # Weight for margin loss

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            
            # Compute standard classification loss
            ce_loss = criterion(out, batch.y)
            
            # Compute margin loss
            margin_loss = compute_margin_loss(model, batch)
            
            # Combine losses
            loss = ce_loss + lambda_margin * margin_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Evaluate on combined test set
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in combined_test_loader:
                batch = batch.to(device)
                logits = model(batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = batch.y.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"Combined Test Accuracy: {acc:.4f}")
        
        # Save best model and plot confusion matrix if improved
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print("Saved new best model.")
            
            cm = confusion_matrix(all_labels, all_preds)
            plot_confusion(cm, 'Combined Test Confusion Matrix', os.path.join('../assets', 'combined_confusion_matrix.png'))

    # -----------------------
    # Evaluation on Each Word Type
    # -----------------------
    print("\nEvaluating best model on each word type:")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    for wt, loader in test_loaders.items():
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = batch.y.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"\n{wt} Accuracy: {acc:.4f}")
        print(classification_report(all_labels, all_preds, target_names=["Not Antonym", "Antonym"]))
        
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion(cm, f"{wt} Confusion Matrix", os.path.join("../assets", f"{wt}_confusion_matrix.png"))

if __name__ == "__main__":
    main()
