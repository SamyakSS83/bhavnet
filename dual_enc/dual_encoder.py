import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, TransformerConv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data(file_path):
    """Load data from a file into lists of word pairs and labels."""
    word1_list, word2_list, labels = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                word1, word2, label = parts[0], parts[1], int(parts[2])
                word1_list.append(word1)
                word2_list.append(word2)
                labels.append(label)
    return word1_list, word2_list, labels

def embed_word_pairs(word1_list, word2_list, model):
    """Embed word pairs using the provided sentence transformer model."""
    print("Embedding word pairs...")
    emb1 = model.encode(word1_list, show_progress_bar=True)
    emb2 = model.encode(word2_list, show_progress_bar=True)
    print(f"Embedding complete. Shape: {emb1.shape}")
    return emb1, emb2

def create_graph_data(word1_emb, word2_emb, label):
    """Create a graph data object for a single word pair (2 nodes)."""
    # Create node features (2 nodes: word1 and word2)
    x = torch.tensor(np.vstack([word1_emb, word2_emb]), dtype=torch.float)
    # Create edges (bidirectional connection between the two words)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # Create label tensor (as float for BCE loss)
    y = torch.tensor([label], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    return data

def create_graph_dataset(word1_embeddings, word2_embeddings, labels):
    """Create a list of graph data objects from word pair embeddings."""
    dataset = []
    for i in range(len(labels)):
        data = create_graph_data(word1_embeddings[i], word2_embeddings[i], labels[i])
        dataset.append(data)
    return dataset

class KernelProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_kernels=16, dropout=0.2):
        super().__init__()
        self.kernel_layer = KernelLayer(input_dim, hidden_dim, num_kernels=num_kernels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.kernel_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
class DualEncoderGraphModel(nn.Module):
    """
    Extended model with two projection branches:
      - syn_proj: projects input to a 'synonym' space.
      - ant_proj: projects input to an 'antonym' space.
    Their outputs are concatenated and fused before graph convolutions.
    Additionally, a margin loss based on inner product (using tanh) is computed.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, heads=4, dropout_rate=0.2):
        super(DualEncoderGraphModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        
        # Synonym projection branch (ENC-1)
        self.syn_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Antonym projection branch (ENC-2)
        self.ant_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Fusion layer: combine both branches (resulting dimension 2*hidden_dim) 
        # and project to the input dimension required for TransformerConv (hidden_dim*heads)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim * heads)
        
        # Attentive Graph Transformer layers
        self.conv1 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1)
        self.convs = nn.ModuleList([
            TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1)
            for _ in range(num_layers - 1)
        ])
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x shape: [N, input_dim]
        
        # Compute projections from both branches
        x_syn = self.syn_proj(x)  # [N, hidden_dim]
        x_ant = self.ant_proj(x)  # [N, hidden_dim]
        
        # Save the branch outputs for margin loss computation
        self.x_syn = x_syn
        self.x_ant = x_ant
        
        # Concatenate the two projections
        x_combined = torch.cat([x_syn, x_ant], dim=1)  # [N, 2*hidden_dim]
        x_fused = self.fusion(x_combined)  # [N, hidden_dim*heads]
        
        # Graph transformer layers with dropout and ReLU
        x_conv = self.conv1(x_fused, edge_index)
        x_conv = self.activation(x_conv)
        x_conv = self.dropout(x_conv)
        for conv in self.convs:
            x_conv = conv(x_conv, edge_index)
            x_conv = self.activation(x_conv)
            x_conv = self.dropout(x_conv)
        
        # Global pooling for graph-level representation
        x_pool = global_mean_pool(x_conv, batch)
        out = self.classifier(x_pool)
        return out.squeeze()

def compute_margin_loss(x_syn, x_ant, data):
    """
    Compute margin-based loss on the dual projections.
    For a graph with two nodes (word pair):
      - For synonym pairs (label==0): we want tanh(inner(x_syn_0, x_syn_1)) to be high.
      - For antonym pairs (label==1): we want tanh(inner(x_ant_0, x_ant_1)) to be low.
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
        
        # Fix: index the correct graph label instead of using idx[0].
        label = data.y[g].item()

        sim_syn = torch.tanh(torch.dot(x_syn[idx[0]], x_syn[idx[1]]))
        sim_ant = torch.tanh(torch.dot(x_ant[idx[0]], x_ant[idx[1]]))
        if label == 0:  # synonym pair
            loss = torch.relu(margin_syn - sim_syn)
        else:  # antonym pair
            loss = torch.relu(sim_ant - margin_ant)
        losses.append(loss)

    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=device)

def evaluate_model(model, dataset, batch_size=32, dataset_name=""):
    """Evaluate graph model and print metrics."""
    model.eval()
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            preds = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"\n--- {dataset_name} Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Antonym', 'Antonym'],
                yticklabels=['Not Antonym', 'Antonym'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'assets/graph_confusion_matrix_{dataset_name.replace(" ", "_")}.png')
    plt.close()
    return {'accuracy': accuracy, 'classification_report': report, 'confusion_matrix': conf_matrix}

def train_model(model, train_dataset, val_dataset=None, epochs=10, batch_size=32, learning_rate=1e-3, lambda_margin=0.5):
    """Train the graph model with combined BCE and margin losses."""
    bce_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dataset:
        val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=batch_size)
    
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_batches = 0
        train_preds = []
        train_labels = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss_bce = bce_criterion(outputs, batch.y)
            # Compute margin loss using the stored dual branch outputs
            loss_margin = compute_margin_loss(model.x_syn, model.x_ant, batch)
            loss = loss_bce + lambda_margin * loss_margin
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_batches += 1
            
            # Collect predictions and labels for accuracy
            preds = (outputs >= 0.5).float().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(batch.y.cpu().numpy())
            
            train_pbar.set_postfix({'loss': total_loss / train_batches})
            
        avg_train_loss = total_loss / train_batches
        train_losses.append(avg_train_loss)
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_accuracies.append(train_accuracy)
        
        if val_dataset:
            model.eval()
            total_val_loss = 0
            val_batches = 0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for batch in val_pbar:
                    batch = batch.to(device)
                    outputs = model(batch)
                    loss_bce = bce_criterion(outputs, batch.y)
                    loss_margin = compute_margin_loss(model.x_syn, model.x_ant, batch)
                    loss = loss_bce + lambda_margin * loss_margin
                    total_val_loss += loss.item()
                    val_batches += 1
                    
                    # Collect predictions and labels for accuracy
                    preds = (outputs >= 0.5).float().cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(batch.y.cpu().numpy())
                    
                    val_pbar.set_postfix({'loss': total_val_loss / val_batches})
                    
            avg_val_loss = total_val_loss / val_batches
            val_losses.append(avg_val_loss)
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_graph_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
    
    # Plot training and validation losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    if val_dataset:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    # Plot training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    if val_dataset:
        plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('assets/graph_training_metrics.png')
    plt.close()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def main():
    dataset_dir = "dataset"
    word_types = ["adjective-pairs", "noun-pairs", "verb-pairs"]
    batch_size = 64
    epochs = 150
    
    print("Loading Nomic embedding model...")
    model_st = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    test_data_by_type = {}
    train_word1, train_word2, train_labels = [], [], []
    val_word1, val_word2, val_labels = [], [], []
    
    for word_type in word_types:
        train_file = os.path.join(dataset_dir, f"{word_type}.train")
        val_file = os.path.join(dataset_dir, f"{word_type}.val")
        test_file = os.path.join(dataset_dir, f"{word_type}.test")
        w1_train, w2_train, y_train = load_data(train_file)
        w1_val, w2_val, y_val = load_data(val_file)
        w1_test, w2_test, y_test = load_data(test_file)
        train_word1.extend(w1_train)
        train_word2.extend(w2_train)
        train_labels.extend(y_train)
        val_word1.extend(w1_val)
        val_word2.extend(w2_val)
        val_labels.extend(y_val)
        test_data_by_type[word_type] = (w1_test, w2_test, y_test)
    
    print(f"Training data: {len(train_labels)} samples")
    print(f"Validation data: {len(val_labels)} samples")
    X_train_word1, X_train_word2 = embed_word_pairs(train_word1, train_word2, model_st)
    X_val_word1, X_val_word2 = embed_word_pairs(val_word1, val_word2, model_st)
    train_dataset = create_graph_dataset(X_train_word1, X_train_word2, train_labels)
    val_dataset = create_graph_dataset(X_val_word1, X_val_word2, val_labels)
    
    input_dim = X_train_word1.shape[1]
    model = DualEncoderGraphModel(input_dim=input_dim).to(device)
    print(f"Model initialized with input dimension: {input_dim}")
    
    print("Training model...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        lambda_margin=0.5  # weight for the margin loss
    )
    
    model.load_state_dict(torch.load("best_graph_model.pt"))
    
    print("\nEvaluating model on test sets:")
    results = {}
    for word_type, (w1_test, w2_test, y_test) in test_data_by_type.items():
        print(f"\nEvaluating on {word_type} test set")
        X_test_word1, X_test_word2 = embed_word_pairs(w1_test, w2_test, model_st)
        test_dataset = create_graph_dataset(X_test_word1, X_test_word2, y_test)
        results[word_type] = evaluate_model(
            model=model,
            dataset=test_dataset,
            batch_size=batch_size,
            dataset_name=f"Test - {word_type}"
        )
    
    all_test_word1, all_test_word2, all_test_labels = [], [], []
    for w1, w2, y in test_data_by_type.values():
        all_test_word1.extend(w1)
        all_test_word2.extend(w2)
        all_test_labels.extend(y)
    X_all_test_word1, X_all_test_word2 = embed_word_pairs(all_test_word1, all_test_word2, model_st)
    all_test_dataset = create_graph_dataset(X_all_test_word1, X_all_test_word2, all_test_labels)
    results["overall"] = evaluate_model(
        model=model,
        dataset=all_test_dataset,
        batch_size=batch_size,
        dataset_name="Test - Overall"
    )
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()
