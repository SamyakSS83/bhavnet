import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the datasets
def load_data(file_path):
    """Load data from a file into a DataFrame."""
    word1_list, word2_list, labels = [], [], []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                word1, word2, label = parts[0], parts[1], int(parts[2])
                word1_list.append(word1)
                word2_list.append(word2)
                labels.append(label)
    
    return pd.DataFrame({
        'word1': word1_list,
        'word2': word2_list,
        'label': labels
    })

# Create a custom dataset
class WordPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word1 = str(self.data.iloc[idx]['word1'])  # Convert to string to handle any NaN values
        word2 = str(self.data.iloc[idx]['word2'])
        label = int(self.data.iloc[idx]['label'])

        # Skip any problematic entries (non-string values)
        if word1 == 'nan' or word2 == 'nan':
            # Provide a fallback for NaN values
            word1 = "unknown" if word1 == 'nan' else word1
            word2 = "unknown" if word2 == 'nan' else word2

        encoding = self.tokenizer.encode_plus(
            word1,
            word2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def main():
    # Path to data directory
    dataset_dir = "../dataset"
    word_types = ["adjective-pairs", "noun-pairs", "verb-pairs"]
    
    # Load and combine datasets
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for word_type in word_types:
        train_file = os.path.join(dataset_dir, f"{word_type}.train")
        val_file = os.path.join(dataset_dir, f"{word_type}.val")
        test_file = os.path.join(dataset_dir, f"{word_type}.test")
        
        if os.path.exists(train_file):
            train_data = pd.concat([train_data, load_data(train_file)])
        if os.path.exists(val_file):
            val_data = pd.concat([val_data, load_data(val_file)])
        if os.path.exists(test_file):
            test_data = pd.concat([test_data, load_data(test_file)])

    print(f"Training data: {len(train_data)} samples")
    print(f"Validation data: {len(val_data)} samples")
    print(f"Test data: {len(test_data)} samples")

    # Check if we have any data
    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        print("Error: No data found. Please check if the dataset files exist in the correct directory.")
        print(f"Looking for files in: {os.path.abspath(dataset_dir)}")
        for word_type in word_types:
            train_file = os.path.join(dataset_dir, f"{word_type}.train")
            val_file = os.path.join(dataset_dir, f"{word_type}.val")
            test_file = os.path.join(dataset_dir, f"{word_type}.test")
            print(f"  {train_file}: {'EXISTS' if os.path.exists(train_file) else 'NOT FOUND'}")
            print(f"  {val_file}: {'EXISTS' if os.path.exists(val_file) else 'NOT FOUND'}")
            print(f"  {test_file}: {'EXISTS' if os.path.exists(test_file) else 'NOT FOUND'}")
        return

    # Check for NaN values
    print("Checking for NaN values in datasets...")
    print(f"Train NaN in word1: {train_data['word1'].isna().sum()}")
    print(f"Train NaN in word2: {train_data['word2'].isna().sum()}")
    print(f"Val NaN in word1: {val_data['word1'].isna().sum()}")
    print(f"Val NaN in word2: {val_data['word2'].isna().sum()}")
    print(f"Test NaN in word1: {test_data['word1'].isna().sum()}")
    print(f"Test NaN in word2: {test_data['word2'].isna().sum()}")

    # Drop rows with NaN values if any
    train_data = train_data.dropna()
    val_data = val_data.dropna()
    test_data = test_data.dropna()

    # Initialize tokenizer and model
    print("Initializing BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = WordPairDataset(train_data, tokenizer)
    val_dataset = WordPairDataset(val_data, tokenizer)
    test_dataset = WordPairDataset(test_data, tokenizer)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    num_epochs = 12
    best_val_accuracy = 0
    os.makedirs("../assets", exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            train_pbar.set_postfix({'loss': train_loss / train_steps})

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        val_preds = []
        val_true = []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                val_loss += loss.item()
                val_steps += 1
                
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': val_loss / val_steps})

        val_accuracy = accuracy_score(val_true, val_preds)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/train_steps:.4f}, Val Loss: {val_loss/val_steps:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "../assets/best_bert_model.pt")
            print(f"Saved new best model with validation accuracy: {val_accuracy:.4f}")

    # Load the best model for testing
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load("../assets/best_bert_model.pt"))

    # Test the model
    model.eval()
    test_preds = []
    test_true = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        test_pbar = tqdm(test_loader)
        for batch in test_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_true, test_preds)
    report = classification_report(test_true, test_preds, target_names=['Not Antonym', 'Antonym'])
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Also test on each word type separately
    print("\nEvaluating performance on each word type...")
    for word_type in word_types:
        test_file = os.path.join(dataset_dir, f"{word_type}.test")
        if os.path.exists(test_file):
            type_data = load_data(test_file).dropna()
            type_dataset = WordPairDataset(type_data, tokenizer)
            type_loader = DataLoader(type_dataset, batch_size=batch_size)
            
            type_preds = []
            type_true = []
            
            with torch.no_grad():
                for batch in tqdm(type_loader, desc=f"Testing {word_type}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1)
                    type_preds.extend(preds.cpu().numpy())
                    type_true.extend(labels.cpu().numpy())
            
            type_accuracy = accuracy_score(type_true, type_preds)
            type_report = classification_report(type_true, type_preds, target_names=['Not Antonym', 'Antonym'])
            
            print(f"\n--- {word_type} Results ---")
            print(f"Accuracy: {type_accuracy:.4f}")
            print("Classification Report:")
            print(type_report)

if __name__ == "__main__":
    main()