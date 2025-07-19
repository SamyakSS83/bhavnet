import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import argparse
import yaml
from tqdm import tqdm
# Set matplotlib backend before importing pyplot to avoid segfaults
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WordPairDataset(Dataset):
    """Dataset for word pair classification."""
    
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        word1 = str(row['word1'])
        word2 = str(row['word2'])
        label = int(row['label'])

        # Handle NaN values
        if word1 == 'nan' or word2 == 'nan':
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

class MultilingualBertTrainer:
    """Trainer class for multilingual BERT models."""
    
    def __init__(self, config, language, output_dir):
        self.config = config
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Get language-specific config
        self.lang_config = config['languages'][language]
        self.training_config = config['training']
        
        # Initialize model and tokenizer
        self.model_name = self.lang_config['bert_model']
        logger.info(f"Using model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2
        )
        self.model.to(self.device)
        
    def load_data(self, data_dir):
        """Load training, validation, and test data."""
        data_path = Path(data_dir) / self.language
        
        # Load data files
        train_file = data_path / 'train.txt'
        val_file = data_path / 'val.txt'
        test_file = data_path / 'test.txt'
        
        train_data = self._load_file(train_file) if train_file.exists() else pd.DataFrame()
        val_data = self._load_file(val_file) if val_file.exists() else pd.DataFrame()
        test_data = self._load_file(test_file) if test_file.exists() else pd.DataFrame()
        
        logger.info(f"Loaded data for {self.language}:")
        logger.info(f"  Train: {len(train_data)} samples")
        logger.info(f"  Val: {len(val_data)} samples")
        logger.info(f"  Test: {len(test_data)} samples")
        
        if len(train_data) == 0:
            raise ValueError(f"No training data found for {self.language}")
        
        return train_data, val_data, test_data
    
    def _load_file(self, file_path):
        """Load a single data file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    data.append({
                        'word1': parts[0],
                        'word2': parts[1],
                        'label': int(parts[2])
                    })
        return pd.DataFrame(data)
    
    def create_dataloaders(self, train_data, val_data, test_data):
        """Create DataLoaders for training, validation, and testing."""
        batch_size = self.training_config['batch_size']
        
        train_dataset = WordPairDataset(train_data, self.tokenizer)
        val_dataset = WordPairDataset(val_data, self.tokenizer)
        test_dataset = WordPairDataset(test_data, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader, val_loader):
        """Train the model."""
        num_epochs = self.training_config['num_epochs']
        learning_rate = self.training_config['learning_rate']
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        best_val_accuracy = 0
        best_model_path = self.output_dir / f'best_{self.language}_bert_model.pt'
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch in train_pbar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                train_pbar.set_postfix({'loss': train_loss / train_steps})

            # Validation phase
            val_accuracy = self._evaluate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss: {train_loss/train_steps:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}"
            )
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"Saved new best model with validation accuracy: {val_accuracy:.4f}")
        
        return best_model_path
    
    def _evaluate(self, data_loader):
        """Evaluate the model on a dataset."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return accuracy_score(all_labels, all_preds)
    
    def test(self, test_loader, model_path):
        """Test the model and generate detailed results."""
        # Load best model
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        logger.info("Evaluating on test set...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Handle single-class prediction issue
        unique_labels = set(all_labels)
        unique_preds = set(all_preds)
        
        if len(unique_labels) == 1 or len(unique_preds) == 1:
            logger.warning(f"Single-class prediction detected. Labels: {unique_labels}, Predictions: {unique_preds}")
            report = f"Accuracy: {accuracy:.4f}\nNote: Only one class present in predictions or labels"
        else:
            report = classification_report(
                all_labels, 
                all_preds, 
                target_names=['Not Antonym', 'Antonym']
            )
        
        logger.info(f"Test Accuracy for {self.language}: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        # Save results
        results = {
            'language': self.language,
            'accuracy': accuracy,
            'classification_report': report
        }
        
        # Plot confusion matrix with error handling
        try:
            cm = confusion_matrix(all_labels, all_preds)
            # Skip plotting to avoid segfaults - just log the confusion matrix
            logger.info(f"Confusion Matrix for {self.language}:")
            logger.info(f"\n{cm}")
            # Optionally save as text file instead
            cm_file = self.output_dir / f'{self.language}_confusion_matrix.txt'
            with open(cm_file, 'w') as f:
                f.write(f"Confusion Matrix for {self.language}:\n")
                f.write(f"{cm}\n")
                f.write(f"Classes: ['Not Antonym', 'Antonym']\n")
            logger.info(f"Saved confusion matrix data to {cm_file}")
        except Exception as plot_error:
            logger.warning(f"Failed to create confusion matrix: {plot_error}")
        
        return results
    
    def _plot_confusion_matrix(self, cm, title):
        """Plot and save confusion matrix."""
        try:
            # Set backend explicitly for this operation
            plt.ioff()  # Turn off interactive mode
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues", 
                xticklabels=["Not Antonym", "Antonym"], 
                yticklabels=["Not Antonym", "Antonym"],
                ax=ax
            )
            ax.set_title(title)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            
            save_path = self.output_dir / f'{self.language}_bert_confusion_matrix.png'
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create confusion matrix plot: {e}")
        finally:
            # Comprehensive cleanup to prevent segfaults
            try:
                plt.close('all')
                plt.clf()
                plt.cla()
                # Force garbage collection
                import gc
                gc.collect()
            except Exception as cleanup_error:
                logger.warning(f"Error during plot cleanup: {cleanup_error}")

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Train multilingual BERT for antonym detection')
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
    trainer = MultilingualBertTrainer(config, args.language, args.output_dir)
    
    # Load data
    train_data, val_data, test_data = trainer.load_data(args.data_dir)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = trainer.create_dataloaders(
        train_data, val_data, test_data
    )
    
    # Train model
    best_model_path = trainer.train(train_loader, val_loader)
    
    # Test model
    results = trainer.test(test_loader, best_model_path)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
