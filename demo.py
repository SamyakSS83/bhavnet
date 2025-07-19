#!/usr/bin/env python3
"""
Demo script for multilingual antonym detection system.
This script demonstrates how to use the trained models for inference.
"""

import torch
import yaml
from pathlib import Path
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.multilingual_dual_encoder import DualEncoderGraphTransformer
from torch_geometric.data import Data as GraphData
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AntonymDetector:
    """Simple interface for antonym detection."""
    
    def __init__(self, language, config_path, model_dir):
        self.language = language
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if language not in self.config['languages']:
            raise ValueError(f"Language {language} not supported")
        
        self.lang_config = self.config['languages'][language]
        self.model_name = self.lang_config['bert_model']
        
        # Load BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        bert_model_path = Path(model_dir) / f'best_{language}_bert_model.pt'
        if bert_model_path.exists():
            print(f"Loading fine-tuned BERT from {bert_model_path}")
            ft_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            ft_model.load_state_dict(torch.load(bert_model_path, map_location=self.device))
            self.bert_model = ft_model.bert.to(self.device)
        else:
            print("Using pre-trained BERT (not fine-tuned)")
            ft_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.bert_model = ft_model.bert.to(self.device)
        
        self.bert_model.eval()
        
        # Load dual encoder model
        dual_encoder_path = Path(model_dir) / f'best_{language}_dual_encoder_model.pt'
        if dual_encoder_path.exists():
            print(f"Loading dual encoder from {dual_encoder_path}")
            self.dual_encoder = DualEncoderGraphTransformer(
                in_channels=768,
                hidden_channels=self.config['training']['hidden_dim'],
                out_channels=2,
                heads=2,
                dropout=self.config['training']['dropout']
            ).to(self.device)
            self.dual_encoder.load_state_dict(torch.load(dual_encoder_path, map_location=self.device))
            self.dual_encoder.eval()
            self.use_dual_encoder = True
        else:
            print("Dual encoder not found, using BERT only")
            self.bert_classifier = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            if bert_model_path.exists():
                self.bert_classifier.load_state_dict(torch.load(bert_model_path, map_location=self.device))
            self.bert_classifier.to(self.device)
            self.bert_classifier.eval()
            self.use_dual_encoder = False
    
    def predict(self, word1, word2):
        """Predict if two words are antonyms."""
        if self.use_dual_encoder:
            return self._predict_dual_encoder(word1, word2)
        else:
            return self._predict_bert(word1, word2)
    
    def _predict_dual_encoder(self, word1, word2):
        """Predict using dual encoder model."""
        # Get BERT embeddings
        with torch.no_grad():
            inputs1 = self.tokenizer(word1, return_tensors='pt').to(self.device)
            inputs2 = self.tokenizer(word2, return_tensors='pt').to(self.device)
            
            emb1 = self.bert_model(**inputs1).last_hidden_state[:, 0, :].squeeze(0)
            emb2 = self.bert_model(**inputs2).last_hidden_state[:, 0, :].squeeze(0)
            
            # Create graph
            x = torch.stack([emb1, emb2]).unsqueeze(0)  # Add batch dimension
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(self.device)
            batch = torch.zeros(2, dtype=torch.long).to(self.device)
            
            graph_data = GraphData(x=x.squeeze(0), edge_index=edge_index, batch=batch)
            
            # Predict
            logits = self.dual_encoder(graph_data)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities.max().item()
            
            return {
                'is_antonym': bool(prediction),
                'confidence': confidence,
                'antonym_probability': probabilities[0][1].item(),
                'model': 'dual_encoder'
            }
    
    def _predict_bert(self, word1, word2):
        """Predict using BERT classifier."""
        with torch.no_grad():
            inputs = self.tokenizer(
                word1, word2,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            outputs = self.bert_classifier(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            confidence = probabilities.max().item()
            
            return {
                'is_antonym': bool(prediction),
                'confidence': confidence,
                'antonym_probability': probabilities[0][1].item(),
                'model': 'bert'
            }

def run_interactive_demo(detector):
    """Run interactive demo."""
    print(f"\\n=== Antonym Detection Demo ({detector.language.title()}) ===")
    print("Enter word pairs to check if they are antonyms.")
    print("Type 'quit' to exit.\\n")
    
    while True:
        try:
            word1 = input("Enter first word: ").strip()
            if word1.lower() == 'quit':
                break
            
            word2 = input("Enter second word: ").strip()
            if word2.lower() == 'quit':
                break
            
            result = detector.predict(word1, word2)
            
            print(f"\\nResult for '{word1}' and '{word2}':")
            print(f"  Antonym: {'Yes' if result['is_antonym'] else 'No'}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Antonym Probability: {result['antonym_probability']:.3f}")
            print(f"  Model Used: {result['model']}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def run_batch_test(detector, language):
    """Run batch test with predefined examples."""
    # Language-specific test pairs
    test_pairs = {
        'english': [
            ('good', 'bad', True),
            ('hot', 'cold', True),
            ('big', 'small', True),
            ('happy', 'sad', True),
            ('house', 'building', False),
            ('car', 'vehicle', False),
            ('dog', 'cat', False),
        ],
        'german': [
            ('gut', 'schlecht', True),
            ('groß', 'klein', True),
            ('heiß', 'kalt', True),
            ('glücklich', 'traurig', True),
            ('haus', 'gebäude', False),
            ('auto', 'fahrzeug', False),
        ],
        'spanish': [
            ('bueno', 'malo', True),
            ('grande', 'pequeño', True),
            ('caliente', 'frío', True),
            ('feliz', 'triste', True),
            ('casa', 'edificio', False),
            ('coche', 'vehículo', False),
        ],
        'french': [
            ('bon', 'mauvais', True),
            ('grand', 'petit', True),
            ('chaud', 'froid', True),
            ('heureux', 'triste', True),
            ('maison', 'bâtiment', False),
            ('voiture', 'véhicule', False),
        ]
    }
    
    pairs = test_pairs.get(language, test_pairs['english'])
    
    print(f"\\n=== Batch Test for {language.title()} ===")
    print(f"Testing {len(pairs)} word pairs...")
    print()
    
    correct = 0
    total = len(pairs)
    
    for word1, word2, expected in pairs:
        result = detector.predict(word1, word2)
        predicted = result['is_antonym']
        
        status = "✓" if predicted == expected else "✗"
        expected_str = "Antonym" if expected else "Not Antonym"
        predicted_str = "Antonym" if predicted else "Not Antonym"
        
        print(f"{status} {word1:12} - {word2:12} | Expected: {expected_str:11} | Predicted: {predicted_str:11} | Conf: {result['confidence']:.3f}")
        
        if predicted == expected:
            correct += 1
    
    accuracy = correct / total
    print(f"\\nAccuracy: {correct}/{total} ({accuracy:.1%})")

def main():
    parser = argparse.ArgumentParser(description='Demo for multilingual antonym detection')
    parser.add_argument('--language', type=str, required=True,
                      help='Language to use (e.g., german, spanish, french)')
    parser.add_argument('--config', type=str, 
                      default='config/language_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--model_dir', type=str,
                      default='assets',
                      help='Directory containing trained models')
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch'],
                      default='interactive',
                      help='Demo mode: interactive or batch testing')
    parser.add_argument('--word1', type=str, help='First word for single prediction')
    parser.add_argument('--word2', type=str, help='Second word for single prediction')
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = AntonymDetector(args.language, args.config, args.model_dir)
        
        if args.word1 and args.word2:
            # Single prediction
            result = detector.predict(args.word1, args.word2)
            print(f"Prediction for '{args.word1}' and '{args.word2}':")
            print(f"  Antonym: {'Yes' if result['is_antonym'] else 'No'}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Model: {result['model']}")
        elif args.mode == 'interactive':
            # Interactive mode
            run_interactive_demo(detector)
        else:
            # Batch testing mode
            run_batch_test(detector, args.language)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
