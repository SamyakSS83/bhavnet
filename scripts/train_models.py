#!/usr/bin/env python3
"""
Multilingual Antonym Detection Training System
Trains BERT and Dual Encoder models on downloaded datasets.
"""

import os
import sys
import yaml
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import model classes
sys.path.append(str(Path(__file__).parent.parent))
from models.multilingual_bert import WordPairDataset, MultilingualBertTrainer
from models.multilingual_dual_encoder import DualEncoderGraphTransformer, WordPairGraphDataset

class MultilingualTrainingSystem:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_directories()
        
    def load_config(self):
        """Load training configuration."""
        default_config = {
            'datasets': {
                'base_dir': '/home/samyak/scratch/temp/multilingual_antonym_detection/datasets',
                'languages': ['german', 'french', 'spanish', 'italian', 'portuguese', 'dutch', 'russian', 'english']
            },
            'models': {
                'bert_dir': '/home/samyak/scratch/temp/multilingual_antonym_detection/models/bert',
                'output_dir': '/home/samyak/scratch/temp/multilingual_antonym_detection/models/trained',
                'model_types': ['bert', 'dual_encoder']
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 2e-5,
                'epochs': 3,
                'max_length': 128,
                'test_size': 0.2,
                'val_size': 0.1,
                # Dual encoder specific parameters
                'dual_encoder_epochs': 10,
                'dual_encoder_lr': 1e-4,
                'hidden_dim': 256,
                'dropout': 0.2,
                'margin_syn': 0.8,
                'margin_ant': 0.2,
                'margin_weight': 0.5,
                # Model architecture
                'heads': 2,
                'graph_layers': 2
            },
            'hardware': {
                'use_gpu': True,
                'mixed_precision': True
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge configs
            self.config = self._merge_configs(default_config, user_config)
        else:
            self.config = default_config
            # Save default config
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Created default config at {self.config_path}")
    
    def _merge_configs(self, default: dict, user: dict) -> dict:
        """Recursively merge configuration dictionaries."""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                default[key] = self._merge_configs(default[key], value)
            else:
                default[key] = value
        return default
    
    def setup_directories(self):
        """Create necessary directories."""
        self.datasets_dir = Path(self.config['datasets']['base_dir'])
        self.bert_models_dir = Path(self.config['models']['bert_dir'])
        self.output_dir = Path(self.config['models']['output_dir'])
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each model type
        for model_type in self.config['models']['model_types']:
            (self.output_dir / model_type).mkdir(exist_ok=True)
    
    def check_prerequisites(self) -> bool:
        """Check if datasets and BERT models are available."""
        logger.info("Checking prerequisites...")
        
        # Check datasets
        available_languages = []
        for language in self.config['datasets']['languages']:
            lang_dir = self.datasets_dir / language
            if lang_dir.exists():
                train_file = lang_dir / 'train.txt'
                val_file = lang_dir / 'val.txt'
                test_file = lang_dir / 'test.txt'
                
                if train_file.exists() and val_file.exists() and test_file.exists():
                    available_languages.append(language)
                    logger.info(f"✓ Dataset found for {language}")
                else:
                    logger.warning(f"✗ Incomplete dataset for {language}")
            else:
                logger.warning(f"✗ No dataset directory for {language}")
        
        if not available_languages:
            logger.error("No complete datasets found! Run dataset_downloader.py first.")
            return False
        
        self.available_languages = available_languages
        logger.info(f"Found datasets for {len(available_languages)} languages: {', '.join(available_languages)}")
        
        # Check BERT models
        bert_available = []
        for language in available_languages:
            bert_dir = self.bert_models_dir / language
            if bert_dir.exists() and (bert_dir / 'model').exists() and (bert_dir / 'tokenizer').exists():
                bert_available.append(language)
                logger.info(f"✓ BERT model found for {language}")
            else:
                logger.warning(f"✗ BERT model missing for {language}")
        
        # Check multilingual model
        multilingual_dir = self.bert_models_dir / 'multilingual'
        if multilingual_dir.exists() and (multilingual_dir / 'model').exists():
            logger.info("✓ Multilingual BERT model found")
            self.has_multilingual = True
        else:
            logger.warning("✗ Multilingual BERT model missing")
            self.has_multilingual = False
        
        if not bert_available and not self.has_multilingual:
            logger.error("No BERT models found! Run bert_downloader.py first.")
            return False
        
        self.bert_available = bert_available
        return True
    
    def load_dataset(self, language: str) -> Dict[str, pd.DataFrame]:
            """Load train/val/test datasets for a language.

            This supports the current layout datasets/<lang>/{train,val,test}.txt
            and also the legacy layout under dataset/ for English (adjective/noun/verb pairs)
            when the datasets/english directory is missing or when forced via CLI.
            """
            # Primary path: datasets/<language>/
            lang_dir = self.datasets_dir / language

            datasets = {}

            # If user forced legacy mode for English, or lang_dir missing, try legacy layout
            if (language.lower() == 'english' and not lang_dir.exists()) or getattr(self, 'force_legacy_dataset', False):
                # Legacy files are in ../dataset/ with splits per POS (adjective/noun/verb)
                legacy_dir = Path(__file__).parent.parent / 'dataset'
                # Try to assemble train/val/test from adjective/noun/verb aggregated files if present
                def try_legacy(split_name: str) -> Optional[pd.DataFrame]:
                    candidates = [
                        legacy_dir / f'adjective-pairs.{split_name}',
                        legacy_dir / f'noun-pairs.{split_name}',
                        legacy_dir / f'verb-pairs.{split_name}',
                    ]
                    frames = []
                    for c in candidates:
                        if c.exists():
                            try:
                                df = pd.read_csv(c, sep='\t', header=None, names=['word1', 'word2', 'label'])
                                frames.append(df)
                            except Exception as e:
                                logger.warning(f"Failed to read legacy file {c}: {e}")
                    if frames:
                        return pd.concat(frames, ignore_index=True)
                    return None

                for split in ['train', 'val', 'test']:
                    df = try_legacy(split)
                    if df is not None:
                        datasets[split] = df
                        logger.info(f"Loaded {len(df)} {split} examples for {language} (legacy)")
                    else:
                        logger.error(f"Legacy dataset split not found for {split} in {legacy_dir}")
                        return None

                return datasets

            # Default (modern) layout
            for split in ['train', 'val', 'test']:
                file_path = lang_dir / f'{split}.txt'

                if file_path.exists():
                    df = pd.read_csv(file_path, sep='\t', header=None, names=['word1', 'word2', 'label'])
                    datasets[split] = df
                    logger.info(f"Loaded {len(df)} {split} examples for {language}")
                else:
                    logger.error(f"Dataset file not found: {file_path}")
                    return None

            return datasets
    
    def get_bert_model_path(self, language: str) -> Optional[Path]:
        """Get the appropriate BERT model path for a language."""
        # Try language-specific model first
        lang_model_dir = self.bert_models_dir / language
        if lang_model_dir.exists() and (lang_model_dir / 'model').exists():
            return lang_model_dir
        
        # Fall back to multilingual model
        if self.has_multilingual:
            multilingual_dir = self.bert_models_dir / 'multilingual'
            logger.info(f"Using multilingual BERT for {language}")
            return multilingual_dir
        
        return None
    
    def train_bert_model(self, language: str) -> bool:
        """Train BERT model for a specific language."""
        logger.info(f"Training BERT model for {language}")
        
        # Load datasets to verify they exist
        datasets = self.load_dataset(language)
        if not datasets:
            return False
        
        # Get BERT model path
        bert_model_path = self.get_bert_model_path(language)
        if not bert_model_path:
            logger.error(f"No BERT model available for {language}")
            return False
        
        trainer = None
        try:
            # Create a compatible config for MultilingualBertTrainer using local model paths
            bert_config = {
                'languages': {
                    language: {
                        'bert_model': str(bert_model_path / 'model')
                    }
                },
                'training': {
                    'num_epochs': int(self.config['training']['epochs']),
                    'learning_rate': float(self.config['training']['learning_rate']),
                    'batch_size': int(self.config['training']['batch_size']),
                    'max_length': int(self.config['training']['max_length'])
                }
            }
            
            # Create trainer
            trainer = MultilingualBertTrainer(
                config=bert_config,
                language=language,
                output_dir=str(self.output_dir / 'bert')
            )
            
            # Load data
            train_data, val_data, test_data = trainer.load_data(str(self.datasets_dir))
            
            # Create dataloaders
            train_loader, val_loader, test_loader = trainer.create_dataloaders(
                train_data, val_data, test_data
            )
            
            # Train model
            best_model_path = trainer.train(train_loader, val_loader)
            
            # Test model with error handling
            try:
                results = trainer.test(test_loader, best_model_path)
                logger.info(f"Test results: {results}")
            except Exception as test_error:
                logger.warning(f"Error during testing: {test_error}")
                logger.info("Training completed but testing failed - model still saved")
                results = {"accuracy": "unknown", "error": str(test_error)}
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"✓ Successfully trained BERT model for {language}")
            return True
            
        except Exception as e:
            logger.error(f"Error training BERT model for {language}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            # Comprehensive cleanup to prevent segfaults
            if trainer is not None:
                try:
                    # Move model to CPU to free GPU memory
                    if hasattr(trainer, 'model') and trainer.model is not None:
                        trainer.model.to('cpu')
                        del trainer.model
                    # Delete trainer object
                    del trainer
                except Exception as cleanup_error:
                    logger.warning(f"Error during trainer cleanup: {cleanup_error}")
            
            # Force comprehensive cleanup
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Clean up matplotlib to prevent segfaults
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
                plt.clf()
                plt.cla()
            except:
                pass
    
    def train_dual_encoder_model(self, language: str) -> bool:
        """Train Dual Encoder model for a specific language."""
        logger.info(f"Training Dual Encoder model for {language}")
        
        # Check for trained BERT model (either directory or .pt file)
        bert_output_dir = self.output_dir / 'bert'
        bert_model_file = bert_output_dir / f'best_{language}_bert_model.pt'
        bert_model_path = self.get_bert_model_path(language)
        
        if not bert_model_file.exists():
            logger.error(f"Trained BERT model not found for {language}. Train BERT first.")
            logger.info(f"Looked for: {bert_model_file}")
            return False
        
        if not bert_model_path:
            logger.error(f"Base BERT model not found for {language}")
            return False
        
        trainer = None
        try:
            # Load datasets
            datasets = self.load_dataset(language)
            if not datasets:
                return False
            
            logger.info(f"Found trained BERT model: {bert_model_file}")
            logger.info(f"Base BERT model path: {bert_model_path}")
            
            # Create dual encoder config
            dual_encoder_config = {
                'languages': {
                    language: {
                        'bert_model': str(bert_model_path / 'model'),
                        'trained_bert_path': str(bert_model_file)
                    }
                },
                'training': {
                    'num_epochs': int(self.config['training'].get('dual_encoder_epochs', 10)),
                    'learning_rate': float(self.config['training'].get('dual_encoder_lr', 1e-4)),
                    'batch_size': int(self.config['training']['batch_size']),
                    'hidden_dim': int(self.config['training'].get('hidden_dim', 256)),
                    'dropout': float(self.config['training'].get('dropout', 0.2)),
                    'margin_syn': float(self.config['training'].get('margin_syn', 0.8)),
                    'margin_ant': float(self.config['training'].get('margin_ant', 0.2)),
                    'margin_weight': float(self.config['training'].get('margin_weight', 0.5))
                }
            }
            
            # Import and create trainer
            from models.multilingual_dual_encoder import MultilingualDualEncoderTrainer
            
            trainer = MultilingualDualEncoderTrainer(
                config=dual_encoder_config,
                language=language,
                output_dir=str(self.output_dir / 'dual_encoder')
            )
            
            # Load data and create graph dataloaders
            train_data, test_data = trainer.load_data(str(self.datasets_dir))
            train_loader, test_loader = trainer.create_dataloaders(train_data, test_data)
            
            # Train the actual graph transformer model
            best_model_path = trainer.train(train_loader, test_loader)
            
            # Test with detailed results
            try:
                results = trainer.test(test_loader, best_model_path)
                logger.info(f"Dual Encoder Test results: {results}")
            except Exception as test_error:
                logger.warning(f"Error during dual encoder testing: {test_error}")
                logger.info("Dual Encoder training completed but testing failed - model still saved")
                results = {"accuracy": "unknown", "error": str(test_error)}
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"✓ Successfully trained Dual Encoder model for {language}")
            return True
            
        except Exception as e:
            logger.error(f"Error training Dual Encoder model for {language}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            # Comprehensive cleanup to prevent segfaults
            if trainer is not None:
                try:
                    # Move models to CPU to free GPU memory
                    if hasattr(trainer, 'model') and trainer.model is not None:
                        trainer.model.to('cpu')
                        del trainer.model
                    if hasattr(trainer, 'bert_model') and trainer.bert_model is not None:
                        trainer.bert_model.to('cpu')
                        del trainer.bert_model
                    # Delete trainer object
                    del trainer
                except Exception as cleanup_error:
                    logger.warning(f"Error during dual encoder trainer cleanup: {cleanup_error}")
            
            # Force comprehensive cleanup
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Clean up matplotlib to prevent segfaults
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
                plt.clf()
                plt.cla()
            except:
                pass
    
    def train_all_languages(self, model_types: List[str] = None) -> Dict[str, Dict[str, bool]]:
        """Train models for all available languages."""
        if model_types is None:
            model_types = self.config['models']['model_types']
        
        results = {}
        
        for language in self.available_languages:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Training models for {language.upper()}")
            logger.info(f"{'='*60}")
            
            results[language] = {}
            
            for model_type in model_types:
                if model_type == 'bert':
                    success = self.train_bert_model(language)
                elif model_type == 'dual_encoder':
                    success = self.train_dual_encoder_model(language)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    success = False
                
                results[language][model_type] = success
        
        return results
    
    def print_training_summary(self, results: Dict[str, Dict[str, bool]]):
        """Print training summary."""
        logger.info(f"\\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        
        for language, model_results in results.items():
            logger.info(f"\\n{language.upper()}:")
            for model_type, success in model_results.items():
                status = "✓ SUCCESS" if success else "✗ FAILED"
                logger.info(f"  {model_type}: {status}")
        
        # Count successes
        total_models = sum(len(model_results) for model_results in results.values())
        successful_models = sum(
            sum(1 for success in model_results.values() if success)
            for model_results in results.values()
        )
        
        logger.info(f"\\nOverall: {successful_models}/{total_models} models trained successfully")
        logger.info(f"Trained models saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train multilingual antonym detection models')
    parser.add_argument('--config', default='/home/samyak/scratch/temp/multilingual_antonym_detection/config/training_config.yaml', help='Training configuration file')
    parser.add_argument('--datasets-dir', help='Override datasets base directory')
    parser.add_argument('--use-legacy-dataset', action='store_true', help='Force using the legacy dataset layout under dataset/ (useful for English)')
    parser.add_argument('--language', help='Train specific language only')
    parser.add_argument('--model-type', choices=['bert', 'dual_encoder'], help='Train specific model type only')
    parser.add_argument('--check-only', action='store_true', help='Only check prerequisites')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs for BERT training')
    parser.add_argument('--dual-encoder-epochs', type=int, help='Number of epochs for dual encoder training')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, help='Learning rate for BERT training')
    parser.add_argument('--dual-encoder-lr', type=float, help='Learning rate for dual encoder training')
    parser.add_argument('--hidden-dim', type=int, help='Hidden dimension for dual encoder')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--margin-weight', type=float, help='Weight for margin loss in dual encoder')
    
    args = parser.parse_args()
    
    # Initialize training system
    training_system = MultilingualTrainingSystem(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        training_system.config['training']['epochs'] = args.epochs
    if args.dual_encoder_epochs:
        training_system.config['training']['dual_encoder_epochs'] = args.dual_encoder_epochs
    if args.batch_size:
        training_system.config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        training_system.config['training']['learning_rate'] = args.learning_rate
    if args.dual_encoder_lr:
        training_system.config['training']['dual_encoder_lr'] = args.dual_encoder_lr
    if args.hidden_dim:
        training_system.config['training']['hidden_dim'] = args.hidden_dim
    if args.dropout:
        training_system.config['training']['dropout'] = args.dropout
    if args.margin_weight:
        training_system.config['training']['margin_weight'] = args.margin_weight
    
    # Check prerequisites
    if not training_system.check_prerequisites():
        logger.error("Prerequisites not met. Please download datasets and BERT models first.")
        sys.exit(1)
    
    if args.check_only:
        logger.info("Prerequisites check completed successfully!")
        return
    
    # Determine what to train
    languages_to_train = [args.language] if args.language else training_system.available_languages
    model_types_to_train = [args.model_type] if args.model_type else None
    
    # Filter languages based on availability
    if args.language and args.language not in training_system.available_languages:
        logger.error(f"Language {args.language} not available. Available: {', '.join(training_system.available_languages)}")
        sys.exit(1)
    
    # Start training
    logger.info(f"Starting training for languages: {', '.join(languages_to_train)}")
    if model_types_to_train:
        logger.info(f"Model types: {', '.join(model_types_to_train)}")
    
    results = {}
    for language in languages_to_train:
        if args.language:
            # Train single language
            results[language] = {}
            for model_type in (model_types_to_train or ['bert', 'dual_encoder']):
                if model_type == 'bert':
                    success = training_system.train_bert_model(language)
                elif model_type == 'dual_encoder':
                    success = training_system.train_dual_encoder_model(language)
                results[language][model_type] = success
        else:
            # Train all languages
            results = training_system.train_all_languages(model_types_to_train)
            break
    
    # Print summary
    training_system.print_training_summary(results)

if __name__ == "__main__":
    main()
