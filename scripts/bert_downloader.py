#!/usr/bin/env python3
"""
BERT Model Downloader for Multilingual Antonym Detection
Downloads the best language-specific BERT models for each language.
"""

import os
import sys
import logging
from pathlib import Path
import subprocess
from typing import Dict, List
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BERTModelDownloader:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Best BERT models for each language (based on HuggingFace popularity and performance)
        self.language_models = {
            'german': {
                'model_name': 'dbmdz/bert-base-german-cased',
                'description': 'German BERT trained on German Wikipedia and news',
                'size': '420MB'
            },
            'french': {
                'model_name': 'camembert-base',
                'description': 'CamemBERT: French BERT trained on OSCAR corpus',
                'size': '440MB'
            },
            'spanish': {
                'model_name': 'dccuchile/bert-base-spanish-wwm-cased',
                'description': 'Spanish BERT with Whole Word Masking',
                'size': '420MB'
            },
            'italian': {
                'model_name': 'dbmdz/bert-base-italian-cased',
                'description': 'Italian BERT trained on Italian Wikipedia and OPUS',
                'size': '420MB'
            },
            'portuguese': {
                'model_name': 'neuralmind/bert-base-portuguese-cased',
                'description': 'BERTimbau: Portuguese BERT trained on brWaC corpus',
                'size': '420MB'
            },
            'dutch': {
                'model_name': 'GroNLP/bert-base-dutch-cased',
                'description': 'BERTje: Dutch BERT trained on Dutch texts',
                'size': '420MB'
            },
            'russian': {
                'model_name': 'DeepPavlov/rubert-base-cased',
                'description': 'RuBERT: Russian BERT trained on Russian Wikipedia and news',
                'size': '670MB'
            }
        }
        
        # Multilingual fallback model
        self.multilingual_model = {
            'model_name': 'FacebookAI/xlm-roberta-base',
            'description': 'XLM-RoBERTa: Multilingual RoBERTa supporting 100+ languages',
            'size': '560MB'
        }
    
    def check_dependencies(self):
        """Check if required packages are installed."""
        required_packages = ['transformers', 'torch', 'tokenizers']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✓ {package} is installed")
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            logger.info("Installing missing packages...")
            
            for package in missing_packages:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 check=True, capture_output=True)
                    logger.info(f"✓ Installed {package}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {package}: {e}")
                    return False
        
        return True
    
    def download_model(self, language: str) -> bool:
        """Download BERT model for a specific language."""
        if language not in self.language_models:
            logger.error(f"No model configured for language: {language}")
            return False
        
        model_config = self.language_models[language]
        model_name = model_config['model_name']
        
        logger.info(f"Downloading {model_config['description']} ({model_config['size']})")
        logger.info(f"Model: {model_name}")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # Create language-specific directory
            lang_model_dir = self.models_dir / language
            lang_model_dir.mkdir(exist_ok=True)
            
            # Download tokenizer
            logger.info(f"Downloading tokenizer for {language}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(lang_model_dir / 'tokenizer')
            logger.info(f"✓ Tokenizer saved to {lang_model_dir / 'tokenizer'}")
            
            # Download model
            logger.info(f"Downloading model weights for {language}...")
            model = AutoModel.from_pretrained(model_name)
            model.save_pretrained(lang_model_dir / 'model')
            logger.info(f"✓ Model saved to {lang_model_dir / 'model'}")
            
            # Save model info
            info_file = lang_model_dir / 'model_info.txt'
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"Language: {language}\n")
                f.write(f"Model Name: {model_name}\n")
                f.write(f"Description: {model_config['description']}\n")
                f.write(f"Size: {model_config['size']}\n")
                f.write(f"Tokenizer: {lang_model_dir / 'tokenizer'}\n")
                f.write(f"Model: {lang_model_dir / 'model'}\n")
                f.write(f"Downloaded: {self._get_current_time()}\n")
            
            logger.info(f"✓ Successfully downloaded {language} BERT model")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model for {language}: {e}")
            return False
    
    def download_multilingual_model(self) -> bool:
        """Download the multilingual XLM-RoBERTa model as fallback."""
        model_name = self.multilingual_model['model_name']
        
        logger.info(f"Downloading {self.multilingual_model['description']} ({self.multilingual_model['size']})")
        logger.info(f"Model: {model_name}")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # Create multilingual directory
            multilingual_dir = self.models_dir / 'multilingual'
            multilingual_dir.mkdir(exist_ok=True)
            
            # Download tokenizer
            logger.info("Downloading multilingual tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(multilingual_dir / 'tokenizer')
            logger.info(f"✓ Tokenizer saved to {multilingual_dir / 'tokenizer'}")
            
            # Download model
            logger.info("Downloading multilingual model weights...")
            model = AutoModel.from_pretrained(model_name)
            model.save_pretrained(multilingual_dir / 'model')
            logger.info(f"✓ Model saved to {multilingual_dir / 'model'}")
            
            # Save model info
            info_file = multilingual_dir / 'model_info.txt'
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"Language: multilingual\n")
                f.write(f"Model Name: {model_name}\n")
                f.write(f"Description: {self.multilingual_model['description']}\n")
                f.write(f"Size: {self.multilingual_model['size']}\n")
                f.write(f"Tokenizer: {multilingual_dir / 'tokenizer'}\n")
                f.write(f"Model: {multilingual_dir / 'model'}\n")
                f.write(f"Downloaded: {self._get_current_time()}\n")
            
            logger.info("✓ Successfully downloaded multilingual BERT model")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading multilingual model: {e}")
            return False
    
    def test_model(self, language: str) -> bool:
        """Test if a downloaded model works correctly."""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            if language == 'multilingual':
                model_dir = self.models_dir / 'multilingual'
            else:
                model_dir = self.models_dir / language
            
            tokenizer_path = model_dir / 'tokenizer'
            model_path = model_dir / 'model'
            
            if not (tokenizer_path.exists() and model_path.exists()):
                logger.error(f"Model files not found for {language}")
                return False
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModel.from_pretrained(model_path)
            
            # Test with a simple sentence
            test_sentences = {
                'german': "Das ist ein Test.",
                'french': "Ceci est un test.",
                'spanish': "Esto es una prueba.",
                'italian': "Questo è un test.",
                'portuguese': "Este é um teste.",
                'dutch': "Dit is een test.",
                'russian': "Это тест.",
                'multilingual': "This is a test."
            }
            
            test_text = test_sentences.get(language, "This is a test.")
            
            # Tokenize and encode
            inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)
            
            # Run through model
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Check output shape
            hidden_states = outputs.last_hidden_state
            logger.info(f"✓ {language} model test passed. Output shape: {hidden_states.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Model test failed for {language}: {e}")
            return False
    
    def _get_current_time(self) -> str:
        """Get current timestamp."""
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S')
    
    def download_all_models(self, include_multilingual: bool = True) -> Dict[str, bool]:
        """Download all language-specific models."""
        results = {}
        
        # Download language-specific models
        for language in self.language_models.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {language.upper()} BERT model")
            logger.info(f"{'='*60}")
            
            success = self.download_model(language)
            results[language] = success
            
            if success:
                # Test the model
                test_success = self.test_model(language)
                if not test_success:
                    logger.warning(f"Model test failed for {language}")
        
        # Download multilingual model
        if include_multilingual:
            logger.info(f"\n{'='*60}")
            logger.info("Processing MULTILINGUAL XLM-RoBERTa model")
            logger.info(f"{'='*60}")
            
            success = self.download_multilingual_model()
            results['multilingual'] = success
            
            if success:
                test_success = self.test_model('multilingual')
                if not test_success:
                    logger.warning("Multilingual model test failed")
        
        return results
    
    def get_download_summary(self, results: Dict[str, bool]):
        """Print download summary."""
        logger.info(f"\n{'='*60}")
        logger.info("BERT MODEL DOWNLOAD SUMMARY")
        logger.info(f"{'='*60}")
        
        successful = []
        failed = []
        
        for language, success in results.items():
            if success:
                successful.append(language)
            else:
                failed.append(language)
        
        logger.info(f"✓ Successfully downloaded: {len(successful)} models")
        for lang in successful:
            model_config = self.language_models.get(lang, self.multilingual_model)
            logger.info(f"  - {lang}: {model_config.get('model_name', 'XLM-RoBERTa')}")
        
        if failed:
            logger.warning(f"✗ Failed to download: {len(failed)} models")
            for lang in failed:
                logger.warning(f"  - {lang}")
        
        total_size_mb = len(successful) * 450  # Approximate
        logger.info(f"\nTotal approximate download size: ~{total_size_mb}MB")
        logger.info(f"Models saved to: {self.models_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download BERT models for multilingual antonym detection')
    parser.add_argument('--models-dir', default='../models/bert', help='Directory to save models')
    parser.add_argument('--language', help='Specific language to download (default: all)')
    parser.add_argument('--skip-multilingual', action='store_true', help='Skip downloading multilingual model')
    parser.add_argument('--test-only', action='store_true', help='Only test existing models')
    
    args = parser.parse_args()
    
    downloader = BERTModelDownloader(args.models_dir)
    
    if args.test_only:
        # Test existing models
        languages = list(downloader.language_models.keys())
        if not args.skip_multilingual:
            languages.append('multilingual')
        
        for language in languages:
            logger.info(f"Testing {language} model...")
            success = downloader.test_model(language)
            if not success:
                logger.error(f"Test failed for {language}")
        return
    
    # Check dependencies
    if not downloader.check_dependencies():
        logger.error("Failed to install required dependencies")
        sys.exit(1)
    
    # Determine which models to download
    if args.language:
        supported_languages = list(downloader.language_models.keys()) + ['multilingual']
        if args.language not in supported_languages:
            logger.error(f"Unsupported language: {args.language}")
            logger.info(f"Supported: {', '.join(supported_languages)}")
            sys.exit(1)
        
        if args.language == 'multilingual':
            results = {'multilingual': downloader.download_multilingual_model()}
        else:
            results = {args.language: downloader.download_model(args.language)}
    else:
        # Download all models
        results = downloader.download_all_models(include_multilingual=not args.skip_multilingual)
    
    # Print summary
    downloader.get_download_summary(results)
    
    # Final success check
    successful_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    logger.info(f"\nBERT model download completed! Successfully downloaded {successful_count}/{total_count} models")

if __name__ == "__main__":
    main()
