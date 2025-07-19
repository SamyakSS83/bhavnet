#!/usr/bin/env python3
"""
Complete Setup Script for Multilingual Antonym Detection System
Downloads datasets, BERT models, and sets up the training environment.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultilingualSetup:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.scripts_dir = self.base_dir / "scripts"
        
    def install_requirements(self) -> bool:
        """Install required Python packages."""
        logger.info("Installing required Python packages...")
        
        requirements = [
            "torch",
            "transformers>=4.20.0",
            "pandas",
            "numpy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "tqdm",
            "pyyaml",
            "requests"
        ]
        
        try:
            for package in requirements:
                logger.info(f"Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                logger.info(f"✓ {package} installed")
            
            logger.info("✓ All requirements installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e}")
            return False
    
    def download_datasets(self) -> bool:
        """Download multilingual antonym datasets."""
        logger.info("Downloading multilingual antonym datasets...")
        
        dataset_script = self.scripts_dir / "dataset_downloader.py"
        if not dataset_script.exists():
            logger.error(f"Dataset downloader script not found: {dataset_script}")
            return False
        
        try:
            cmd = [sys.executable, str(dataset_script)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.scripts_dir)
            
            if result.returncode == 0:
                logger.info("✓ Datasets downloaded successfully")
                return True
            else:
                logger.error(f"Dataset download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running dataset downloader: {e}")
            return False
    
    def download_bert_models(self) -> bool:
        """Download BERT models for all languages."""
        logger.info("Downloading BERT models...")
        
        bert_script = self.scripts_dir / "bert_downloader.py"
        if not bert_script.exists():
            logger.error(f"BERT downloader script not found: {bert_script}")
            return False
        
        try:
            cmd = [sys.executable, str(bert_script)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.scripts_dir)
            
            if result.returncode == 0:
                logger.info("✓ BERT models downloaded successfully")
                return True
            else:
                logger.error(f"BERT download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running BERT downloader: {e}")
            return False
    
    def check_setup(self) -> bool:
        """Check if setup is complete."""
        logger.info("Checking setup...")
        
        checks = []
        
        # Check datasets
        datasets_dir = self.base_dir / "datasets"
        if datasets_dir.exists():
            languages_with_data = 0
            for lang_dir in datasets_dir.iterdir():
                if lang_dir.is_dir():
                    train_file = lang_dir / "train.txt"
                    val_file = lang_dir / "val.txt" 
                    test_file = lang_dir / "test.txt"
                    if train_file.exists() and val_file.exists() and test_file.exists():
                        languages_with_data += 1
            
            if languages_with_data > 0:
                logger.info(f"✓ Found datasets for {languages_with_data} languages")
                checks.append(True)
            else:
                logger.error("✗ No complete datasets found")
                checks.append(False)
        else:
            logger.error("✗ Datasets directory not found")
            checks.append(False)
        
        # Check BERT models
        bert_dir = self.base_dir / "models" / "bert"
        if bert_dir.exists():
            bert_models = 0
            for model_dir in bert_dir.iterdir():
                if model_dir.is_dir():
                    model_path = model_dir / "model"
                    tokenizer_path = model_dir / "tokenizer"
                    if model_path.exists() and tokenizer_path.exists():
                        bert_models += 1
            
            if bert_models > 0:
                logger.info(f"✓ Found BERT models for {bert_models} languages")
                checks.append(True)
            else:
                logger.error("✗ No BERT models found")
                checks.append(False)
        else:
            logger.error("✗ BERT models directory not found")
            checks.append(False)
        
        # Check Python packages
        try:
            import torch
            import transformers
            import pandas
            logger.info("✓ Required Python packages available")
            checks.append(True)
        except ImportError as e:
            logger.error(f"✗ Missing Python packages: {e}")
            checks.append(False)
        
        return all(checks)
    
    def run_training_check(self) -> bool:
        """Run training system check."""
        logger.info("Checking training system...")
        
        train_script = self.scripts_dir / "train_models.py"
        if not train_script.exists():
            logger.error(f"Training script not found: {train_script}")
            return False
        
        try:
            cmd = [sys.executable, str(train_script), "--check-only"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.scripts_dir)
            
            if result.returncode == 0:
                logger.info("✓ Training system check passed")
                return True
            else:
                logger.error(f"Training system check failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running training check: {e}")
            return False
    
    def setup_complete_system(self) -> bool:
        """Set up the complete multilingual antonym detection system."""
        logger.info("Starting complete system setup...")
        
        steps = [
            ("Installing requirements", self.install_requirements),
            ("Downloading datasets", self.download_datasets),
            ("Downloading BERT models", self.download_bert_models),
            ("Checking setup", self.check_setup),
            ("Training system check", self.run_training_check)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Step: {step_name}")
            logger.info(f"{'='*60}")
            
            try:
                if not step_func():
                    logger.error(f"Step failed: {step_name}")
                    return False
            except Exception as e:
                logger.error(f"Error in step '{step_name}': {e}")
                return False
        
        logger.info(f"\\n{'='*60}")
        logger.info("SETUP COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*60}")
        logger.info("You can now train models using:")
        logger.info("  python scripts/train_models.py")
        logger.info("Or train a specific language:")
        logger.info("  python scripts/train_models.py --language german")
        
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup multilingual antonym detection system')
    parser.add_argument('--step', choices=['requirements', 'datasets', 'bert', 'check', 'training-check'], 
                       help='Run specific setup step only')
    parser.add_argument('--base-dir', default='.', help='Base directory of the project')
    
    args = parser.parse_args()
    
    setup = MultilingualSetup(args.base_dir)
    
    if args.step:
        # Run specific step
        if args.step == 'requirements':
            success = setup.install_requirements()
        elif args.step == 'datasets':
            success = setup.download_datasets()
        elif args.step == 'bert':
            success = setup.download_bert_models()
        elif args.step == 'check':
            success = setup.check_setup()
        elif args.step == 'training-check':
            success = setup.run_training_check()
        
        if success:
            logger.info(f"✓ Step '{args.step}' completed successfully")
        else:
            logger.error(f"✗ Step '{args.step}' failed")
            sys.exit(1)
    else:
        # Run complete setup
        success = setup.setup_complete_system()
        if not success:
            logger.error("Setup failed")
            sys.exit(1)

if __name__ == "__main__":
    main()
