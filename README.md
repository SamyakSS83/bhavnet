# Multilingual Antonym Detection System

A complete system for detecting antonyms across multiple languages using BERT and Graph Neural Networks.

## Features

- **7 Languages Supported**: German, French, Spanish, Italian, Portuguese, Dutch, Russian
- **Real Data Sources**: WordNet (OMW) + ConceptNet APIs  
- **Dual Model Architecture**: BERT + Graph Neural Network Dual Encoder
- **Language-Specific Models**: Optimized BERT models for each language
- **Complete Pipeline**: Data download → Model training → Evaluation

## Quick Start

### 1. Complete System Setup (Recommended)

```bash
# Clone and enter directory
cd multilingual_antonym_detection/scripts

# Run complete setup (downloads everything)
python system_setup.py
```

This will:
- Install all required packages (PyTorch, Transformers, etc.)
- Download real antonym datasets for all languages
- Download language-specific BERT models
- Verify everything is working

### 2. Manual Setup (Step by Step)

```bash
# Install requirements
python setup_system.py --step requirements

# Download datasets (real data from WordNet + ConceptNet)
python dataset_downloader.py

# Download BERT models for all languages
python bert_downloader.py

# Check setup
python setup_system.py --step check
```

### 3. Train Models

```bash
# Train all models for all languages
python scripts/train_models.py

# Train specific language only
python scripts/train_models.py --language german

# Train only BERT models
python scripts/train_models.py --model-type bert
```

## Dataset Statistics

Our system downloads **real antonym pairs** from professional sources:

| Language   | WordNet Pairs | ConceptNet Pairs | Total Pairs |
|------------|---------------|------------------|-------------|
| German     | 0*            | 2,678           | 2,678       |
| French     | 571           | 5,703           | 6,095       |
| Spanish    | 1,634         | 878             | 2,263       |
| Italian    | 917           | 1,889           | 2,495       |
| Portuguese | TBD           | TBD             | TBD         |
| Dutch      | TBD           | TBD             | TBD         |
| Russian    | TBD           | TBD             | TBD         |

*German WordNet data not available in OMW format, using ConceptNet only.

## Project Structure

```
multilingual_antonym_detection/
├── scripts/
│   ├── dataset_downloader.py      # Downloads real antonym datasets
│   ├── bert_downloader.py         # Downloads language-specific BERT models  
│   ├── train_models.py           # Complete training system
│   └── setup_system.py           # One-click setup script
├── models/
│   ├── multilingual_bert.py      # BERT-based antonym detection
│   ├── multilingual_dual_encoder.py  # Graph Neural Network dual encoder
│   ├── bert/                     # Downloaded BERT models
│   └── trained/                  # Trained models
├── datasets/                     # Downloaded antonym datasets
│   ├── german/
│   ├── french/
│   └── ...
├── config/
│   └── training_config.yaml     # Training configuration
└── README.md
```

## Model Architecture

### 1. BERT Classification
- Language-specific BERT models (e.g., CamemBERT for French)
- Fine-tuned for binary antonym classification
- Input: `[CLS] word1 [SEP] word2 [SEP]`

### 2. Dual Encoder Graph Transformer  
- Separate synonym and antonym projection branches
- Graph Neural Network with 2-node word pairs
- Mathematical formulation with margin-based loss

## Data Sources

### Primary Sources
- **Open Multilingual WordNet (OMW)**: Professional multilingual wordnets
- **ConceptNet**: Large-scale semantic knowledge graph

### Quality Features
- Real linguistic data (not synthetic)
- Language-specific character cleaning (ñ, ü, ç, etc.)
- Automatic train/validation/test splits (70%/15%/15%)
- Prefix-based antonym detection for WordNet data

## BERT Models Used

| Language   | Model                                    | Size  |
|------------|------------------------------------------|-------|
| German     | `dbmdz/bert-base-german-cased`          | 420MB |
| French     | `camembert-base`                         | 440MB |
| Spanish    | `dccuchile/bert-base-spanish-wwm-cased`  | 420MB |
| Italian    | `dbmdz/bert-base-italian-cased`         | 420MB |
| Portuguese | `neuralmind/bert-base-portuguese-cased` | 420MB |
| Dutch      | `GroNLP/bert-base-dutch-cased`          | 420MB |
| Russian    | `DeepPavlov/rubert-base-cased`          | 670MB |
| Fallback   | `FacebookAI/xlm-roberta-base`           | 560MB |

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- Internet connection (for downloading)
- ~4GB disk space (for all models and data)
- GPU recommended (but not required)

## Usage Examples

```python
# Load trained model for German
from models.multilingual_bert import load_trained_model

model, tokenizer = load_trained_model('german')

# Predict if words are antonyms
word1, word2 = "gut", "schlecht"  # good, bad
is_antonym = predict_antonym(model, tokenizer, word1, word2)
print(f"{word1} - {word2}: {is_antonym}")  # True
```

## Advanced Usage

### Custom Training Configuration

Edit `config/training_config.yaml`:

```yaml
training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 5

language_overrides:
  german:
    batch_size: 64  # German has most data
```

### Single Language Setup

```bash
# Download only German data
cd scripts
python dataset_downloader.py --language german

# Download only German BERT model  
python bert_downloader.py --language german

# Train only German models
python train_models.py --language german
```

## Contributing

This system is designed for research and educational purposes. The datasets are sourced from open repositories and APIs.

## License

Open source - uses data from WordNet and ConceptNet under their respective licenses.
