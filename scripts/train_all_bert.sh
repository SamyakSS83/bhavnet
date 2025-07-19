#!/bin/bash
# Train all languages one by one to avoid memory issues

echo "=== Training all languages for BERT models ==="

languages=("german" "french" "spanish" "italian" "portuguese" "dutch" "russian")

for lang in "${languages[@]}"; do
    echo ""
    echo "==============================================="
    echo "Training BERT model for $lang"
    echo "==============================================="
    
    python3 train_models.py --language $lang --model-type bert --bert-epochs 12
    
    # Check if training was successful by looking for the model file
    if [ -f "../models/trained/bert/best_${lang}_bert_model.pt" ]; then
        echo "✓ Successfully trained $lang BERT model"
    else
        echo "✗ Failed to train $lang BERT model"
    fi
    
    echo "Waiting 5 seconds before next training..."
    sleep 5
done

echo ""
echo "==============================================="
echo "Training Summary"
echo "==============================================="
echo "Trained models:"
ls -la ../models/trained/bert/
