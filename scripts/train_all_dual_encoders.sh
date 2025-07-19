#!/bin/bash
# Train all languages with dual encoder models

echo "=== Training all languages for Dual Encoder models ==="

languages=("german" "french" "spanish" "italian" "portuguese" "dutch" "russian")

for lang in "${languages[@]}"; do
    echo ""
    echo "==============================================="
    echo "Training Dual Encoder model for $lang"
    echo "==============================================="
    
    # Train dual encoder with 20 epochs and early stopping
    python3 train_models.py --language $lang --model-type dual_encoder --dual-encoder-epochs 20
    
    # Check if training was successful by looking for the model file
    if [ -f "../models/trained/dual_encoder/best_${lang}_dual_encoder_model.pt" ]; then
        echo "✓ Successfully trained $lang Dual Encoder model"
    else
        echo "✗ Failed to train $lang Dual Encoder model"
    fi
    
    echo "Waiting 10 seconds before next training..."
    sleep 10
done

echo ""
echo "==============================================="
echo "Training Summary"
echo "==============================================="
echo "Trained dual encoder models:"
ls -la ../models/trained/dual_encoder/

echo ""
echo "All models summary:"
echo "BERT models:"
ls -la ../models/trained/bert/ 2>/dev/null || echo "No BERT models found"
echo "Dual Encoder models:"
ls -la ../models/trained/dual_encoder/ 2>/dev/null || echo "No Dual Encoder models found"
