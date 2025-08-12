#!/bin/bash
# Train all languages in parallel (2 at a time) for Dual Encoder models

echo "=== Training all languages for Dual Encoder models (2 in parallel) ==="

languages=("german" "french" "spanish" "italian" "portuguese" "dutch" "russian")

# Function to train a single language
train_language() {
    local lang=$1
    local logfile="../logs/dual_encoder_${lang}_$(date +%Y%m%d_%H%M%S).log"
    
    # Create logs directory if it doesn't exist
    mkdir -p ../logs
    
    echo ""
    echo "==============================================="
    echo "Training Dual Encoder model for $lang"
    echo "Log file: $logfile"
    echo "==============================================="
    
    # Train dual encoder with 20 epochs and early stopping, capture all output to log file
    python3 train_models.py --language $lang --model-type dual_encoder --dual-encoder-epochs 20 > "$logfile" 2>&1
    
    # Check if training was successful by looking for the model file
    if [ -f "../models/trained/dual_encoder/best_${lang}_dual_encoder_model.pt" ]; then
        echo "✓ Successfully trained $lang Dual Encoder model"
        echo "✓ Log saved to: $logfile"
    else
        echo "✗ Failed to train $lang Dual Encoder model"
        echo "✗ Check log file for errors: $logfile"
    fi
}

# Process languages in pairs
for ((i=0; i<${#languages[@]}; i+=2)); do
    lang1=${languages[i]}
    lang2=${languages[i+1]}
    
    echo ""
    echo "==============================================="
    echo "Starting parallel training: $lang1 and $lang2"
    echo "==============================================="
    
    # Start both trainings in background
    if [ -n "$lang2" ]; then
        train_language $lang1 &
        pid1=$!
        train_language $lang2 &
        pid2=$!
        
        # Wait for both to complete
        wait $pid1
        wait $pid2
        
        echo "Completed parallel training: $lang1 and $lang2"
    else
        # Handle odd number of languages (train last one alone)
        train_language $lang1
    fi
    
    echo "Waiting 10 seconds before next pair..."
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

echo ""
echo "Log files created:"
echo "BERT logs:"
ls -la ../logs/bert_*.log 2>/dev/null || echo "No BERT log files found"
echo "Dual Encoder logs:"
ls -la ../logs/dual_encoder_*.log 2>/dev/null || echo "No Dual Encoder log files found"
