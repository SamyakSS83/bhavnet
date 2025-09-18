#!/bin/bash
# Train all languages in parallel (2 at a time) for BERT models

echo "=== Training all languages for BERT models (2 in parallel) ==="

languages=("german" "french" "spanish" "italian" "portuguese" "dutch" "russian")

# Function to train a single language
train_language() {
    local lang=$1
    local logfile="/home/samyak/scratch/temp/multilingual_antonym_detection/logs/bert_${lang}_$(date +%Y%m%d_%H%M%S).log"
    
    # Create logs directory if it doesn't exist
    mkdir -p /home/samyak/scratch/temp/multilingual_antonym_detection/logs
    
    echo ""
    echo "==============================================="
    echo "Training BERT model for $lang"
    echo "Log file: $logfile"
    echo "==============================================="
    
    # Run training and capture all output to log file
    python3 train_models.py --language $lang --model-type bert --epochs 12 > "$logfile" 2>&1
    
    # Check if training was successful by looking for the model file
    if [ -f "/home/samyak/scratch/temp/multilingual_antonym_detection/models/trained/bert/best_${lang}_bert_model.pt" ]; then
        echo "✓ Successfully trained $lang BERT model"
        echo "✓ Log saved to: $logfile"
    else
        echo "✗ Failed to train $lang BERT model"
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
echo "Trained models:"
ls -la /home/samyak/scratch/temp/multilingual_antonym_detection/models/trained/bert/

echo ""
echo "Log files created:"
ls -la /home/samyak/scratch/temp/multilingual_antonym_detection/logs/bert_*.log 2>/dev/null || echo "No BERT log files found"
