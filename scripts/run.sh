#!/bin/bash
# Master training script - runs all BERT models first, then all Dual Encoder models

echo "========================================================="
echo "  Multilingual Antonym Detection - Complete Training"
echo "========================================================="
echo "Starting time: $(date)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p ../logs

# Log file for the master script
MASTER_LOG="../logs/master_training_$(date +%Y%m%d_%H%M%S).log"
echo "Master log file: $MASTER_LOG"

# Function to log and display messages
log_message() {
    local message="$1"
    echo "$message" | tee -a "$MASTER_LOG"
}

log_message "========================================================="
log_message "PHASE 1: Training BERT Models (2 in parallel)"
log_message "========================================================="

# Run BERT training
log_message "Starting BERT training at: $(date)"
if ./train_all_bert.sh 2>&1 | tee -a "$MASTER_LOG"; then
    log_message "✓ BERT training completed successfully at: $(date)"
    BERT_SUCCESS=true
else
    log_message "✗ BERT training failed at: $(date)"
    BERT_SUCCESS=false
fi

log_message ""
log_message "Waiting 30 seconds before starting Dual Encoder training..."
sleep 30

log_message "========================================================="
log_message "PHASE 2: Training Dual Encoder Models (2 in parallel)"
log_message "========================================================="

# Run Dual Encoder training
log_message "Starting Dual Encoder training at: $(date)"
if ./train_all_dual_encoders.sh 2>&1 | tee -a "$MASTER_LOG"; then
    log_message "✓ Dual Encoder training completed successfully at: $(date)"
    DUAL_ENC_SUCCESS=true
else
    log_message "✗ Dual Encoder training failed at: $(date)"
    DUAL_ENC_SUCCESS=false
fi

log_message ""
log_message "========================================================="
log_message "FINAL TRAINING SUMMARY"
log_message "========================================================="
log_message "Completion time: $(date)"

# Display final results
if [ "$BERT_SUCCESS" = true ] && [ "$DUAL_ENC_SUCCESS" = true ]; then
    log_message "🎉 ALL TRAINING COMPLETED SUCCESSFULLY!"
    EXIT_CODE=0
elif [ "$BERT_SUCCESS" = true ]; then
    log_message "⚠️  BERT training succeeded, but Dual Encoder training failed"
    EXIT_CODE=1
elif [ "$DUAL_ENC_SUCCESS" = true ]; then
    log_message "⚠️  BERT training failed, but Dual Encoder training succeeded"
    EXIT_CODE=2
else
    log_message "❌ BOTH TRAINING PHASES FAILED"
    EXIT_CODE=3
fi

log_message ""
log_message "========================================================="
log_message "TRAINED MODELS SUMMARY"
log_message "========================================================="

log_message "BERT Models:"
if ls ../models/trained/bert/*.pt >/dev/null 2>&1; then
    ls -la ../models/trained/bert/*.pt | tee -a "$MASTER_LOG"
else
    log_message "No BERT models found"
fi

log_message ""
log_message "Dual Encoder Models:"
if ls ../models/trained/dual_encoder/*.pt >/dev/null 2>&1; then
    ls -la ../models/trained/dual_encoder/*.pt | tee -a "$MASTER_LOG"
else
    log_message "No Dual Encoder models found"
fi

log_message ""
log_message "========================================================="
log_message "LOG FILES CREATED"
log_message "========================================================="

log_message "Master log: $MASTER_LOG"
log_message ""
log_message "BERT training logs:"
if ls ../logs/bert_*.log >/dev/null 2>&1; then
    ls -la ../logs/bert_*.log | tee -a "$MASTER_LOG"
else
    log_message "No BERT log files found"
fi

log_message ""
log_message "Dual Encoder training logs:"
if ls ../logs/dual_encoder_*.log >/dev/null 2>&1; then
    ls -la ../logs/dual_encoder_*.log | tee -a "$MASTER_LOG"
else
    log_message "No Dual Encoder log files found"
fi

log_message ""
log_message "========================================================="
log_message "Training session completed with exit code: $EXIT_CODE"
log_message "Check individual log files for detailed training information"
log_message "========================================================="

exit $EXIT_CODE
