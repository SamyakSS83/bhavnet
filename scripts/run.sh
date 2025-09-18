!/bin/bash
Master training script - runs all BERT models first, then all Dual Encoder models

echo "========================================================="
echo "  Multilingual Antonym Detection - Complete Training"
echo "========================================================="
echo "Starting time: $(date)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /home/samyak/scratch/temp/multilingual_antonym_detection/logs

# Log file for the master script
MASTER_LOG="/home/samyak/scratch/temp/multilingual_antonym_detection/logs/master_training_$(date +%Y%m%d_%H%M%S).log"
echo "Master log file: $MASTER_LOG"
# MASTER_LOG="/home/samyak/scratch/temp/multilingual_antonym_detection/logs/master_training_20250918_181913.log"

# # Function to log and display messages
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
if ls /home/samyak/scratch/temp/multilingual_antonym_detection/models/trained/bert/*.pt >/dev/null 2>&1; then
    ls -la /home/samyak/scratch/temp/multilingual_antonym_detection/models/trained/bert/*.pt | tee -a "$MASTER_LOG"
else
    log_message "No BERT models found"
fi

log_message ""
log_message "Dual Encoder Models:"
if ls /home/samyak/scratch/temp/multilingual_antonym_detection/models/trained/dual_encoder/*.pt >/dev/null 2>&1; then
    ls -la /home/samyak/scratch/temp/multilingual_antonym_detection/models/trained/dual_encoder/*.pt | tee -a "$MASTER_LOG"
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
if ls /home/samyak/scratch/temp/multilingual_antonym_detection/logs/bert_*.log >/dev/null 2>&1; then
    ls -la /home/samyak/scratch/temp/multilingual_antonym_detection/logs/bert_*.log | tee -a "$MASTER_LOG"
else
    log_message "No BERT log files found"
fi

log_message ""
log_message "Dual Encoder training logs:"
if ls /home/samyak/scratch/temp/multilingual_antonym_detection/logs/dual_encoder_*.log >/dev/null 2>&1; then
    ls -la /home/samyak/scratch/temp/multilingual_antonym_detection/logs/dual_encoder_*.log | tee -a "$MASTER_LOG"
else
    log_message "No Dual Encoder log files found"
fi

# log_message ""
# log_message "========================================================="
# log_message "PHASE 3: ANALYSIS & BASELINES"
# log_message "========================================================="

# # Run analysis: embeddings, plots, and baseline probes
# ANALYSIS_CMD=("python3" "analysis.py")
# log_message "Running analysis script (embeddings + basic plots)"
# "${ANALYSIS_CMD[@]}" --config /home/samyak/scratch/temp/multilingual_antonym_detection/config/language_config.yaml 2>&1 | tee -a "$MASTER_LOG"

# # Run baseline probes (mBERT and XLM-R) in parallel for all languages
# log_message "Launching baseline probes (mBERT, XLM-R) sequentially (one language at a time)"
# for lang in $(python3 - <<'PY'
# import yaml
# cfg = yaml.safe_load(open('/home/samyak/scratch/temp/multilingual_antonym_detection/config/language_config.yaml'))
# print(' '.join(cfg['languages'].keys()))
# PY
# ); do
#     log_message "Running baselines for $lang: mBERT"
#     python3 analysis.py --config /home/samyak/scratch/temp/multilingual_antonym_detection/config/language_config.yaml --languages $lang --baseline mbert 2>&1 | tee -a /home/samyak/scratch/temp/multilingual_antonym_detection/logs/analysis_${lang}_mbert.log

#     log_message "Running baselines for $lang: XLM-R"
#     python3 analysis.py --config /home/samyak/scratch/temp/multilingual_antonym_detection/config/language_config.yaml --languages $lang --baseline xlmr 2>&1 | tee -a /home/samyak/scratch/temp/multilingual_antonym_detection/logs/analysis_${lang}_xlmr.log
# done

# # Run ablation (small example grid) for first language to demonstrate
# FIRST_LANG=$(python3 - <<'PY'
# import yaml
# cfg = yaml.safe_load(open('/home/samyak/scratch/temp/multilingual_antonym_detection/config/language_config.yaml'))
# print(list(cfg['languages'].keys())[0])
# PY
# )
# log_message "Launching ablation runs for $FIRST_LANG"
# python3 analysis.py --config /home/samyak/scratch/temp/multilingual_antonym_detection/config/language_config.yaml --ablation --languages $FIRST_LANG 2>&1 | tee -a /home/samyak/scratch/temp/multilingual_antonym_detection/logs/ablation_${FIRST_LANG}.log

# log_message ""
# log_message "========================================================="
# log_message "Training session completed with exit code: $EXIT_CODE"
# log_message "Check individual log files for detailed training information"
# log_message "========================================================="

# exit $EXIT_CODE
