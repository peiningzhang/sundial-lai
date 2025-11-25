#!/bin/bash

# Train LSTM models with different window sizes
# Window sizes to iterate over
WINDOW_SIZES=(16 32 64 128 256 512)

# Base command arguments
TRAIN_FRACTION=0.9
EVAL_MODE=multi
NUM_SAMPLES=100
SAMPLE_SEED=123
TRAINING_EVAL_MODE=multi
TRAIN_DATA_FRACTION=1
EPOCHS=5
CELLS_PER_EPOCH_FRACTION=1.0
FORECAST_LENGTH=12

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Loop through each window size
for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    echo "=========================================="
    echo "Training LSTM with window-size=${WINDOW_SIZE}"
    echo "=========================================="
    
    python "${SCRIPT_DIR}/time_series_prediction_lstm.py" \
        --window-size "${WINDOW_SIZE}" \
        --train-fraction "${TRAIN_FRACTION}" \
        --eval-mode "${EVAL_MODE}" \
        --num-samples "${NUM_SAMPLES}" \
        --sample-seed "${SAMPLE_SEED}" \
        --training-eval-mode "${TRAINING_EVAL_MODE}" \
        --train-data-fraction "${TRAIN_DATA_FRACTION}" \
        --epochs "${EPOCHS}" \
        --cells-per-epoch-fraction "${CELLS_PER_EPOCH_FRACTION}" \
        --forecast-length "${FORECAST_LENGTH}"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for window-size=${WINDOW_SIZE}"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "All training completed successfully!"
echo "=========================================="

