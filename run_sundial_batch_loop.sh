#!/bin/bash

# Script to run time_series_prediction_sundial_batch.py with different forecast_length and lookback_length values

# Base parameters
EVAL_MODE="multi"
NUM_SAMPLES=100
SAMPLE_SEED=123
SUNDIAL_NUM_SAMPLES=20

# Values to loop through
FORECAST_LENGTHS=(1)
LOOKBACK_LENGTHS=(32 64 128 256 512 1024)

# Loop through forecast_length and lookback_length combinations
for forecast_length in "${FORECAST_LENGTHS[@]}"; do
    for lookback_length in "${LOOKBACK_LENGTHS[@]}"; do
        echo "=========================================="
        echo "Running with forecast_length=$forecast_length, lookback_length=$lookback_length"
        echo "=========================================="
        
        python time_series_prediction_sundial_batch.py \
            --eval-mode "$EVAL_MODE" \
            --num-samples "$NUM_SAMPLES" \
            --sample-seed "$SAMPLE_SEED" \
            --forecast-length "$forecast_length" \
            --sundial-num-samples "$SUNDIAL_NUM_SAMPLES" \
            --lookback-length "$lookback_length"
        
        # Check if the command succeeded
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed with forecast_length=$forecast_length, lookback_length=$lookback_length"
        else
            echo "SUCCESS: Completed forecast_length=$forecast_length, lookback_length=$lookback_length"
        fi
        echo ""
    done
done

echo "All runs completed!"

