# sundial-lai

A comprehensive benchmark for evaluating [Sundial](https://github.com/thuml/Sundial), a family of highly capable time series foundation models, on Leaf Area Index (LAI) forecasting tasks using the HiQ LAI archive.

## Overview

This repository provides tools and scripts for benchmarking Sundial foundation models on LAI time series prediction. The benchmark evaluates Sundial's zero-shot and fine-tuned performance across various forecasting horizons and context lengths, comparing against traditional baselines (ARIMA, SARIMA) and deep learning approaches (LSTM).

## Dataset

The benchmark uses the **HiQ LAI archive**, which provides high-quality Leaf Area Index data with:
- **Spatial coverage**: Continental United States
- **Temporal resolution**: Daily observations
- **Temporal span**: Multiple years (2001+)
- **Spatial resolution**: Gridded data aligned with GridMET

The dataset exhibits substantial spatial and temporal heterogeneity, with pronounced seasonal cycles and geographic gradients in vegetation greenness, making it an ideal testbed for evaluating time series foundation models.

### Dataset Configuration

**Note**: The dataset files are not included in this repository. You need to provide your own HiQ LAI NetCDF files.

To configure the dataset path, set the environment variable:

```bash
export DL4VEG_NC_FILE="/path/to/your/HiQ_LAI_regrid_to_gridMET_all.nc"
```

**Test the dataset loading:**

You can test if your dataset path is configured correctly by running:

```bash
python load_all_data.py
```

This will load the dataset and print basic information about it (time range, spatial coverage, etc.). If the file is not found, you'll get a clear error message with the path it tried to use.

## Features

- **Zero-shot evaluation**: Test Sundial's pre-trained performance without fine-tuning
- **Multi-step forecasting**: Support for forecasting horizons from 1 to 12+ time steps
- **Flexible context length**: Evaluate performance with lookback lengths ranging from 32 to 1024+ time steps
- **Multi-location evaluation**: Aggregate metrics across multiple spatial locations
- **Comprehensive metrics**: MAE, RMSE, MAPE, R², MBE, CVRMSE
- **Baseline comparisons**: Compare against ARIMA, SARIMA, and LSTM models

## Installation

### Option 1: Using Conda (Recommended)

If you have conda installed, you can create the environment from the provided `environment.yml`:

```bash
# Create conda environment from file
conda env create -f environment.yml

# Activate the environment
conda activate dl4veg
```

Or if you already have a conda environment (e.g., `dl4veg`), activate it and install dependencies:

```bash
# Activate your existing conda environment
conda activate dl4veg

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### Option 2: Using pip only

Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
# Core dependencies
pip install torch transformers xarray numpy pandas scipy scikit-learn matplotlib tqdm

# For SARIMA baseline
pip install statsmodels

# For NetCDF file handling
pip install netcdf4 h5netcdf

# For data visualization (optional)
pip install seaborn
```

### Model Access

Sundial models are available on HuggingFace:
- Base model: `thuml/sundial-base-128m`
- Alternative: `thuml/timer-base-84m` (Timer-XL family)

The models will be automatically downloaded from HuggingFace on first use.

## Quick Start

### 1. Activate Environment

If using conda:

```bash
conda activate dl4veg
```

### 2. Configure Dataset Path

First, set the environment variable for your dataset:

```bash
export DL4VEG_NC_FILE="/path/to/your/HiQ_LAI_regrid_to_gridMET_all.nc"
```

### 3. Zero-shot Evaluation

Evaluate Sundial on a single location:

```bash
python time_series_prediction_sundial_batch.py \
    --eval-mode single \
    --forecast-length 4 \
    --lookback-length 512 \
    --sundial-num-samples 20
```

Evaluate Sundial across multiple locations:

```bash
python time_series_prediction_sundial_batch.py \
    --eval-mode multi \
    --num-samples 100 \
    --sample-seed 123 \
    --forecast-length 12 \
    --lookback-length 1024 \
    --sundial-num-samples 20
```

### 4. Baseline Comparisons

Run LSTM baseline:

```bash
python time_series_prediction_lstm.py \
    --eval-mode multi \
    --num-samples 100 \
    --sample-seed 123 \
    --forecast-length 4 \
    --window-size 32
```

Run simple baseline methods (mean/last/trend):

```bash
python time_series_prediction_baseline_batch.py \
    --eval-mode multi \
    --num-samples 100 \
    --sample-seed 123 \
    --forecast-length 4 \
    --methods mean last trend \
    --window-size 4
```

Run ARIMA/SARIMA baselines:

```bash
python time_series_prediction_baseline_batch.py \
    --eval-mode multi \
    --num-samples 100 \
    --sample-seed 123 \
    --forecast-length 4 \
    --methods arima sarima \
    --window-size 32
```

### Batch Evaluation

Run comprehensive evaluation across multiple forecast lengths and lookback lengths:

```bash
bash run_sundial_batch_loop.sh
```

This script evaluates combinations of:
- Forecast lengths: 8, 12
- Lookback lengths: 32, 64, 128, 256, 512, 1024

## Key Findings

- **Context length matters**: Sundial's performance improves significantly with longer input windows, demonstrating the importance of sufficient context length for capturing seasonal patterns
- **Zero-shot capability**: Sundial achieves competitive performance without fine-tuning, showcasing its foundation model capabilities
- **Multi-step forecasting**: The model maintains reasonable performance even for longer forecast horizons (8-12 steps)

## Project Structure

```
sundial-lai/
├── README.md                          # This file
├── LICENSE                            # Apache License 2.0
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies (pip)
├── environment.yml                    # Conda environment file (optional)
│
├── Core Model Scripts                 # Core model evaluation scripts
│   ├── time_series_prediction_sundial_batch.py      # Sundial zero-shot evaluation
│   ├── time_series_prediction_lstm.py               # LSTM baseline
│   └── time_series_prediction_baseline_batch.py     # Baseline methods (mean/last/trend + ARIMA/SARIMA)
│
├── Data Utilities                     # Data loading utilities
│   ├── load_all_data.py              # Main data loading functions
│   └── util.py                       # Helper functions (metrics, location selection, etc.)
│
├── Visualization                      # Visualization scripts
│   ├── plot_ground_truth.py           # Plot ground truth data
│   ├── plot_spatial_distribution.py   # Plot spatial distributions
│   ├── plot_window_size_effect.py    # Analyze window size effects
│   └── plot_forecast_length_effect.py # Analyze forecast length effects
│
└── Scripts                            # Batch processing scripts
    ├── run_sundial_batch_loop.sh     # Batch evaluation script
    └── train_lstm.sh                 # LSTM training script
```

### Model Scripts Description

- **`time_series_prediction_sundial_batch.py`**: Evaluates Sundial foundation model (zero-shot) on LAI time series forecasting
- **`time_series_prediction_lstm.py`**: LSTM baseline model with training and evaluation capabilities
- **`time_series_prediction_baseline_batch.py`**: Simple baseline methods including:
  - **Mean/Last/Trend**: Simple statistical methods (mean of window, last value, linear trend)
  - **ARIMA/SARIMA**: Traditional time series models (ARIMA and Seasonal ARIMA)

## Evaluation Metrics

The benchmark reports the following metrics:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)
- **MBE** (Mean Bias Error)
- **CVRMSE** (Coefficient of Variation of RMSE)

For multi-location evaluation, metrics are reported as mean ± standard deviation across locations.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{liu2025sundial,
  title={Sundial: A Family of Highly Capable Time Series Foundation Models},
  author={Liu, Zhijian and others},
  journal={ICML 2025},
  year={2025}
}
```

## References

- **Sundial**: [GitHub](https://github.com/thuml/Sundial) | [Paper](https://arxiv.org/abs/2502.00816) | [HuggingFace](https://huggingface.co/thuml/sundial-base-128m)
- **HiQ LAI Archive**: (Add reference when available)

## Visualization

Generate visualizations of results:

```bash
# Plot ground truth data
python plot_ground_truth.py

# Analyze window size effects
python plot_window_size_effect.py

# Analyze forecast length effects
python plot_forecast_length_effect.py

# Plot spatial distributions
python plot_spatial_distribution.py
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

This benchmark builds upon the Sundial foundation model developed by THUML Lab. We thank the developers for making their models publicly available.

