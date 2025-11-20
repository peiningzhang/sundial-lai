# Sundial-LAI-Benchmark

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

## Features

- **Zero-shot evaluation**: Test Sundial's pre-trained performance without fine-tuning
- **Multi-step forecasting**: Support for forecasting horizons from 1 to 12+ time steps
- **Flexible context length**: Evaluate performance with lookback lengths ranging from 32 to 1024+ time steps
- **Multi-location evaluation**: Aggregate metrics across multiple spatial locations
- **Comprehensive metrics**: MAE, RMSE, MAPE, R², MBE, CVRMSE
- **Baseline comparisons**: Compare against ARIMA, SARIMA, and LSTM models
- **Fine-tuning support**: LoRA-based fine-tuning capabilities for domain adaptation

## Installation

### Requirements

```bash
# Core dependencies
pip install torch transformers xarray numpy pandas scipy scikit-learn matplotlib tqdm

# For SARIMA baseline
pip install statsmodels

# For data loading utilities
# (Add your data loading dependencies here)
```

### Model Access

Sundial models are available on HuggingFace:
- Base model: `thuml/sundial-base-128m`
- Alternative: `thuml/timer-base-84m` (Timer-XL family)

The models will be automatically downloaded from HuggingFace on first use.

## Quick Start

### Zero-shot Evaluation

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
Sundial-LAI-Benchmark/
├── README.md                          # This file
├── time_series_prediction_sundial_batch.py  # Main evaluation script
├── run_sundial_batch_loop.sh          # Batch evaluation script
├── time_series_prediction_sundial_finetuning.py  # Fine-tuning script
├── time_series_prediction_baseline_batch.py  # Baseline methods (ARIMA, SARIMA)
├── time_series_prediction_lstm.py     # LSTM baseline
├── load_all_data.py                   # Data loading utilities
├── util.py                            # Helper functions
└── figures/                           # Generated plots and visualizations
```

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

## License

[Add your license information here]

## Contact

[Add contact information here]

## Acknowledgments

This benchmark builds upon the Sundial foundation model developed by THUML Lab. We thank the developers for making their models publicly available.

