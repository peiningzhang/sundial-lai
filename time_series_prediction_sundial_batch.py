"""
Time Series Prediction using Sliding Window Average

This module implements a simple time series prediction method using
sliding window average for LAI (Leaf Area Index) data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

import xarray as xr
from tqdm import tqdm
from util import rolling_eval, select_single_location, sample_locations, pretty_print_sampled_locations

from load_all_data import load_all_data, get_time_series

# Sundial (transformers) optional import
try:
    import torch
    from transformers import AutoModelForCausalLM
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]


def sliding_window_average(
    data: np.ndarray,
    window_size: int,
    center: bool = False,
) -> np.ndarray:
    """
    Compute sliding window average of a time series.

    Parameters
    ----------
    data : np.ndarray
        1D array of time series data.
    window_size : int
        Size of the sliding window.
    center : bool, default False
        If True, center the window. If False, use trailing window.

    Returns
    -------
    np.ndarray
        Array of smoothed values (same length as input, dtype=float).
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    
    # Ensure float dtype
    data = np.asarray(data, dtype=float)
    
    if len(data) < window_size:
        # If data is shorter than window, return mean of all data
        mean_val = float(np.nanmean(data))
        return np.full(len(data), mean_val, dtype=float)
    
    if center:
        # Centered window: pad with NaN at edges
        pad_left = window_size // 2
        pad_right = window_size - 1 - pad_left
        padded = np.pad(data, (pad_left, pad_right), 
                       mode='constant', constant_values=np.nan)
        # Use nan-safe rolling average
        series = pd.Series(padded)
        result = series.rolling(window=window_size, min_periods=1).mean().values
        # Extract the valid window slice to match input length
        result = result[pad_left:pad_left + len(data)]
    else:
        # Trailing window: use pandas rolling for NaN handling
        series = pd.Series(data)
        result = series.rolling(window=window_size, min_periods=1).mean().values
    
    # Ensure float dtype and same length
    result = np.asarray(result, dtype=float)
    assert len(result) == len(data), f"Output length {len(result)} != input length {len(data)}"
    
    return result


def predict_with_sliding_window(
    time_series: xr.DataArray | np.ndarray,
    window_size: int = 4,
    forecast_steps: int = 1,
    method: str = "mean",
    recursive: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Predict future values using sliding window average.

    This function takes a historical time series and predicts the next
    forecast_steps immediately following the input series.
    
    Parameters
    ----------
    time_series : xr.DataArray or np.ndarray
        Input historical time series data (1D array). This should be the
        training/observed data only, not including future values.
    window_size : int, default 4
        Size of the sliding window for averaging.
    forecast_steps : int, default 1
        Number of future steps to predict immediately following the input.
    method : str, default "mean"
        Prediction method:
        - "mean": Use mean of the last window_size values
        - "last": Use the last value
        - "trend": Use linear trend from last window_size values
    recursive : bool, default False
        If True and forecast_steps > 1, use recursive prediction (for trend method).
        For "mean" and "last" methods, recursive has no effect.
    
    Returns
    -------
    tuple
        (predictions, historical_smoothed, time_coords)
        - predictions: Array of predicted values (length = forecast_steps, dtype=float)
        - historical_smoothed: Array of smoothed historical values (same length as input)
        - time_coords: Optional time coordinates if input was xarray (None otherwise)
    """
    # Convert to numpy array if xarray
    if isinstance(time_series, xr.DataArray):
        data = time_series.values
        time_coords = time_series.time.values if 'time' in time_series.coords else None
    else:
        data = np.asarray(time_series, dtype=float)
        time_coords = None
    
    # Ensure float dtype
    data = np.asarray(data, dtype=float)
    
    # Compute smoothed historical values (same length as input)
    historical_smoothed = sliding_window_average(data, window_size, center=False)
    
    # Get the last window_size valid (non-NaN) values for prediction
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        raise ValueError("All values in time series are NaN")
    
    valid_data = data[valid_mask]
    
    # Extract last window_size valid values
    if len(valid_data) >= window_size:
        last_window = valid_data[-window_size:]
    else:
        # If not enough valid data, use all available
        last_window = valid_data
    
    if len(last_window) == 0:
        raise ValueError("No valid (non-NaN) values found in time series")
    
    # Generate predictions based on method
    predictions = np.zeros(forecast_steps, dtype=float)
    
    if method == "mean":
        # Simple mean of last window (constant prediction for all steps)
        pred_value = float(np.nanmean(last_window))
        predictions[:] = pred_value
    
    elif method == "last":
        # Use last value (constant prediction for all steps)
        pred_value = float(last_window[-1])
        predictions[:] = pred_value
    
    elif method == "trend":
        if recursive and forecast_steps > 1 and len(last_window) >= 2:
            # Recursive prediction: iteratively append predictions and refit
            current_window = last_window.copy()
            for step in range(forecast_steps):
                # Fit trend on current window
                x = np.arange(len(current_window))
                coeffs = np.polyfit(x, current_window, 1)
                # Predict next value
                next_x = len(current_window)
                next_value = float(np.polyval(coeffs, next_x))
                predictions[step] = next_value
                # Append to window for next iteration
                current_window = np.append(current_window, next_value)
                # Keep window size fixed
                if len(current_window) > window_size:
                    current_window = current_window[-window_size:]
        else:
            # Direct multi-step prediction: fit once and extrapolate
            if len(last_window) >= 2:
                x = np.arange(len(last_window))
                coeffs = np.polyfit(x, last_window, 1)
                # Extrapolate forward
                future_x = np.arange(len(last_window), len(last_window) + forecast_steps)
                predictions = np.polyval(coeffs, future_x).astype(float)
            else:
                # Single value: use it for all predictions
                predictions[:] = float(last_window[-1])
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'mean', 'last', 'trend'")
    
    return predictions, historical_smoothed, time_coords


def rolling_evaluation(
    train_data: xr.DataArray | np.ndarray,
    test_data: xr.DataArray | np.ndarray,
    window_size: int = 4,
    forecast_steps: int = 1,
    method: str = "mean",
    step_size: int = 1,
    recursive: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Perform rolling evaluation across the entire test set.
    
    For each position in the test set (at step_size intervals), predict the next
    forecast_steps and collect all predictions and true values for evaluation.
    """
    # Convert to numpy if needed
    if isinstance(train_data, xr.DataArray):
        train_values = train_data.values
    else:
        train_values = np.asarray(train_data, dtype=float)
    
    if isinstance(test_data, xr.DataArray):
        test_values = test_data.values
    else:
        test_values = np.asarray(test_data, dtype=float)
    
    all_predictions: list[float] = []
    all_true_values: list[float] = []
    
    # Start with initial training data
    current_train = train_values.copy()
    
    # Rolling evaluation: move through test set
    n_test = len(test_values)
    positions = list(range(0, n_test - forecast_steps + 1, step_size))
    if len(positions) == 0:
        positions = [0] if n_test >= forecast_steps else []
    
    for pos in positions:
        if pos + forecast_steps <= n_test:
            pred, _, _ = predict_with_sliding_window(
                current_train,
                window_size=window_size,
                forecast_steps=forecast_steps,
                method=method,
                recursive=recursive,
            )
            true_vals = test_values[pos:pos + forecast_steps]
            all_predictions.extend(pred)
            all_true_values.extend(true_vals)
            if step_size == 1 and pos + 1 <= n_test:
                current_train = np.append(current_train, test_values[pos])
            elif step_size > 1:
                end_pos = min(pos + step_size, n_test)
                current_train = np.append(current_train, test_values[pos:end_pos])
    
    all_predictions = np.array(all_predictions, dtype=float)
    all_true_values = np.array(all_true_values, dtype=float)
    metrics = evaluate_prediction(all_true_values, all_predictions)
    return all_predictions, all_true_values, metrics


def evaluate_prediction(
    true_values: np.ndarray | xr.DataArray,
    predicted_values: np.ndarray,
) -> dict:
    """
    Evaluate prediction accuracy using common metrics.
    
    Parameters
    ----------
    true_values : np.ndarray or xr.DataArray
        True/observed values. If xarray, will extract .values.
    predicted_values : np.ndarray
        Predicted values.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - mape: Mean Absolute Percentage Error
        - r2: R-squared coefficient
    """
    # Convert xarray to numpy if needed
    if isinstance(true_values, xr.DataArray):
        true_values = true_values.values
    
    # Ensure numpy arrays
    true_values = np.asarray(true_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)
    
    # Check length alignment
    if len(true_values) != len(predicted_values):
        # Truncate to shorter length with warning
        min_len = min(len(true_values), len(predicted_values))
        print(f"Warning: Length mismatch - true_values={len(true_values)}, "
              f"predicted_values={len(predicted_values)}. Truncating to {min_len}.")
        true_values = true_values[:min_len]
        predicted_values = predicted_values[:min_len]
    
    # Remove NaN values for evaluation
    mask = ~(np.isnan(true_values) | np.isnan(predicted_values))
    if not np.any(mask):
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
    
    y_true = true_values[mask]
    y_pred = predicted_values[mask]
    
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # MAPE (avoid division by zero)
    mape_mask = y_true != 0
    if np.any(mape_mask):
        mape = float(np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100)
    else:
        mape = np.nan
    
    # R-squared
    # Note: R² can be negative when SS_res > SS_tot, meaning the model performs
    # worse than simply predicting the mean of true values. This is common for:
    # 1. Simple baseline methods (mean/last/trend) that predict constant or near-constant values
    # 2. Short-horizon predictions where predictions have less variance than true values
    # 3. Systematic bias in predictions
    # For short-horizon prediction with simple methods, MAE and RMSE are more meaningful.
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else np.nan
    
    # Additional metric: Mean Bias Error (MBE) - indicates systematic bias
    mbe = float(np.mean(y_pred - y_true))
    
    # Additional metric: Coefficient of Variation of RMSE (CVRMSE)
    # Normalizes RMSE by the mean of true values (useful for comparing across scales)
    cvrmse = float((rmse / np.mean(y_true)) * 100) if np.mean(y_true) != 0 else np.nan
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "mbe": mbe,  # Mean Bias Error (positive = over-prediction, negative = under-prediction)
        "cvrmse": cvrmse,  # Coefficient of Variation of RMSE (%)
    }


def test_sliding_window_average():
    """Unit tests for sliding_window_average function."""
    print("Running unit tests for sliding_window_average...")
    
    # Test A: Basic trailing window
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sliding_window_average(data, window_size=2, center=False)
    expected = np.array([1.0, 1.5, 2.5, 3.5, 4.5])  # Rolling mean
    assert len(result) == len(data), f"Length mismatch: {len(result)} != {len(data)}"
    assert np.allclose(result, expected, rtol=1e-10), f"Result mismatch: {result} != {expected}"
    print("  ✓ Test A passed: Basic trailing window")
    
    # Test B: Centered window
    data = np.array([1.0, 2.0, 3.0, 4.0])
    result = sliding_window_average(data, window_size=2, center=True)
    assert len(result) == len(data), f"Length mismatch: {len(result)} != {len(data)}"
    print("  ✓ Test B passed: Centered window (length check)")
    
    # Test C: Data shorter than window
    data = np.array([1.0, 2.0])
    result = sliding_window_average(data, window_size=3, center=False)
    expected_mean = np.nanmean(data)
    assert len(result) == len(data), f"Length mismatch: {len(result)} != {len(data)}"
    assert np.allclose(result, expected_mean), f"Result mismatch: {result} != {expected_mean}"
    print("  ✓ Test C passed: Data shorter than window")
    
    # Test D: NaN handling
    data = np.array([1.0, np.nan, 3.0, 4.0])
    result = sliding_window_average(data, window_size=2, center=False)
    assert len(result) == len(data), f"Length mismatch: {len(result)} != {len(data)}"
    assert not np.isnan(result[0]), "First value should not be NaN"
    print("  ✓ Test D passed: NaN handling")
    
    print("All sliding_window_average tests passed!\n")

# Sundial code removed


def main():
    """
    Main function demonstrating time series prediction.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Time series prediction with Sundial generation (and optional baselines).")
    # Unified evaluation options
    parser.add_argument("--eval-mode", type=str, choices=["single", "multi"], default="single", help="Evaluation mode: single location or multi-location averaging")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sampled locations for multi evaluation")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for location sampling (multi mode)")
    parser.add_argument("--train-fraction", type=float, default=0.9, help="Fraction of time steps used for training (0-1)")
    parser.add_argument("--min-valid-fraction", type=float, default=0.0, help="Min non-NaN fraction over time for a cell (0 = include any non-NaN)")
    parser.add_argument("--variable", type=str, default="LAI", help="Variable name in dataset")
    parser.add_argument("--lat", type=float, default=None, help="Latitude to evaluate (default: center)")
    parser.add_argument("--lon", type=float, default=None, help="Longitude to evaluate (default: center)")
    # Minimal Sundial args
    parser.add_argument("--forecast-length", type=int, default=1, help="Forecast horizon for Sundial (use 1 for rolling one-step)")
    parser.add_argument("--sundial-num-samples", type=int, default=20, help="Number of generated samples for Sundial")
    parser.add_argument("--lookback-length", type=int, default=2880, help="Lookback length for Sundial context (<=2880)")
    args, unknown = parser.parse_known_args()

    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("Warning: matplotlib not available, skipping plots")
    
    # Load data
    print("Loading LAI dataset...")
    ds = load_all_data()
    # Sundial helpers
    _SUNDIAL_MODEL = None
    def _get_sundial_model() -> "AutoModelForCausalLM":
        nonlocal _SUNDIAL_MODEL
        if _SUNDIAL_MODEL is None:
            if not _HAS_TRANSFORMERS:
                raise ImportError("transformers/torch not available. Install to use Sundial.")
            _SUNDIAL_MODEL = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True)
            _SUNDIAL_MODEL.eval()
            if torch.cuda.is_available():
                _SUNDIAL_MODEL.to("cuda")
        return _SUNDIAL_MODEL
    def sundial_generate_mean(history_values: np.ndarray, forecast_length: int, num_samples: int, lookback_length: int) -> np.ndarray:
        model = _get_sundial_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vals = np.asarray(history_values, dtype=np.float32)
        vals = vals[~np.isnan(vals)]
        if vals.ndim != 1 or vals.size == 0:
            raise ValueError("Invalid history for Sundial: no valid (non-NaN) values.")
        # Use available valid length if smaller than requested lookback_length
        actual_lookback = min(int(lookback_length), vals.size)
        if actual_lookback == 0:
            raise ValueError("Invalid history for Sundial: no valid values after filtering NaNs.")
        ctx = vals[-actual_lookback:]
        seqs = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(device)  # [1, L]
        with torch.no_grad():
            try:
                out = model.generate(
                    seqs,
                    max_new_tokens=int(forecast_length),
                    num_samples=int(num_samples)
                )
            except TypeError as e:
                # Fallback if use_cache is not supported by this model shim
                if "use_cache" in str(e) or "unexpected keyword" in str(e):
                    out = model.generate(
                        seqs,
                        max_new_tokens=int(forecast_length),
                        num_samples=int(num_samples),
                    )
                else:
                    raise
        out_cpu = out.detach().to("cpu")
        # Normalize shapes to [S, F]
        if out_cpu.ndim == 1:
            out_cpu = out_cpu.view(1, -1)
        elif out_cpu.ndim == 2:
            # assume [S, F] or [1, F]
            if out_cpu.shape[0] == 1 and num_samples > 1:
                out_cpu = out_cpu.repeat(num_samples, 1)
        else:
            # Flatten any leading batch/sample dims to [*, F]
            out_cpu = out_cpu.view(-1, out_cpu.shape[-1])
        samples = out_cpu.numpy().astype(np.float32)
        # Keep at most requested samples
        if samples.shape[0] > int(num_samples):
            samples = samples[:int(num_samples), :]
        mean_pred = samples.mean(axis=0)
        return mean_pred.astype(np.float32)
    # Validate variable
    if args.variable not in ds.data_vars:
        raise ValueError(f"Variable '{args.variable}' not found. Available: {list(ds.data_vars)}")
    if args.eval_mode == "single":
        sample_lat, sample_lon = select_single_location(ds, args.lat, args.lon)
        print(f"\nSelected location: lat={sample_lat:.4f}, lon={sample_lon:.4f}")
        print("Evaluation mode: single")
        series_da = get_time_series(ds, float(sample_lat), float(sample_lon), variable=args.variable)
        # Split train/test by fraction
        values = series_da.values.astype(float)
        n_total = len(values)
        n_train = max(1, int(n_total * float(args.train_fraction)))
        train_vals = values[:n_train]
        test_vals = values[n_train:]
        preds_all = []
        if not _HAS_TRANSFORMERS:
            print("transformers/torch not available; skipping Sundial.")
            return
        if int(args.forecast_length) == 1:
            # Rolling one-step evaluation
            history = train_vals.astype(float)
            for pos in range(len(test_vals)):
                try:
                    mean_step = sundial_generate_mean(history, 1, int(args.sundial_num_samples), int(args.lookback_length))
                    preds_all.append(float(mean_step[0]))
                    history = np.append(history, float(test_vals[pos]))
                except Exception as e:
                    print(f"Sundial generation failed at step {pos+1}: {e}")
                    break
            preds_np = np.asarray(preds_all, dtype=float)
            trues_np = test_vals[:len(preds_np)].astype(float)
        else:
            # Rolling multi-step evaluation: predict forecast_length steps ahead, then roll forward by forecast_length
            history = train_vals.astype(float)
            forecast_len = int(args.forecast_length)
            pos = 0
            while pos + forecast_len <= len(test_vals):
                try:
                    # Predict forecast_length steps ahead
                    preds_step = sundial_generate_mean(history, forecast_len, int(args.sundial_num_samples), int(args.lookback_length))
                    preds_all.extend(preds_step.tolist())
                    # Update history with true values (not predictions) for next iteration
                    history = np.append(history, test_vals[pos:pos + forecast_len].astype(float))
                    pos += forecast_len
                except Exception as e:
                    print(f"Sundial generation failed at position {pos+1}: {e}")
                    break
            preds_np = np.asarray(preds_all, dtype=float)
            trues_np = test_vals[:len(preds_np)].astype(float)
        if trues_np.size > 0 and preds_np.size > 0:
            mae = float(np.mean(np.abs(trues_np - preds_np)))
            rmse = float(np.sqrt(np.mean((trues_np - preds_np) ** 2)))
            # MAPE
            mape_mask = trues_np != 0
            mape = float(np.mean(np.abs((trues_np[mape_mask] - preds_np[mape_mask]) / trues_np[mape_mask])) * 100) if np.any(mape_mask) else float("nan")
            # R²
            ss_res = float(np.sum((trues_np - preds_np) ** 2))
            ss_tot = float(np.sum((trues_np - np.mean(trues_np)) ** 2))
            r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else float("nan")
            # MBE
            mbe = float(np.mean(preds_np - trues_np))
            # CVRMSE
            cvrmse = float((rmse / np.mean(trues_np)) * 100) if np.mean(trues_np) != 0 else float("nan")
            print(f"  Number of predictions: {len(preds_np)}")
            print(f"  MAE:   {mae:.4f}")
            print(f"  RMSE:  {rmse:.4f}")
            print(f"  MAPE:  {mape:.2f}%")
            print(f"  R²:    {r2:.4f}")
            print(f"  MBE:   {mbe:.4f}")
            print(f"  CVRMSE: {cvrmse:.2f}%")
    else:
        sampled_locations = sample_locations(ds, args.variable, float(args.min_valid_fraction), int(args.num_samples), int(args.sample_seed))
        print("Evaluation mode: multi")
        print(f"Locations: {len(sampled_locations)}")
        pretty_print_sampled_locations(sampled_locations, max_show=5)
        if not _HAS_TRANSFORMERS:
            print("transformers/torch not available; skipping Sundial.")
            return
        maes: list[float] = []
        rmses: list[float] = []
        mapes: list[float] = []
        r2s: list[float] = []
        mbes: list[float] = []
        cvrmses: list[float] = []
        n_preds_total = 0
        # Pool all predictions and true values for proper R² calculation
        all_preds_pooled: list[float] = []
        all_trues_pooled: list[float] = []
        for (lt, ln) in tqdm(sampled_locations, desc="Evaluating locations", leave=True):
            series_da = get_time_series(ds, float(lt), float(ln), variable=args.variable)
            values = series_da.values.astype(float)
            n_total = len(values)
            n_train = max(1, int(n_total * float(args.train_fraction)))
            train_vals = values[:n_train]
            test_vals = values[n_train:]
            preds_list: list[float] = []
            if int(args.forecast_length) == 1:
                history = train_vals.astype(float)
                for pos in range(len(test_vals)):
                    try:
                        mean_step = sundial_generate_mean(history, 1, int(args.sundial_num_samples), int(args.lookback_length))
                        preds_list.append(float(mean_step[0]))
                        history = np.append(history, float(test_vals[pos]))
                    except Exception:
                        # If generation fails, stop for this location but keep partial results
                        break
                preds_np = np.asarray(preds_list, dtype=float)
                trues_np = test_vals[:len(preds_np)].astype(float)
            else:
                # Rolling multi-step evaluation: predict forecast_length steps ahead, then roll forward by forecast_length
                history = train_vals.astype(float)
                forecast_len = int(args.forecast_length)
                pos = 0
                while pos + forecast_len <= len(test_vals):
                    try:
                        # Predict forecast_length steps ahead
                        preds_step = sundial_generate_mean(history, forecast_len, int(args.sundial_num_samples), int(args.lookback_length))
                        preds_list.extend(preds_step.tolist())
                        # Update history with true values (not predictions) for next iteration
                        history = np.append(history, test_vals[pos:pos + forecast_len].astype(float))
                        pos += forecast_len
                    except Exception:
                        # If generation fails, stop for this location but keep partial results
                        break
                preds_np = np.asarray(preds_list, dtype=float)
                trues_np = test_vals[:len(preds_np)].astype(float)
            if trues_np.size == 0 or preds_np.size == 0:
                continue
        
            mae_i = float(np.mean(np.abs(trues_np - preds_np)))
            rmse_i = float(np.sqrt(np.mean((trues_np - preds_np) ** 2)))
            # MAPE
            mape_mask = trues_np != 0
            mape_i = float(np.mean(np.abs((trues_np[mape_mask] - preds_np[mape_mask]) / trues_np[mape_mask])) * 100) if np.any(mape_mask) else float("nan")
            # R²
            ss_res = float(np.sum((trues_np - preds_np) ** 2))
            ss_tot = float(np.sum((trues_np - np.mean(trues_np)) ** 2))
            r2_i = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else float("nan")
            # MBE
            mbe_i = float(np.mean(preds_np - trues_np))
            # CVRMSE
            cvrmse_i = float((rmse_i / np.mean(trues_np)) * 100) if np.mean(trues_np) != 0 else float("nan")
            maes.append(mae_i)
            rmses.append(rmse_i)
            n_preds_total += len(preds_np)
            # Pool predictions and true values for proper R² calculation
            all_preds_pooled.extend(preds_np.tolist())
            all_trues_pooled.extend(trues_np.tolist())
            if not np.isnan(mape_i):
                mapes.append(mape_i)
            if not np.isnan(r2_i):
                r2s.append(r2_i)
            mbes.append(mbe_i)
            if not np.isnan(cvrmse_i):
                cvrmses.append(cvrmse_i)
        mae_mean = float(np.mean(maes)) if len(maes) > 0 else float("nan")
        mae_std = float(np.std(maes)) if len(maes) > 0 else float("nan")
        rmse_mean = float(np.mean(rmses)) if len(rmses) > 0 else float("nan")
        rmse_std = float(np.std(rmses)) if len(rmses) > 0 else float("nan")
        mape_mean = float(np.mean(mapes)) if len(mapes) > 0 else float("nan")
        mape_std = float(np.std(mapes)) if len(mapes) > 0 else float("nan")
        # Compute R² on pooled data (correct approach for multi-location evaluation)
        if len(all_preds_pooled) > 0 and len(all_trues_pooled) > 0:
            preds_pooled_np = np.asarray(all_preds_pooled, dtype=float)
            trues_pooled_np = np.asarray(all_trues_pooled, dtype=float)
            ss_res_pooled = float(np.sum((trues_pooled_np - preds_pooled_np) ** 2))
            ss_tot_pooled = float(np.sum((trues_pooled_np - np.mean(trues_pooled_np)) ** 2))
            r2_pooled = float(1 - (ss_res_pooled / ss_tot_pooled)) if ss_tot_pooled != 0 else float("nan")
            # Also compute per-location R² for std (for reporting)
            r2_mean = r2_pooled  # Use pooled R² as the mean
            r2_std = float(np.std(r2s)) if len(r2s) > 0 else float("nan")
        else:
            r2_mean = float(np.mean(r2s)) if len(r2s) > 0 else float("nan")
            r2_std = float(np.std(r2s)) if len(r2s) > 0 else float("nan")
        mbe_mean = float(np.mean(mbes)) if len(mbes) > 0 else float("nan")
        mbe_std = float(np.std(mbes)) if len(mbes) > 0 else float("nan")
        cvrmse_mean = float(np.mean(cvrmses)) if len(cvrmses) > 0 else float("nan")
        cvrmse_std = float(np.std(cvrmses)) if len(cvrmses) > 0 else float("nan")
        print(f"  Number of predictions: {n_preds_total}")
        print(f"  MAE:   {mae_mean:.4f} ± {mae_std:.4f}")
        print(f"  RMSE:  {rmse_mean:.4f} ± {rmse_std:.4f}")
        print(f"  MAPE:  {mape_mean:.2f}% ± {mape_std:.2f}%")
        print(f"  R²:    {r2_mean:.4f} ± {r2_std:.4f}")
        print(f"  MBE:   {mbe_mean:.4f} ± {mbe_std:.4f}")
        print(f"  CVRMSE: {cvrmse_mean:.2f}% ± {cvrmse_std:.2f}%")

if __name__ == "__main__":
    main()