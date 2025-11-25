"""
Time Series Prediction using Sliding Window Average

This module implements a simple time series prediction method using
sliding window average for LAI (Leaf Area Index) data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import xarray as xr
from tqdm import tqdm
from util import rolling_eval, select_single_location, sample_locations, pretty_print_sampled_locations

from load_all_data import load_all_data, get_time_series


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



def main():
    """
    Main function demonstrating time series prediction.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Time series prediction with baselines.")
    # Unified evaluation options
    parser.add_argument("--eval-mode", type=str, choices=["single", "multi"], default="single", help="Evaluation mode: single location or multi-location averaging")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sampled locations for multi evaluation")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for location sampling (multi mode)")
    parser.add_argument("--methods", type=str, nargs="+", default=["mean", "last", "trend"], help="Prediction methods to evaluate")
    parser.add_argument("--train-fraction", type=float, default=0.9, help="Fraction of time steps used for training (0-1)")
    parser.add_argument("--min-valid-fraction", type=float, default=0.0, help="Min non-NaN fraction over time for a cell (0 = include any non-NaN)")
    parser.add_argument("--variable", type=str, default="LAI", help="Variable name in dataset")
    parser.add_argument("--lat", type=float, default=None, help="Latitude to evaluate (default: center)")
    parser.add_argument("--lon", type=float, default=None, help="Longitude to evaluate (default: center)")
    # Legacy/demo options
    parser.add_argument("--window-size", type=int, default=8, help="Window size for sliding window")
    parser.add_argument("--baseline-forecast-steps", type=int, default=1, help="Baseline forecast steps for fair comparison")
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
    # Global ARIMA statistics tracker
    arima_global_stats = {"success": 0, "fail": 0, "errors": []}
    sarima_global_stats = {"success": 0, "fail": 0, "errors": []}
    # Multi-step rolling evaluation function
    def rolling_eval_multi_step(series: np.ndarray, method: str, window_size: int, train_fraction: float, forecast_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-step rolling evaluation that properly handles forecast_steps > 1.
        Returns (all_predictions, all_true_values) where each prediction corresponds
        to a forecast step ahead.
        """
        values = pd.Series(series.astype(float)).interpolate(method="linear", limit_direction="both").values.astype(float)
        n_total = len(values)
        n_train = max(int(window_size) + 1, int(n_total * float(train_fraction)))
        n_test = n_total - n_train
        if n_test <= 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        
        # Adjust forecast_steps if test set is too short
        actual_forecast_steps = min(forecast_steps, n_test)
        
        all_predictions = []
        all_true_values = []
        
        # Rolling evaluation: at each position, predict forecast_steps ahead
        # We can evaluate up to n_test - actual_forecast_steps + 1 positions
        max_positions = max(0, n_test - actual_forecast_steps + 1)
        for i in range(max_positions):
            end_idx = n_train + i
            start_idx = end_idx - int(window_size)
            if start_idx < 0:
                window_vals = values[max(0, start_idx):end_idx]
                if len(window_vals) < int(window_size):
                    window_vals = np.pad(window_vals, (int(window_size) - len(window_vals), 0), mode="edge")
            else:
                window_vals = values[start_idx:end_idx]
            
            # Get predictions for actual_forecast_steps ahead
            if method.lower() == "mean":
                pred_value = float(np.nanmean(window_vals))
                preds = np.full(actual_forecast_steps, pred_value, dtype=float)
            elif method.lower() == "last":
                pred_value = float(window_vals[-1])
                preds = np.full(actual_forecast_steps, pred_value, dtype=float)
            elif method.lower() == "trend":
                if len(window_vals) >= 2:
                    x = np.arange(len(window_vals))
                    coeffs = np.polyfit(x, window_vals, 1)
                    future_x = np.arange(len(window_vals), len(window_vals) + actual_forecast_steps)
                    preds = np.polyval(coeffs, future_x).astype(float)
                else:
                    pred_value = float(window_vals[-1])
                    preds = np.full(actual_forecast_steps, pred_value, dtype=float)
            elif method.lower() == "arima":
                try:
                    model = ARIMA(window_vals, order=(2,1,0))
                    fitted = model.fit(method='statespace')
                    preds = fitted.forecast(steps=actual_forecast_steps).astype(float)
                    arima_global_stats["success"] += 1
                except Exception as e:
                    arima_global_stats["fail"] += 1
                    if len(arima_global_stats["errors"]) < 10:
                        arima_global_stats["errors"].append(str(e))
                    pred_value = float(window_vals[-1])
                    preds = np.full(actual_forecast_steps, pred_value, dtype=float)
            elif method.lower() == "sarima":
                SARIMA_ORDER = (1, 1, 0)
                SEASONAL_ORDER = (0, 1, 1, 46)
                SEASONAL_PERIOD = SEASONAL_ORDER[3]
                MIN_WINDOW_SIZE = max(20, 2 * SEASONAL_PERIOD)
                try:
                    if len(window_vals) < MIN_WINDOW_SIZE:
                        raise ValueError(f"Window too small: {len(window_vals)} < {MIN_WINDOW_SIZE}")
                    model = SARIMAX(
                        window_vals,
                        order=SARIMA_ORDER,
                        seasonal_order=SEASONAL_ORDER,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fitted = model.fit(maxiter=15, disp=False)
                    preds = fitted.forecast(steps=actual_forecast_steps).astype(float)
                    sarima_global_stats["success"] += 1
                except Exception as e:
                    try:
                        simple_model = SARIMAX(
                            window_vals,
                            order=(0, 1, 1),
                            seasonal_order=(0, 1, 1, SEASONAL_PERIOD),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        simple_fitted = simple_model.fit(maxiter=10, disp=False)
                        preds = simple_fitted.forecast(steps=actual_forecast_steps).astype(float)
                        sarima_global_stats["success"] += 1
                    except Exception as e2:
                        sarima_global_stats["fail"] += 1
                        if len(sarima_global_stats["errors"]) < 10:
                            sarima_global_stats["errors"].append(str(e2))
                        pred_value = float(window_vals[-1])
                        preds = np.full(actual_forecast_steps, pred_value, dtype=float)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Get true values for actual_forecast_steps ahead
            true_vals = values[end_idx:end_idx + actual_forecast_steps].astype(float)
            
            all_predictions.extend(preds)
            all_true_values.extend(true_vals)
        
        return np.array(all_predictions, dtype=float), np.array(all_true_values, dtype=float)
    
    # Unified evaluation path using multi-step rolling evaluation
    def make_predict_fn(method: str):
        # For backward compatibility, but we'll use rolling_eval_multi_step instead
        m = method.lower()
        if m == "mean":
            return lambda win: float(np.mean(win))
        elif m == "last":
            return lambda win: float(win[-1])
        elif m == "trend":
            def predict_trend(win: np.ndarray) -> float:
                W = win.shape[0]
                x = np.arange(W, dtype=np.float64)
                n = float(W)
                Sx = float(np.sum(x))
                Sxx = float(np.sum(x * x))
                Sy = float(np.sum(win))
                Sxy = float(np.sum(win * x))
                den = (n * Sxx - Sx * Sx)
                den = den if den != 0.0 else 1e-12
                slope = (n * Sxy - Sx * Sy) / den
                intercept = (Sy - slope * Sx) / n
                return float(intercept + slope * float(W))
            return predict_trend
        elif m == "arima":
            # ARIMA(2,1,0) is a simple reasonable default
            def predict_arima(win: np.ndarray) -> float:
                try:
                    model = ARIMA(win, order=(2,1,0))
                    # Note: disp parameter removed in newer statsmodels versions
                    fitted = model.fit(method='statespace')
                    pred = fitted.forecast(steps=args.baseline_forecast_steps)
                    arima_global_stats["success"] += 1
                    return float(pred[0])
                except Exception as e:
                    # fallback if ARIMA fails
                    arima_global_stats["fail"] += 1
                    if len(arima_global_stats["errors"]) < 10:  # Store first 10 errors
                        arima_global_stats["errors"].append(str(e))
                    return float(win[-1])
            return predict_arima
        elif m == "sarima":
            # SARIMA Implementation using SARIMAX (Seasonal ARIMA)
            # LAI Periodicity (m): 365 days / 8-day cadence = ~46
            # Use simpler orders for speed: (1,1,0) x (0,1,1,46) - simpler seasonal component
            SARIMA_ORDER = (1, 1, 0)  # Simpler non-seasonal part
            SEASONAL_ORDER = (0, 1, 1, 46)  # Simpler seasonal: only MA term
            SEASONAL_PERIOD = SEASONAL_ORDER[3]  # Extract seasonal period (46)
            MIN_WINDOW_SIZE = max(20, 2 * SEASONAL_PERIOD)  # Need at least 2 seasonal cycles
            
            def predict_sarima(win: np.ndarray) -> float:
                try:
                    # Check if window is large enough for seasonal model
                    if len(win) < MIN_WINDOW_SIZE:
                        raise ValueError(f"Window too small: {len(win)} < {MIN_WINDOW_SIZE} (need at least 2 seasonal periods of {SEASONAL_PERIOD})")
                    
                    # Use SARIMAX with simplified orders for faster fitting
                    model = SARIMAX(
                        win, 
                        order=SARIMA_ORDER, 
                        seasonal_order=SEASONAL_ORDER,
                        enforce_stationarity=False,  # Allow non-stationarity for speed
                        enforce_invertibility=False  # Allow non-invertibility for speed
                    )
                    
                    # Fit with speed optimizations: limit iterations aggressively
                    fitted = model.fit(
                        maxiter=15,  # Reduced iterations for speed
                        disp=False
                    )
                    
                    # Predict next step(s)
                    pred = fitted.forecast(steps=args.baseline_forecast_steps)
                    sarima_global_stats["success"] += 1
                    return float(pred[0])
                    
                except Exception as e:
                    # If optimization fails, try even simpler model as fallback
                    try:
                        # Fallback to very simple SARIMA(0,1,1)x(0,1,1,46) - fastest option
                        simple_model = SARIMAX(
                            win,
                            order=(0, 1, 1),
                            seasonal_order=(0, 1, 1, SEASONAL_PERIOD),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        simple_fitted = simple_model.fit(maxiter=10, disp=False)
                        pred = simple_fitted.forecast(steps=args.baseline_forecast_steps)
                        sarima_global_stats["success"] += 1
                        return float(pred[0])
                    except Exception as e2:
                        # Final fallback: use last value
                        sarima_global_stats["fail"] += 1
                        if len(sarima_global_stats["errors"]) < 10:
                            sarima_global_stats["errors"].append(str(e2))
                        return float(win[-1])
                    
            return predict_sarima
        else:
            raise ValueError(f"Unknown method: {method}")
    # Validate variable
    if args.variable not in ds.data_vars:
        raise ValueError(f"Variable '{args.variable}' not found. Available: {list(ds.data_vars)}")
    if args.eval_mode == "single":
        sample_lat, sample_lon = select_single_location(ds, args.lat, args.lon)
        print(f"\nSelected location: lat={sample_lat:.4f}, lon={sample_lon:.4f}")
        print("Evaluation mode: single")
        print(f"Forecast steps: {args.baseline_forecast_steps}")
        series_da = get_time_series(ds, float(sample_lat), float(sample_lon), variable=args.variable)
        for method in args.methods:
            if args.baseline_forecast_steps > 1:
                preds, trues = rolling_eval_multi_step(
                    series_da.values.astype(float), 
                    method, 
                    int(args.window_size), 
                    float(args.train_fraction),
                    int(args.baseline_forecast_steps)
                )
            else:
                predict_fn = make_predict_fn(method)
                preds, trues = rolling_eval(series_da.values.astype(float), predict_fn, int(args.window_size), float(args.train_fraction))
            if trues.size == 0:
                print(f"{method.upper()}: No predictions")
                continue
            mae = float(np.mean(np.abs(trues - preds)))
            rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))
            # MAPE
            mape_mask = trues != 0
            mape = float(np.mean(np.abs((trues[mape_mask] - preds[mape_mask]) / trues[mape_mask])) * 100) if np.any(mape_mask) else float("nan")
            # R²
            ss_res = float(np.sum((trues - preds) ** 2))
            ss_tot = float(np.sum((trues - np.mean(trues)) ** 2))
            r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else float("nan")
            # MBE
            mbe = float(np.mean(preds - trues))
            # CVRMSE
            cvrmse = float((rmse / np.mean(trues)) * 100) if np.mean(trues) != 0 else float("nan")
            print(f"{method.upper()}:")
            print(f"  Number of predictions: {len(preds)}")
            print(f"  MAE:   {mae:.4f}")
            print(f"  RMSE:  {rmse:.4f}")
            print(f"  MAPE:  {mape:.2f}%")
            print(f"  R²:    {r2:.4f}")
            print(f"  MBE:   {mbe:.4f}")
            print(f"  CVRMSE: {cvrmse:.2f}%")
        return
    else:
        sampled_locations = sample_locations(ds, args.variable, float(args.min_valid_fraction), int(args.num_samples), int(args.sample_seed))
        print("Evaluation mode: multi")
        print(f"Locations: {len(sampled_locations)}")
        print(f"Forecast steps: {args.baseline_forecast_steps}")
        pretty_print_sampled_locations(sampled_locations, max_show=5)
        per_method_maes = {m: [] for m in args.methods}
        per_method_rmses = {m: [] for m in args.methods}
        per_method_mapes = {m: [] for m in args.methods}
        per_method_r2s = {m: [] for m in args.methods}
        per_method_mbes = {m: [] for m in args.methods}
        per_method_cvrmses = {m: [] for m in args.methods}
        per_method_n_preds = {m: [] for m in args.methods}
        for (lt, ln) in tqdm(sampled_locations, desc="Evaluating locations", leave=True):
            series_da = get_time_series(ds, float(lt), float(ln), variable=args.variable)
            for method in args.methods:
                if args.baseline_forecast_steps > 1:
                    preds, trues = rolling_eval_multi_step(
                        series_da.values.astype(float), 
                        method, 
                        int(args.window_size), 
                        float(args.train_fraction),
                        int(args.baseline_forecast_steps)
                    )
                else:
                    predict_fn = make_predict_fn(method)
                    preds, trues = rolling_eval(series_da.values.astype(float), predict_fn, int(args.window_size), float(args.train_fraction))
                if trues.size == 0:
                    continue
                mae_i = float(np.mean(np.abs(trues - preds)))
                rmse_i = float(np.sqrt(np.mean((trues - preds) ** 2)))
                # MAPE
                mape_mask = trues != 0
                mape_i = float(np.mean(np.abs((trues[mape_mask] - preds[mape_mask]) / trues[mape_mask])) * 100) if np.any(mape_mask) else float("nan")
                # R²
                ss_res = float(np.sum((trues - preds) ** 2))
                ss_tot = float(np.sum((trues - np.mean(trues)) ** 2))
                r2_i = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else float("nan")
                # MBE
                mbe_i = float(np.mean(preds - trues))
                # CVRMSE
                cvrmse_i = float((rmse_i / np.mean(trues)) * 100) if np.mean(trues) != 0 else float("nan")
                per_method_maes[method].append(mae_i)
                per_method_rmses[method].append(rmse_i)
                if not np.isnan(mape_i):
                    per_method_mapes[method].append(mape_i)
                if not np.isnan(r2_i):
                    per_method_r2s[method].append(r2_i)
                per_method_mbes[method].append(mbe_i)
                if not np.isnan(cvrmse_i):
                    per_method_cvrmses[method].append(cvrmse_i)
                per_method_n_preds[method].append(len(preds))
        for method in args.methods:
            maes = per_method_maes[method]
            rmses = per_method_rmses[method]
            mapes = per_method_mapes[method]
            r2s = per_method_r2s[method]
            mbes = per_method_mbes[method]
            cvrmses = per_method_cvrmses[method]
            n_preds = per_method_n_preds[method]
            mae_mean = float(np.mean(maes)) if len(maes) > 0 else float("nan")
            mae_std = float(np.std(maes)) if len(maes) > 0 else float("nan")
            rmse_mean = float(np.mean(rmses)) if len(rmses) > 0 else float("nan")
            rmse_std = float(np.std(rmses)) if len(rmses) > 0 else float("nan")
            mape_mean = float(np.mean(mapes)) if len(mapes) > 0 else float("nan")
            mape_std = float(np.std(mapes)) if len(mapes) > 0 else float("nan")
            r2_mean = float(np.mean(r2s)) if len(r2s) > 0 else float("nan")
            r2_std = float(np.std(r2s)) if len(r2s) > 0 else float("nan")
            mbe_mean = float(np.mean(mbes)) if len(mbes) > 0 else float("nan")
            mbe_std = float(np.std(mbes)) if len(mbes) > 0 else float("nan")
            cvrmse_mean = float(np.mean(cvrmses)) if len(cvrmses) > 0 else float("nan")
            cvrmse_std = float(np.std(cvrmses)) if len(cvrmses) > 0 else float("nan")
            n_preds_total = int(np.sum(n_preds)) if len(n_preds) > 0 else 0
            print(f"{method.upper()}:")
            print(f"  Number of predictions: {n_preds_total}")
            print(f"  MAE:   {mae_mean:.4f} ± {mae_std:.4f}")
            print(f"  RMSE:  {rmse_mean:.4f} ± {rmse_std:.4f}")
            print(f"  MAPE:  {mape_mean:.2f}% ± {mape_std:.2f}%")
            print(f"  R²:    {r2_mean:.4f} ± {r2_std:.4f}")
            print(f"  MBE:   {mbe_mean:.4f} ± {mbe_std:.4f}")
            print(f"  CVRMSE: {cvrmse_mean:.2f}% ± {cvrmse_std:.2f}%")
        # Print ARIMA debugging info if ARIMA was used
        if "arima" in [m.lower() for m in args.methods]:
            total_calls = arima_global_stats["success"] + arima_global_stats["fail"]
            print(f"\n[ARIMA Debug] Total calls: {total_calls}")
            print(f"  Success: {arima_global_stats['success']} ({100.0 * arima_global_stats['success'] / max(1, total_calls):.1f}%)")
            print(f"  Failed: {arima_global_stats['fail']} ({100.0 * arima_global_stats['fail'] / max(1, total_calls):.1f}%)")
            if arima_global_stats["errors"]:
                print(f"  Sample errors (first {len(arima_global_stats['errors'])}):")
                for i, err in enumerate(arima_global_stats["errors"][:5], 1):
                    print(f"    {i}. {err[:200]}")  # Truncate long errors
        if "sarima" in [m.lower() for m in args.methods]:
            total_calls = sarima_global_stats["success"] + sarima_global_stats["fail"]
            print(f"\n[SARIMA Debug] Total calls: {total_calls}")
            print(f"  Success: {sarima_global_stats['success']} ({100.0 * sarima_global_stats['success'] / max(1, total_calls):.1f}%)")
            print(f"  Failed: {sarima_global_stats['fail']} ({100.0 * sarima_global_stats['fail'] / max(1, total_calls):.1f}%)")
            if sarima_global_stats["errors"]:
                print(f"  Sample errors (first {len(sarima_global_stats['errors'])}):")
                for i, err in enumerate(sarima_global_stats["errors"][:5], 1):
                    print(f"    {i}. {err[:200]}")  # Truncate long errors
        return
    import random
    random.seed(42)
    # Select a location (center of the domain)
    sample_lat = float(ds.lat[len(ds.lat) // 2])
    sample_lon = float(ds.lon[len(ds.lon) // 2])
    print(f"\nSelected location: lat={sample_lat:.2f}, lon={sample_lon:.2f}")
    # sample_lat = random.choice(ds.lat.values.tolist())
    # sample_lon = random.choice(ds.lon.values.tolist())
    # print(f"\nSelected location: lat={sample_lat:.2f}, lon={sample_lon:.2f}")
    
    # Extract time series
    print("Extracting time series...")
    time_series = get_time_series(ds, sample_lat, sample_lon)
    
    # Split into train and test (use last 10% for testing)
    n_total = len(time_series)
    n_test = max(1, int(n_total * 0.1))
    n_train = n_total - n_test
    
    train_data = time_series[:n_train]
    test_data = time_series[n_train:]
    
    print(f"\nData split:")
    print(f"  Training: {n_train} time steps")
    print(f"  Testing: {n_test} time steps")
    
    # Prediction parameters
    window_size = args.window_size
    # For fair comparison, allow configuring baseline forecast horizon
    forecast_steps = min(max(1, args.baseline_forecast_steps), n_test)
    # Step size for rolling evaluation (1 = every position, larger = fewer evaluations)
    step_size = 1
    
    print(f"\nPrediction parameters:")
    print(f"  Window size: {window_size}")
    print(f"  Forecast steps: {forecast_steps} (short-horizon prediction)")
    print(f"  Step size: {step_size} (rolling evaluation across entire test set)")
    print(f"  Note: Rolling evaluation - predict next {forecast_steps} steps at each position")
    
    # Make predictions using different methods with rolling evaluation
    methods = ["mean", "last", "trend"]
    results = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Method: {method}")
        print(f"{'='*80}")
        
        # Perform rolling evaluation across entire test set
        all_predictions, all_true_values, metrics = rolling_evaluation(
            train_data,
            test_data,
            window_size=window_size,
            forecast_steps=forecast_steps,
            method=method,
            step_size=step_size,
            recursive=False,
        )
        
        # Also get initial prediction and smoothed history for plotting
        initial_predictions, historical_smoothed, time_coords = predict_with_sliding_window(
            train_data,
            window_size=window_size,
            forecast_steps=forecast_steps,
            method=method,
            recursive=False,
        )
        
        results[method] = {
            "predictions": initial_predictions,  # For plotting first prediction
            "all_predictions": all_predictions,  # All rolling predictions
            "all_true_values": all_true_values,  # All true values
            "smoothed": historical_smoothed,
            "metrics": metrics,
            "time_coords": time_coords,
            "forecast_steps": forecast_steps,  # Store for plotting
        }
        
        print(f"  Number of predictions: {len(all_predictions)}")
        print(f"  MAE:   {metrics['mae']:.4f}")
        print(f"  RMSE:  {metrics['rmse']:.4f}")
        print(f"  MAPE:  {metrics['mape']:.2f}%")
        print(f"  R²:    {metrics['r2']:.4f}")
        print(f"  MBE:   {metrics['mbe']:.4f}")
        print(f"  CVRMSE: {metrics['cvrmse']:.2f}%")
    
    # Sundial removed – baselines only
    
    # Plot results (if matplotlib is available)
    if HAS_MATPLOTLIB:
        print(f"\n{'='*80}")
        print("Generating plots...")
        print(f"{'='*80}")
        
        meth_list = list(results.keys())
        fig, axes = plt.subplots(len(meth_list), 1, figsize=(12, 4 * len(meth_list)))
        if len(meth_list) == 1:
            axes = [axes]
        
        time_train = time_series.time.values[:n_train]
        time_test = time_series.time.values[n_train:]
        
        for idx, method in enumerate(meth_list):
            ax = axes[idx]
            
            # Plot original training data
            ax.plot(time_train, train_data.values, 'b-', alpha=0.5, label='Training data', linewidth=1)
            
            # Plot smoothed historical (training only)
            ax.plot(time_train, results[method]["smoothed"], 
                   'b--', alpha=0.7, label='Smoothed (training)', linewidth=1.5)
            
            # Plot all true test values (entire test set)
            ax.plot(time_test, test_data.values, 'g-', alpha=0.3, 
                   label='True test values (all)', linewidth=1)
            
            # Plot rolling predictions and evaluated true values
            # Note: all_predictions and all_true_values have overlapping values because
            # we predict forecast_steps at each position, so each test point appears
            # multiple times in the evaluation
            n_pred = len(results[method]["all_predictions"]) if "all_predictions" in results[method] else 0
            if n_pred > 0:
                # For visualization, show all predictions (with overlap) and per-time means
                forecast_steps = results[method].get("forecast_steps", 1)  # Get from stored value
                n_positions = max(0, len(test_data) - forecast_steps + 1)
                # Create time indices for predictions
                pred_times = []
                pred_values = []
                true_times = []
                true_values = []
                for pos in range(n_positions):
                    for step in range(forecast_steps):
                        if pos + step < len(time_test):
                            pred_times.append(time_test[pos + step])
                            pred_values.append(results[method]["all_predictions"][pos * forecast_steps + step])
                            true_times.append(time_test[pos + step])
                            true_values.append(results[method]["all_true_values"][pos * forecast_steps + step])
                # Plot evaluated true values (with overlap shown as scatter)
                ax.scatter(true_times, true_values, c='green', alpha=0.5, s=20,
                          label=f'True values (evaluated, {len(results[method]["all_true_values"])} points)', 
                          marker='s', zorder=3)
                # Plot predictions (with overlap shown as scatter)
                ax.scatter(pred_times, pred_values, c='red', alpha=0.5, s=20,
                          label=f'Predictions ({method}, {len(results[method]["all_predictions"])} points)', 
                          marker='o', zorder=3)
                # Also plot mean prediction per time point (to show trend)
                from collections import defaultdict
                pred_by_time = defaultdict(list)
                true_by_time = defaultdict(list)
                for pos in range(n_positions):
                    for step in range(forecast_steps):
                        if pos + step < len(time_test):
                            t_idx = pos + step
                            pred_by_time[t_idx].append(results[method]["all_predictions"][pos * forecast_steps + step])
                            true_by_time[t_idx].append(results[method]["all_true_values"][pos * forecast_steps + step])
                time_indices = sorted(pred_by_time.keys())
                mean_preds = [np.mean(pred_by_time[t]) for t in time_indices]
                mean_trues = [np.mean(true_by_time[t]) for t in time_indices]
                time_mean = [time_test[t] for t in time_indices]
                if len(time_indices) > 0:
                    ax.plot(time_mean, mean_trues, 'g-', alpha=0.8, linewidth=2,
                           label='True values (mean per time)', zorder=2)
                    ax.plot(time_mean, mean_preds, 'r--', alpha=0.8, linewidth=2,
                           label=f'Predictions ({method}, mean per time)', zorder=2)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('LAI')
            ax.set_title(f'Time Series Prediction - {method.capitalize()} Method (rolling evaluation, {len(results[method]["all_predictions"])} predictions)\n'
                        f'MAE: {results[method]["metrics"]["mae"]:.4f}, '
                        f'RMSE: {results[method]["metrics"]["rmse"]:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = Path(__file__).parent / "time_series_prediction_results_with_timer.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
        # ----- Additional figure: last 150 time points (46 train, 104 test) -----
        try:
            last_window = 150
            last_train = 46
            last_test = 104
            assert last_train + last_test == last_window

            total_times = time_series.time.values
            total_vals = time_series.values
            start_idx = max(0, len(total_times) - last_window)

            times_window = total_times[start_idx:]
            vals_window = total_vals[start_idx:]
            times_train_w = times_window[:last_train]
            vals_train_w = vals_window[:last_train]
            times_test_w = times_window[last_train:]
            vals_test_w = vals_window[last_train:]

            fig2, axes2 = plt.subplots(len(meth_list), 1, figsize=(12, 4 * len(meth_list)))
            if len(meth_list) == 1:
                axes2 = [axes2]

            # Lower bound time for filtering predictions to last test range
            time_lb = times_test_w[0]

            for idx, method in enumerate(meth_list):
                ax2 = axes2[idx]

                # Plot training segment (last 46)
                ax2.plot(times_train_w, vals_train_w, 'b-', alpha=0.5, label='Training (last 46)', linewidth=1)

                # Plot test segment (last 104)
                ax2.plot(times_test_w, vals_test_w, 'g-', alpha=0.3, label='True test (last 104)', linewidth=1)

                # Rebuild mapping and then filter to last test window by time threshold
                n_pred = len(results[method]["all_predictions"]) if "all_predictions" in results[method] else 0
                if n_pred > 0:
                    forecast_steps_m = results[method].get("forecast_steps", 1)
                    n_positions = max(0, len(test_data) - forecast_steps_m + 1)
                    pred_times = []
                    pred_values = []
                    true_times_f = []
                    true_values_f = []
                    # Use original test times (full) to map, then keep those >= time_lb
                    for pos in range(n_positions):
                        for step in range(forecast_steps_m):
                            if pos + step < len(time_test):
                                t_val = time_test[pos + step]
                                y_pred_val = results[method]["all_predictions"][pos * forecast_steps_m + step]
                                y_true_val = results[method]["all_true_values"][pos * forecast_steps_m + step]
                                if t_val >= time_lb:
                                    pred_times.append(t_val)
                                    pred_values.append(y_pred_val)
                                    true_times_f.append(t_val)
                                    true_values_f.append(y_true_val)
                    if len(true_times_f) > 0:
                        ax2.scatter(true_times_f, true_values_f, c='green', alpha=0.5, s=20,
                                    label=f'True (eval, tail)', marker='s', zorder=3)
                    if len(pred_times) > 0:
                        ax2.scatter(pred_times, pred_values, c='red', alpha=0.5, s=20,
                                    label=f'Pred ({method}, tail)', marker='o', zorder=3)
                        # Also connect predictions with a dashed line for readability
                        try:
                            order = np.argsort(np.array(pred_times, dtype='datetime64[ns]'))
                            pred_times_sorted = np.array(pred_times, dtype='datetime64[ns]')[order]
                            pred_values_sorted = np.array(pred_values, dtype=float)[order]
                            ax2.plot(pred_times_sorted, pred_values_sorted, 'r--', alpha=0.7, linewidth=1.5, label=f'Pred ({method}) line')
                        except Exception:
                            pass

                ax2.set_xlabel('Time')
                ax2.set_ylabel('LAI')
                lbl = method.capitalize()
                ax2.set_title(f'Last 150 Points - {lbl}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file2 = Path(__file__).parent / "time_series_prediction_results_with_timer_last150.png"
            plt.savefig(output_file2, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {output_file2}")
        except Exception as e:
            print(f"Failed to generate last-150 plot: {e}")

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print("Best method by RMSE:")
    best_method = min(results.keys(), key=lambda m: results[m]["metrics"]["rmse"])
    print(f"  {best_method}: RMSE = {results[best_method]['metrics']['rmse']:.4f}")


if __name__ == "__main__":
    main()