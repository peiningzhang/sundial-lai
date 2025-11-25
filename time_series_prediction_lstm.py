"""
Enhanced LSTM baseline for LAI time series forecasting with multi-mode evaluation,
early stopping, multi-step forecasting, and model saving/loading.

Workflow:
- Load dataset (LAI) with xarray
- Support single or multi-location evaluation
- Use the first train_fraction time points as training, remainder as testing
- Train a small LSTM on sliding windows from training segment
- Evaluate rolling predictions (one-step or multi-step) across test segment
- Print detailed metrics (MAE, RMSE, MAPE, R², MBE, CVRMSE)
- Support model saving/loading for evaluation-only mode
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from load_all_data import load_all_data, get_time_series
from util import select_single_location, sample_locations, pretty_print_sampled_locations

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, IterableDataset
except Exception as exc:
    raise ImportError("PyTorch is required for the LSTM baseline. Please install torch.") from exc


def interpolate_nans(values: np.ndarray) -> np.ndarray:
    """Interpolate NaNs linearly (both directions)."""
    series = pd.Series(values.astype(float))
    series = series.interpolate(method="linear", limit_direction="both")
    return series.values.astype(float)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 1]
        out, _ = self.lstm(x)
        # take last hidden state
        last_h = out[:, -1, :]  # [B, H]
        y = self.head(last_h)   # [B, 1]
        return y.squeeze(-1)    # [B]


def compute_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> dict:
    """Compute comprehensive metrics."""
    mae = float(np.mean(np.abs(true_vals - pred_vals)))
    rmse = float(np.sqrt(np.mean((true_vals - pred_vals) ** 2)))
    # MAPE
    mape_mask = true_vals != 0
    mape = float(np.mean(np.abs((true_vals[mape_mask] - pred_vals[mape_mask]) / true_vals[mape_mask])) * 100) if np.any(mape_mask) else float("nan")
    # R²
    ss_res = float(np.sum((true_vals - pred_vals) ** 2))
    ss_tot = float(np.sum((true_vals - np.mean(true_vals)) ** 2))
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else float("nan")
    # MBE
    mbe = float(np.mean(pred_vals - true_vals))
    # CVRMSE
    cvrmse = float((rmse / np.mean(true_vals)) * 100) if np.mean(true_vals) != 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "mbe": mbe,
        "cvrmse": cvrmse,
    }


def rolling_forecast_multi_step(
    model: nn.Module,
    history: np.ndarray,
    forecast_length: int,
    window_size: int,
    train_mean: float,
    train_std: float,
    device: str,
    eval_batch_size: int = 512,
) -> np.ndarray:
    """Rolling multi-step forecast: predict forecast_length steps ahead recursively."""
    predictions = []
    current_history = history.copy()
    
    for step in range(forecast_length):
        # Build window from current history
        if len(current_history) < window_size:
            # Pad with edge values if needed
            window_vals = np.pad(current_history, (window_size - len(current_history), 0), mode="edge")
        else:
            window_vals = current_history[-window_size:]
        
        # Normalize
        win_norm = (window_vals.astype(np.float32) - train_mean) / train_std
        x_t = torch.from_numpy(win_norm).unsqueeze(0).unsqueeze(-1).to(device)  # [1, W, 1]
        
        # Predict one step ahead
        with torch.no_grad():
            pred_norm = model(x_t)  # [1]
        
        pred = (pred_norm.detach().cpu().item() * train_std) + train_mean
        predictions.append(float(pred))
        
        # Update history with prediction for next step
        current_history = np.append(current_history, pred)
    
    return np.asarray(predictions, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced LSTM baseline for LAI forecasting")
    # Model architecture
    parser.add_argument("--window-size", type=int, default=32, help="Sliding window length for LSTM input")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout in LSTM (ignored if num_layers=1)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--train-fraction", type=float, default=0.9, help="Fraction of time steps used for training (0-1)")
    parser.add_argument("--train-data-fraction", type=float, default=1.0, help="Fraction of training cells to use (0-1]")
    parser.add_argument("--cells-per-epoch-fraction", type=float, default=1.0, help="Fraction of available cells to use per epoch (0-1]")
    
    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Early stopping patience (0 = disabled)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="Minimum change to qualify as improvement")
    parser.add_argument("--early-stopping-metric", type=str, default="rmse", choices=["rmse", "mae"], help="Metric to monitor for early stopping")
    
    # Evaluation
    parser.add_argument("--eval-mode", type=str, choices=["single", "multi"], default="single", help="Evaluation mode: single location or multi-location")
    parser.add_argument("--num-samples", type=int, default=25, help="Number of sampled locations for multi evaluation")
    parser.add_argument("--sample-seed", type=int, default=123, help="Random seed for location sampling (multi mode)")
    parser.add_argument("--training-eval-mode", type=str, choices=["single", "multi"], default="single", help="Evaluation mode during training (for early stopping)")
    parser.add_argument("--forecast-length", type=int, default=1, help="Forecast horizon (1 = rolling one-step, >1 = rolling multi-step)")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Batch size for evaluation forward passes")
    
    # Data
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/mps (auto if None)")
    parser.add_argument("--min-valid-fraction", type=float, default=0.0, help="Min non-NaN fraction over time for a cell (0 = include any non-NaN)")
    parser.add_argument("--max-train-cells", type=int, default=0, help="Max number of spatial cells to use for training (0 = no cap)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lat", type=float, default=None, help="Latitude to evaluate (default: center, single mode)")
    parser.add_argument("--lon", type=float, default=None, help="Longitude to evaluate (default: center, single mode)")
    parser.add_argument("--variable", type=str, default="LAI", help="Variable name")
    
    # Model saving/loading
    parser.add_argument("--load-model", type=str, default=None, help="Path to a saved model to load (if provided, skips training and only evaluates)")
    parser.add_argument("--save-model-dir", type=str, default="checkpoints/lstm", help="Directory to save the trained model (default: checkpoints/lstm)")
    
    # Plotting
    parser.add_argument("--plot", action="store_true", help="Save a PNG plot of predictions")
    
    args = parser.parse_args()

    # Validate arguments
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError(f"--train-fraction must be in (0,1), got {args.train_fraction}")
    if not (0.0 < args.train_data_fraction <= 1.0):
        raise ValueError(f"--train-data-fraction must be in (0,1], got {args.train_data_fraction}")
    if not (0.0 < args.cells_per_epoch_fraction <= 1.0):
        raise ValueError(f"--cells-per-epoch-fraction must be in (0,1], got {args.cells_per_epoch_fraction}")

    # Device selection
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("Loading LAI dataset...")
    ds = load_all_data()
    if args.variable not in ds.data_vars:
        raise ValueError(f"Variable '{args.variable}' not found. Available: {list(ds.data_vars)}")

    rng = np.random.default_rng(args.random_seed)

    # Helper: compute global train mean/std in a streaming pass; return valid indices
    def compute_global_stats_and_valid_indices() -> Tuple[np.ndarray, float, float]:
        lai = ds[args.variable]
        if args.min_valid_fraction <= 0.0:
            valid_mask = lai.notnull().any(dim="time")
        else:
            valid_frac = lai.notnull().mean(dim="time")
            valid_mask = (valid_frac >= float(args.min_valid_fraction))
        valid_indices = np.argwhere(valid_mask.values)
        if valid_indices.size == 0:
            raise RuntimeError("No valid cells meet min_valid_fraction")
        if args.max_train_cells and args.max_train_cells > 0 and valid_indices.shape[0] > args.max_train_cells:
            chosen = rng.choice(valid_indices.shape[0], size=args.max_train_cells, replace=False)
            valid_indices = valid_indices[chosen]

        count = 0
        s1 = 0.0
        s2 = 0.0
        for (i_lat, i_lon) in tqdm(valid_indices, desc="Computing global stats"):
            series = ds[args.variable].isel(lat=int(i_lat), lon=int(i_lon)).values.astype(float)
            series = interpolate_nans(series)
            n_total_i = len(series)
            n_train_i = max(args.window_size + 1, int(n_total_i * args.train_fraction))
            if n_train_i <= 0:
                continue
            train_vals = series[:n_train_i]
            s1 += float(np.sum(train_vals))
            s2 += float(np.sum(train_vals.astype(np.float64) ** 2))
            count += int(n_train_i)
        if count == 0:
            raise RuntimeError("No training values found to compute stats.")
        mean = s1 / count
        var = max(1e-12, (s2 / count) - (mean * mean))
        std = float(np.sqrt(var))
        return valid_indices, float(mean), std

    class StreamingWindowDataset(IterableDataset):
        def __init__(
            self,
            lai_da: xr.DataArray,
            valid_indices: np.ndarray,
            window_size: int,
            train_mean: float,
            train_std: float,
            train_fraction: float,
            cells_per_epoch_fraction: float,
            shuffle_cells: bool,
            rng: np.random.Generator,
        ):
            super().__init__()
            self.lai = lai_da
            self.valid_indices = valid_indices
            self.window_size = window_size
            self.train_mean = train_mean
            self.train_std = train_std if train_std > 0 else 1.0
            self.train_fraction = train_fraction
            self.cells_per_epoch_fraction = cells_per_epoch_fraction
            self.shuffle_cells = shuffle_cells
            self.rng = rng

        def __iter__(self):
            indices = self.valid_indices
            if self.shuffle_cells:
                perm = self.rng.permutation(indices.shape[0])
                indices = indices[perm]
            
            # Apply cells_per_epoch_fraction
            available_cells_count = indices.shape[0]
            cells_to_use = max(1, int(np.ceil(available_cells_count * self.cells_per_epoch_fraction)))
            if cells_to_use > available_cells_count:
                # Repeat indices if needed
                repeat_times = (cells_to_use // available_cells_count) + 1
                indices = np.tile(indices, (repeat_times, 1))[:cells_to_use]
            else:
                indices = indices[:cells_to_use]
            
            for (i_lat, i_lon) in indices:
                series = self.lai.isel(lat=int(i_lat), lon=int(i_lon)).values.astype(float)
                series = interpolate_nans(series)
                n_total_i = len(series)
                n_train_i = max(self.window_size + 1, int(n_total_i * self.train_fraction))
                if n_train_i <= self.window_size:
                    continue
                # Randomly select one window within the training segment for this cell
                end_idx = int(self.rng.integers(self.window_size, n_train_i))
                start_idx = end_idx - self.window_size
                x_win = series[start_idx:end_idx]
                y_val = series[end_idx]
                x_win = (x_win - self.train_mean) / self.train_std
                y_val = (y_val - self.train_mean) / self.train_std
                yield torch.from_numpy(x_win.astype(np.float32)).unsqueeze(-1).float(), torch.tensor(float(y_val)).float()

    # Prepare training data
    print("Preparing training data across all valid spatial cells (streaming)...")
    valid_indices, train_mean, train_std = compute_global_stats_and_valid_indices()
    print(f"Global stats -> mean: {train_mean:.6f}, std: {train_std:.6f}, valid cells: {valid_indices.shape[0]}")
    
    # Optionally subsample the training dataset (cells) while keeping the time-based split unchanged
    if args.train_data_fraction < 1.0:
        num_cells = valid_indices.shape[0]
        num_keep = max(1, int(np.ceil(num_cells * float(args.train_data_fraction))))
        chosen = rng.choice(num_cells, size=num_keep, replace=False)
        train_indices = valid_indices[chosen]
        print(f"Using a fraction of training dataset: kept {train_indices.shape[0]} / {num_cells} cells (~{100.0 * args.train_data_fraction:.1f}%)")
    else:
        train_indices = valid_indices

    # Model
    model = LSTMRegressor(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Early stopping setup
    early_stopping_enabled = args.early_stopping_patience > 0
    best_model_state = None
    best_metric = float("inf")
    best_epoch = 0
    patience_counter = 0

    # Sample validation locations for early stopping (if enabled and using multi-mode)
    validation_locations: Optional[List[Tuple[float, float]]] = None
    if early_stopping_enabled and args.training_eval_mode == "multi":
        validation_locations = sample_locations(
            ds, args.variable, args.min_valid_fraction,
            args.num_samples, args.random_seed + 9999  # Different seed from final test
        )
        print(f"Early stopping validation: {len(validation_locations)} locations")

    # Training loop
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        model.eval()
        print("Model loaded successfully. Skipping training.")
    else:
        model.train()
        for epoch in range(1, args.epochs + 1):
            # Create dataset for this epoch (with cells_per_epoch_fraction)
            stream_ds = StreamingWindowDataset(
                lai_da=ds[args.variable],
                valid_indices=train_indices,
                window_size=args.window_size,
                train_mean=train_mean,
                train_std=train_std,
                train_fraction=args.train_fraction,
                cells_per_epoch_fraction=args.cells_per_epoch_fraction,
                shuffle_cells=True,
                rng=rng,
            )
            train_loader = DataLoader(stream_ds, batch_size=max(1, args.batch_size))
            
            epoch_loss = 0.0
            num_batches = 0
            batch_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
            for xb, yb in batch_bar:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                num_batches += 1
                if num_batches == 1 or (num_batches % 10 == 0):
                    batch_bar.set_postfix(last_loss=f"{float(loss.item()):.6f}", avg_loss=f"{(epoch_loss/max(1,num_batches)):.6f}")
            avg_loss = epoch_loss / max(1, num_batches)
            print(f"Epoch {epoch:03d}/{args.epochs} - train MSE: {avg_loss:.6f} (batches: {num_batches})")

            # Evaluation during training (for early stopping or monitoring)
            if early_stopping_enabled or args.training_eval_mode:
                model.eval()
                eval_locations = validation_locations if (early_stopping_enabled and args.training_eval_mode == "multi") else None
                
                if args.training_eval_mode == "single":
                    # Single location evaluation
                    sample_lat, sample_lon = select_single_location(ds, args.lat, args.lon)
                    series_da = get_time_series(ds, sample_lat, sample_lon, variable=args.variable)
                    values = interpolate_nans(series_da.values.astype(float))
                    n_total = len(values)
                    n_train = max(args.window_size + 1, int(n_total * args.train_fraction))
                    n_test = n_total - n_train
                    test_vals = values[n_train:]
                    
                    if args.forecast_length == 1:
                        # Rolling one-step
                        preds_list = []
                        history = values[:n_train].copy()
                        for pos in range(len(test_vals)):
                            if len(history) < args.window_size:
                                window_vals = np.pad(history, (args.window_size - len(history), 0), mode="edge")
                            else:
                                window_vals = history[-args.window_size:]
                            win_norm = (window_vals.astype(np.float32) - train_mean) / train_std
                            x_t = torch.from_numpy(win_norm).unsqueeze(0).unsqueeze(-1).to(device)
                            with torch.no_grad():
                                pred_norm = model(x_t)
                            pred = (pred_norm.detach().cpu().item() * train_std) + train_mean
                            preds_list.append(float(pred))
                            history = np.append(history, float(test_vals[pos]))
                        preds_np = np.asarray(preds_list, dtype=float)
                        trues_np = test_vals[:len(preds_np)].astype(float)
                    else:
                        # Rolling multi-step
                        preds_list = []
                        history = values[:n_train].copy()
                        forecast_len = int(args.forecast_length)
                        pos = 0
                        while pos + forecast_len <= len(test_vals):
                            preds_step = rolling_forecast_multi_step(
                                model, history, forecast_len, args.window_size,
                                train_mean, train_std, device
                            )
                            preds_list.extend(preds_step.tolist())
                            history = np.append(history, test_vals[pos:pos + forecast_len].astype(float))
                            pos += forecast_len
                        preds_np = np.asarray(preds_list, dtype=float)
                        trues_np = test_vals[:len(preds_np)].astype(float)
                    
                    metrics = compute_metrics(trues_np, preds_np)
                    current_metric = metrics[args.early_stopping_metric] if early_stopping_enabled else metrics["rmse"]
                    print(f"  -> Training eval RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")
                
                elif args.training_eval_mode == "multi" and eval_locations:
                    # Multi-location evaluation
                    all_preds_pooled: List[float] = []
                    all_trues_pooled: List[float] = []
                    rmses: List[float] = []
                    maes: List[float] = []
                    
                    for (lt, ln) in eval_locations:
                        series_da = get_time_series(ds, float(lt), float(ln), variable=args.variable)
                        values = interpolate_nans(series_da.values.astype(float))
                        n_total = len(values)
                        n_train = max(args.window_size + 1, int(n_total * args.train_fraction))
                        n_test = n_total - n_train
                        test_vals = values[n_train:]
                        
                        if args.forecast_length == 1:
                            preds_list = []
                            history = values[:n_train].copy()
                            for pos in range(len(test_vals)):
                                if len(history) < args.window_size:
                                    window_vals = np.pad(history, (args.window_size - len(history), 0), mode="edge")
                                else:
                                    window_vals = history[-args.window_size:]
                                win_norm = (window_vals.astype(np.float32) - train_mean) / train_std
                                x_t = torch.from_numpy(win_norm).unsqueeze(0).unsqueeze(-1).to(device)
                                with torch.no_grad():
                                    pred_norm = model(x_t)
                                pred = (pred_norm.detach().cpu().item() * train_std) + train_mean
                                preds_list.append(float(pred))
                                history = np.append(history, float(test_vals[pos]))
                            preds_np = np.asarray(preds_list, dtype=float)
                            trues_np = test_vals[:len(preds_np)].astype(float)
                        else:
                            preds_list = []
                            history = values[:n_train].copy()
                            forecast_len = int(args.forecast_length)
                            pos = 0
                            while pos + forecast_len <= len(test_vals):
                                preds_step = rolling_forecast_multi_step(
                                    model, history, forecast_len, args.window_size,
                                    train_mean, train_std, device
                                )
                                preds_list.extend(preds_step.tolist())
                                history = np.append(history, test_vals[pos:pos + forecast_len].astype(float))
                                pos += forecast_len
                            preds_np = np.asarray(preds_list, dtype=float)
                            trues_np = test_vals[:len(preds_np)].astype(float)
                        
                        if trues_np.size > 0 and preds_np.size > 0:
                            metrics_i = compute_metrics(trues_np, preds_np)
                            rmses.append(metrics_i["rmse"])
                            maes.append(metrics_i["mae"])
                            all_preds_pooled.extend(preds_np.tolist())
                            all_trues_pooled.extend(trues_np.tolist())
                    
                    if len(all_preds_pooled) > 0:
                        preds_pooled_np = np.asarray(all_preds_pooled, dtype=float)
                        trues_pooled_np = np.asarray(all_trues_pooled, dtype=float)
                        metrics = compute_metrics(trues_pooled_np, preds_pooled_np)
                        current_metric = metrics[args.early_stopping_metric] if early_stopping_enabled else metrics["rmse"]
                        print(f"  -> Training eval RMSE: {metrics['rmse']:.4f} ± {float(np.std(rmses)):.4f} | MAE: {metrics['mae']:.4f} ± {float(np.std(maes)):.4f}")
                    else:
                        current_metric = float("inf")
                
                # Early stopping logic
                if early_stopping_enabled:
                    improved = False
                    if args.early_stopping_min_delta > 0:
                        improved = (best_metric - current_metric) >= args.early_stopping_min_delta
                    else:
                        improved = current_metric < best_metric
                    
                    if improved:
                        best_metric = current_metric
                        best_epoch = epoch
                        patience_counter = 0
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        print(f"  -> Best {args.early_stopping_metric.upper()} improved to {best_metric:.4f}")
                    else:
                        patience_counter += 1
                        print(f"  -> No improvement ({patience_counter}/{args.early_stopping_patience})")
                    
                    if patience_counter >= args.early_stopping_patience:
                        print(f"\nEarly stopping triggered at epoch {epoch}. Best {args.early_stopping_metric.upper()} = {best_metric:.4f} at epoch {best_epoch}")
                        break
                
                model.train()

        # Restore best model if early stopping was enabled
        if early_stopping_enabled and best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\nRestored best model from epoch {best_epoch} ({args.early_stopping_metric.upper()} = {best_metric:.4f})")

        # Save the trained model
        model.eval()
        save_dir = Path(args.save_model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_filename = f"lstm_model_window{args.window_size}_forecast{args.forecast_length}.pth"
        save_path = save_dir / model_filename
        torch.save(model.state_dict(), save_path)
        print(f"\nModel saved to: {save_path}")

    # Final evaluation
    model.eval()
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    if args.eval_mode == "single":
        sample_lat, sample_lon = select_single_location(ds, args.lat, args.lon)
        print(f"Selected location: lat={sample_lat:.4f}, lon={sample_lon:.4f}")
        print("Evaluation mode: single")
        
        series_da = get_time_series(ds, sample_lat, sample_lon, variable=args.variable)
        values = interpolate_nans(series_da.values.astype(float))
        times = series_da.time.values
        n_total = len(values)
        n_train = max(args.window_size + 1, int(n_total * args.train_fraction))
        n_test = n_total - n_train
        train_series = values[:n_train]
        test_series = values[n_train:]
        times_train = times[:n_train]
        times_test = times[n_train:]
        
        if args.forecast_length == 1:
            # Rolling one-step evaluation
            preds_list = []
            history = train_series.copy()
            for pos in tqdm(range(len(test_series)), desc="Rolling one-step forecasting"):
                if len(history) < args.window_size:
                    window_vals = np.pad(history, (args.window_size - len(history), 0), mode="edge")
                else:
                    window_vals = history[-args.window_size:]
                win_norm = (window_vals.astype(np.float32) - train_mean) / train_std
                x_t = torch.from_numpy(win_norm).unsqueeze(0).unsqueeze(-1).to(device)
                with torch.no_grad():
                    pred_norm = model(x_t)
                pred = (pred_norm.detach().cpu().item() * train_std) + train_mean
                preds_list.append(float(pred))
                history = np.append(history, float(test_series[pos]))
            preds_np = np.asarray(preds_list, dtype=float)
            trues_np = test_series[:len(preds_np)].astype(float)
        else:
            # Rolling multi-step evaluation
            preds_list = []
            history = train_series.copy()
            forecast_len = int(args.forecast_length)
            pos = 0
            while pos + forecast_len <= len(test_series):
                preds_step = rolling_forecast_multi_step(
                    model, history, forecast_len, args.window_size,
                    train_mean, train_std, device
                )
                preds_list.extend(preds_step.tolist())
                history = np.append(history, test_series[pos:pos + forecast_len].astype(float))
                pos += forecast_len
            preds_np = np.asarray(preds_list, dtype=float)
            trues_np = test_series[:len(preds_np)].astype(float)
        
        metrics = compute_metrics(trues_np, preds_np)
        print(f"  Number of predictions: {len(preds_np)}")
        print(f"  MAE:   {metrics['mae']:.4f}")
        print(f"  RMSE:  {metrics['rmse']:.4f}")
        print(f"  MAPE:  {metrics['mape']:.2f}%")
        print(f"  R²:    {metrics['r2']:.4f}")
        print(f"  MBE:   {metrics['mbe']:.4f}")
        print(f"  CVRMSE: {metrics['cvrmse']:.2f}%")
        
        # Plotting
        if args.plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1, figsize=(12, 4))
                ax.plot(times_train, train_series, "b-", alpha=0.5, label="Training")
                ax.plot(times_test[:len(preds_np)], trues_np, "g-", alpha=0.4, label="True (test)")
                ax.plot(times_test[:len(preds_np)], preds_np, "r-", alpha=0.9, label="LSTM pred (test)")
                ax.set_title("LSTM baseline - rolling forecasts on test set")
                ax.set_xlabel("Time")
                ax.set_ylabel("LAI")
                ax.grid(True, alpha=0.3)
                ax.legend()
                out_path = Path(__file__).parent / "time_series_prediction_lstm_results.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"Plot saved to: {out_path}")
            except Exception as exc:
                print(f"Plotting skipped due to: {exc}")
    
    else:  # multi mode
        sampled_locations = sample_locations(
            ds, args.variable, args.min_valid_fraction,
            args.num_samples, args.sample_seed
        )
        print("Evaluation mode: multi")
        print(f"Locations: {len(sampled_locations)}")
        pretty_print_sampled_locations(sampled_locations, max_show=5)
        
        maes: List[float] = []
        rmses: List[float] = []
        mapes: List[float] = []
        r2s: List[float] = []
        mbes: List[float] = []
        cvrmses: List[float] = []
        n_preds_total = 0
        # Pool all predictions and true values for proper R² calculation
        all_preds_pooled: List[float] = []
        all_trues_pooled: List[float] = []
        
        for (lt, ln) in tqdm(sampled_locations, desc="Evaluating locations", leave=True):
            series_da = get_time_series(ds, float(lt), float(ln), variable=args.variable)
            values = interpolate_nans(series_da.values.astype(float))
            n_total = len(values)
            n_train = max(args.window_size + 1, int(n_total * args.train_fraction))
            n_test = n_total - n_train
            test_vals = values[n_train:]
            
            if args.forecast_length == 1:
                preds_list = []
                history = values[:n_train].copy()
                for pos in range(len(test_vals)):
                    if len(history) < args.window_size:
                        window_vals = np.pad(history, (args.window_size - len(history), 0), mode="edge")
                    else:
                        window_vals = history[-args.window_size:]
                    win_norm = (window_vals.astype(np.float32) - train_mean) / train_std
                    x_t = torch.from_numpy(win_norm).unsqueeze(0).unsqueeze(-1).to(device)
                    with torch.no_grad():
                        pred_norm = model(x_t)
                    pred = (pred_norm.detach().cpu().item() * train_std) + train_mean
                    preds_list.append(float(pred))
                    history = np.append(history, float(test_vals[pos]))
                preds_np = np.asarray(preds_list, dtype=float)
                trues_np = test_vals[:len(preds_np)].astype(float)
            else:
                preds_list = []
                history = values[:n_train].copy()
                forecast_len = int(args.forecast_length)
                pos = 0
                while pos + forecast_len <= len(test_vals):
                    preds_step = rolling_forecast_multi_step(
                        model, history, forecast_len, args.window_size,
                        train_mean, train_std, device
                    )
                    preds_list.extend(preds_step.tolist())
                    history = np.append(history, test_vals[pos:pos + forecast_len].astype(float))
                    pos += forecast_len
                preds_np = np.asarray(preds_list, dtype=float)
                trues_np = test_vals[:len(preds_np)].astype(float)
            
            if trues_np.size == 0 or preds_np.size == 0:
                continue
            
            metrics_i = compute_metrics(trues_np, preds_np)
            maes.append(metrics_i["mae"])
            rmses.append(metrics_i["rmse"])
            n_preds_total += len(preds_np)
            all_preds_pooled.extend(preds_np.tolist())
            all_trues_pooled.extend(trues_np.tolist())
            if not np.isnan(metrics_i["mape"]):
                mapes.append(metrics_i["mape"])
            if not np.isnan(metrics_i["r2"]):
                r2s.append(metrics_i["r2"])
            mbes.append(metrics_i["mbe"])
            if not np.isnan(metrics_i["cvrmse"]):
                cvrmses.append(metrics_i["cvrmse"])
        
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
            metrics_pooled = compute_metrics(trues_pooled_np, preds_pooled_np)
            r2_mean = metrics_pooled["r2"]
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
