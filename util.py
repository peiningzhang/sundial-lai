from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Tuple, List
import xarray as xr
try:
    import torch
    from torch.utils.data import IterableDataset as TorchIterableDataset  # type: ignore
except Exception:
    torch = None  # type: ignore[assignment]
    TorchIterableDataset = object  # type: ignore[misc,assignment]


def rolling_eval(series: np.ndarray, model_predict_fn: Callable[[np.ndarray], float], window_size: int, train_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic rolling one-step-ahead evaluation.
    - Interpolates NaNs linearly (both directions)
    - Splits by train_fraction (train first portion, test remainder)
    - Builds windows from true values and predicts next step via model_predict_fn
    - Returns (preds, trues) on original scale
    """
    values = pd.Series(series.astype(float)).interpolate(method="linear", limit_direction="both").values.astype(float)
    n_total = len(values)
    n_train = max(int(window_size) + 1, int(n_total * float(train_fraction)))
    n_test = n_total - n_train
    if n_test <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    preds = np.empty(n_test, dtype=float)
    trues = values[n_train:].astype(float)
    for i in range(n_test):
        end_idx = n_train + i
        start_idx = end_idx - int(window_size)
        if start_idx < 0:
            window_vals = values[max(0, start_idx):end_idx]
            if len(window_vals) < int(window_size):
                window_vals = np.pad(window_vals, (int(window_size) - len(window_vals), 0), mode="edge")
        else:
            window_vals = values[start_idx:end_idx]
        preds[i] = float(model_predict_fn(window_vals.astype(np.float32)))
    return preds.astype(float), trues.astype(float)


def interpolate_nans(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values.astype(float))
    series = series.interpolate(method="linear", limit_direction="both")
    return series.values.astype(float)


def select_single_location(ds: xr.Dataset, lat_arg: float | None, lon_arg: float | None) -> Tuple[float, float]:
    if lat_arg is None:
        sample_lat = float(ds.lat[len(ds.lat) // 2])
    else:
        sample_lat = float(lat_arg)
    if lon_arg is None:
        sample_lon = float(ds.lon[len(ds.lon) // 2])
    else:
        sample_lon = float(lon_arg)
    return sample_lat, sample_lon


def sample_locations(ds: xr.Dataset, variable: str, min_valid_fraction: float, num_samples: int, sample_seed: int) -> List[Tuple[float, float]]:
    rng = np.random.default_rng(int(sample_seed))
    lai = ds[variable]
    if float(min_valid_fraction) <= 0.0:
        valid_mask = lai.notnull().any(dim="time")
    else:
        valid_frac = lai.notnull().mean(dim="time")
        valid_mask = (valid_frac >= float(min_valid_fraction))
    valid_indices_all = np.argwhere(valid_mask.values)
    if valid_indices_all.size == 0:
        raise RuntimeError("No valid cells for sampling")
    k = min(int(num_samples), valid_indices_all.shape[0])
    chosen = rng.choice(valid_indices_all.shape[0], size=k, replace=False)
    sampled_indices = valid_indices_all[chosen]
    lats_arr = ds.lat.values
    lons_arr = ds.lon.values
    return [(float(lats_arr[int(i_lat)]), float(lons_arr[int(i_lon)])) for (i_lat, i_lon) in sampled_indices]


def pretty_print_sampled_locations(sampled_locations: List[Tuple[float, float]], max_show: int = 5) -> None:
    n = len(sampled_locations)
    if n == 0:
        print("No locations sampled.")
        return
    if n <= max_show:
        print("Sampled coordinates (lat, lon):")
        for (lt, ln) in sampled_locations:
            print(f"  ({lt:.4f}, {ln:.4f})")
    else:
        print(f"First {max_show} sampled coordinates (lat, lon):")
        for (lt, ln) in sampled_locations[:max_show]:
            print(f"  ({lt:.4f}, {ln:.4f})")


def build_train_windows(series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    inputs: List[np.ndarray] = []
    targets: List[float] = []
    n = len(series)
    for end_idx in range(int(window_size), n):
        start_idx = end_idx - int(window_size)
        inputs.append(series[start_idx:end_idx])
        targets.append(series[end_idx])
    X = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    return X, y


def compute_mae_rmse(true_vals: np.ndarray, pred_vals: np.ndarray) -> Tuple[float, float]:
    mae = float(np.mean(np.abs(true_vals - pred_vals)))
    rmse = float(np.sqrt(np.mean((true_vals - pred_vals) ** 2)))
    return mae, rmse


class StreamingWindowDataset(TorchIterableDataset):  # Subclass torch IterableDataset so DataLoader treats it correctly
    """
    Unified streaming dataset for training from an xarray DataArray over valid spatial cells.
    Modes:
      - "random_window_per_cell": yield one random window per cell (per pass)
      - "all_windows_per_cell": yield all training windows per cell (optionally subsampled)
    """
    def __init__(
        self,
        lai_data: xr.DataArray,
        valid_idx: np.ndarray,
        window_size: int,
        train_mean: float,
        train_std: float,
        train_fraction: float,
        mode: str = "random_window_per_cell",
        max_windows_per_cell: int = 0,
        max_windows_total: int = 0,
        shuffle_cells: bool = True,
        rng: np.random.Generator | None = None,
        unsqueeze_channel: bool = False,
    ):
        if torch is None:
            raise ImportError("torch is required for StreamingWindowDataset")
        self.lai = lai_data
        self.valid_indices = valid_idx
        self.window_size = int(window_size)
        self.train_mean = float(train_mean)
        self.train_std = float(train_std) if float(train_std) > 0 else 1.0
        self.train_fraction = float(train_fraction)
        self.mode = str(mode)
        self.max_windows_per_cell = int(max_windows_per_cell)
        self.max_windows_total = int(max_windows_total)
        self.shuffle_cells = bool(shuffle_cells)
        self.rng = rng or np.random.default_rng(42)
        self.unsqueeze_channel = bool(unsqueeze_channel)

    def __iter__(self):
        indices = self.valid_indices
        if self.shuffle_cells:
            perm = self.rng.permutation(indices.shape[0])
            indices = indices[perm]
        total_emitted = 0
        for (i_lat, i_lon) in indices:
            series = self.lai.isel(lat=int(i_lat), lon=int(i_lon)).values.astype(float)
            series = interpolate_nans(series)
            n_total_i = len(series)
            n_train_i = max(self.window_size + 1, int(n_total_i * self.train_fraction))
            if n_train_i <= self.window_size:
                continue
            if self.mode == "random_window_per_cell":
                end_idx = int(self.rng.integers(self.window_size, n_train_i))
                start_idx = end_idx - self.window_size
                x_win = series[start_idx:end_idx]
                y_val = series[end_idx]
                x_win = (x_win - self.train_mean) / self.train_std
                y_val = (y_val - self.train_mean) / self.train_std
                x_t = torch.from_numpy(x_win.astype(np.float32))
                if self.unsqueeze_channel:
                    x_t = x_t.unsqueeze(-1)  # [W,1]
                yield x_t.float(), torch.tensor(float(y_val)).float()
                total_emitted += 1
                if self.max_windows_total and total_emitted >= self.max_windows_total:
                    return
            elif self.mode == "all_windows_per_cell":
                train_series_i = series[:n_train_i]
                X_i, y_i = build_train_windows(train_series_i, self.window_size)
                if X_i.shape[0] == 0:
                    continue
                if self.max_windows_per_cell and X_i.shape[0] > self.max_windows_per_cell:
                    sel = self.rng.choice(X_i.shape[0], size=int(self.max_windows_per_cell), replace=False)
                    X_i = X_i[sel]
                    y_i = y_i[sel]
                X_i = (X_i - self.train_mean) / self.train_std
                y_i = (y_i - self.train_mean) / self.train_std
                for j in range(X_i.shape[0]):
                    x_t = torch.from_numpy(X_i[j]).float()
                    if self.unsqueeze_channel:
                        x_t = x_t.unsqueeze(-1)  # [W,1]
                    yield x_t, torch.tensor(float(y_i[j])).float()
                    total_emitted += 1
                    if self.max_windows_total and total_emitted >= self.max_windows_total:
                        return
            else:
                raise ValueError(f"Unknown StreamingWindowDataset mode: {self.mode}")


def compute_global_train_stats_and_indices(
    lai_data: xr.DataArray,
    train_fraction: float,
    window_size: int,
    min_valid_fraction: float = 0.0,
    max_train_cells: int = 0,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute valid spatial indices and global mean/std over the training portion across those indices.
    - lai_data: xr.DataArray with dims including 'time', 'lat', 'lon'
    - train_fraction: fraction of time used for training per cell
    - window_size: minimum history length to allow a cell
    - min_valid_fraction: minimum fraction of non-NaN over time to consider a cell valid
    - max_train_cells: optional cap of how many valid cells to include
    Returns (valid_indices [N,2], mean, std).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if float(min_valid_fraction) <= 0.0:
        valid_mask = lai_data.notnull().any(dim="time")
    else:
        valid_frac = lai_data.notnull().mean(dim="time")
        valid_mask = (valid_frac >= float(min_valid_fraction))
    valid_indices = np.argwhere(valid_mask.values)
    if valid_indices.size == 0:
        raise RuntimeError("No valid cells meet min_valid_fraction")
    if max_train_cells and max_train_cells > 0 and valid_indices.shape[0] > int(max_train_cells):
        chosen = rng.choice(valid_indices.shape[0], size=int(max_train_cells), replace=False)
        valid_indices = valid_indices[chosen]
    # Streaming mean/std over training segments
    count = 0
    s1 = 0.0
    s2 = 0.0
    # Optional tqdm if available
    try:
        from tqdm import tqdm as _tqdm  # type: ignore
        iterator = _tqdm(valid_indices, desc="Computing global stats")
    except Exception:
        iterator = valid_indices
    for (i_lat, i_lon) in iterator:
        series = lai_data.isel(lat=int(i_lat), lon=int(i_lon)).values.astype(float)
        series = interpolate_nans(series)
        n_total_i = len(series)
        n_train_i = max(int(window_size) + 1, int(n_total_i * float(train_fraction)))
        if n_train_i <= 0:
            continue
        train_vals_i = series[:n_train_i]
        s1 += float(np.sum(train_vals_i))
        s2 += float(np.sum(train_vals_i.astype(np.float64) ** 2))
        count += int(n_train_i)
    if count == 0:
        raise RuntimeError("No training values found to compute stats.")
    mean = s1 / count
    var = max(1e-12, (s2 / count) - (mean * mean))
    std = float(np.sqrt(var))
    return valid_indices, float(mean), float(std)

