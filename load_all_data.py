"""
Load DL4Veg NetCDF data from HiQ_LAI_regrid_to_gridMET_all.nc

This module provides functions to load and work with the LAI (Leaf Area Index)
NetCDF dataset.
"""

from pathlib import Path
from typing import Optional

import os
import xarray as xr


# Default NetCDF file path - can be overridden via DL4VEG_NC_FILE environment variable
DEFAULT_NC_FILE = os.getenv(
    "DL4VEG_NC_FILE",
    "HiQ_LAI_regrid_to_gridMET_all.nc"  # Default to relative path, user should set DL4VEG_NC_FILE env var
)


def load_all_data(
    nc_file: str | Path = DEFAULT_NC_FILE,
    chunk_time: Optional[int] = None,
    engine: Optional[str] = None,
) -> xr.Dataset:
    """
    Load the complete LAI NetCDF dataset.

    Parameters
    ----------
    nc_file : str or Path, optional
        Path to the NetCDF file. Defaults to HiQ_LAI_regrid_to_gridMET_all.nc
    chunk_time : int, optional
        Chunk size for the time dimension. If None, no chunking is applied.
    engine : str, optional
        Backend engine to use ('netcdf4', 'h5netcdf', or None for auto).

    Returns
    -------
    xr.Dataset
        The loaded dataset with dimensions (time, lat, lon) and data variable LAI.

    Examples
    --------
    >>> ds = load_all_data()
    >>> print(ds)
    <xarray.Dataset>
    Dimensions:  (time: 1047, lat: 585, lon: 1386)
    Coordinates:
      * time     (time) datetime64[ns] 2000-02-18 ... 2022-12-27
      * lat      (lat) float64 49.4 49.36 ... 25.07
      * lon      (lon) float64 -124.8 -124.7 ... -67.06
    Data variables:
        LAI      (time, lat, lon) float64 ...
    """
    nc_path = Path(nc_file)
    
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    chunks = {"time": chunk_time} if chunk_time and chunk_time > 0 else None

    # Disable file cache to avoid GC-time close warnings
    with xr.set_options(file_cache_maxsize=10):
        ds = xr.open_dataset(
            str(nc_path),
            chunks=chunks,
            engine=engine,
        )
    
    return ds


def get_time_series(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    variable: str = "LAI",
    method: str = "nearest",
) -> xr.DataArray:
    """
    Extract a time series for a specific location.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the data.
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    variable : str, default "LAI"
        Name of the variable to extract.
    method : str, default "nearest"
        Method for selecting the location ('nearest', 'linear', etc.).

    Returns
    -------
    xr.DataArray
        Time series data for the specified location.
    """
    if variable not in ds.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset. Available: {list(ds.data_vars)}")
    
    data_var = ds[variable]
    time_series = data_var.sel(lat=lat, lon=lon, method=method)
    
    return time_series


def get_spatial_snapshot(
    ds: xr.Dataset,
    time: str | int,
    variable: str = "LAI",
) -> xr.DataArray:
    """
    Extract a spatial snapshot for a specific time.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the data.
    time : str or int
        Time index or time string to select.
    variable : str, default "LAI"
        Name of the variable to extract.

    Returns
    -------
    xr.DataArray
        Spatial snapshot data for the specified time.
    """
    if variable not in ds.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset. Available: {list(ds.data_vars)}")
    
    data_var = ds[variable]
    snapshot = data_var.sel(time=time)
    
    return snapshot


if __name__ == "__main__":
    # Example usage
    print("Loading LAI dataset...")
    ds = load_all_data()
    
    print("\nDataset summary:")
    print(ds)
    
    print("\nDataset info:")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"  Number of time steps: {len(ds.time)}")
    print(f"  Latitude range: {float(ds.lat.min())} to {float(ds.lat.max())}")
    print(f"  Longitude range: {float(ds.lon.min())} to {float(ds.lon.max())}")
    print(f"  Spatial resolution: {len(ds.lat)} x {len(ds.lon)}")
    
    # Example: Extract time series for a location
    print("\n" + "="*80)
    print("Example: Extract time series for a location")
    print("="*80)
    sample_lat = float(ds.lat[len(ds.lat) // 2])
    sample_lon = float(ds.lon[len(ds.lon) // 2])
    print(f"Location: lat={sample_lat:.2f}, lon={sample_lon:.2f}")
    
    time_series = get_time_series(ds, sample_lat, sample_lon)
    print(f"Time series shape: {time_series.shape}")
    print(f"Time series values (first 10): {time_series.values[:10]}")

