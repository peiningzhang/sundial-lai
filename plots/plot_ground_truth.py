"""
Plot ground truth time series for a selected single location.

This script loads LAI data and plots the complete ground truth time series
for a specified location, using a clean plotting style similar to the baseline script.
"""

import numpy as np
from pathlib import Path
import argparse

import xarray as xr
from load_all_data import load_all_data, get_time_series
from util import select_single_location


def main():
    """Main function to plot ground truth time series."""
    parser = argparse.ArgumentParser(
        description="Plot ground truth LAI time series for a selected location"
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Latitude to plot (default: center of domain)"
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Longitude to plot (default: center of domain)"
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="LAI",
        help="Variable name to plot (default: LAI)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ground_truth_timeseries.png",
        help="Output path for the figure (default: ground_truth_timeseries.png)"
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 4],
        help="Figure size in inches (width height, default: 12 4)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for saved figure (default: 150)"
    )
    
    args = parser.parse_args()
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("Error: matplotlib not available. Cannot generate plots.")
        return
    
    # Load data
    print("Loading LAI dataset...")
    ds = load_all_data()
    
    # Validate variable
    if args.variable not in ds.data_vars:
        raise ValueError(f"Variable '{args.variable}' not found. Available: {list(ds.data_vars)}")
    
    # Select location
    sample_lat, sample_lon = select_single_location(ds, args.lat, args.lon)
    print(f"\nSelected location: lat={sample_lat:.4f}, lon={sample_lon:.4f}")
    
    # Extract time series
    print("Extracting time series...")
    time_series = get_time_series(ds, sample_lat, sample_lon, variable=args.variable)
    
    # Get time coordinates and values
    times = time_series.time.values
    values = time_series.values.astype(float)
    
    n_total = len(time_series)
    print(f"\nTime series information:")
    print(f"  Total time steps: {n_total}")
    print(f"  Time range: {times[0]} to {times[-1]}")
    print(f"  Variable: {args.variable}")
    print(f"  Mean value: {np.nanmean(values):.4f}")
    print(f"  Std value: {np.nanstd(values):.4f}")
    print(f"  Min value: {np.nanmin(values):.4f}")
    print(f"  Max value: {np.nanmax(values):.4f}")
    
    # Create plot
    print(f"\nGenerating plot...")
    fig, ax = plt.subplots(1, 1, figsize=tuple(args.figsize), dpi=args.dpi)
    
    # Plot ground truth time series
    ax.plot(times, values, 'b-', alpha=0.7, label=f'{args.variable} (ground truth)', linewidth=2)
    
    # Add styling with larger fonts
    ax.set_xlabel('Time', fontweight='bold', fontsize=24)
    ax.set_ylabel(f'{args.variable}', fontweight='bold', fontsize=24)
    ax.set_title(
        f'Ground Truth Time Series - {args.variable}\n'
        f'Location: lat={sample_lat:.4f}, lon={sample_lon:.4f}',
        fontweight='bold',
        fontsize=28
    )
    ax.legend(loc='best', framealpha=0.9, fontsize=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=20)
    
    # Format x-axis dates
    try:
        import matplotlib.dates as mdates
        # Auto-format dates based on time range
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        # Rotate date labels for readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=20)
    except Exception:
        # If date formatting fails, just use default
        pass
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    main()

