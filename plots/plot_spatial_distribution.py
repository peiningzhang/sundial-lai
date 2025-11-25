"""
Script to plot the typical spatial distribution of LAI (Leaf Area Index).
It computes the time-averaged LAI from the full archive and visualizes it as a map.
Colors indicate average canopy greenness.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from pathlib import Path
import argparse
from load_all_data import load_all_data
from util import select_single_location

def main():
    parser = argparse.ArgumentParser(
        description="Plot spatial distribution of LAI with optional location marker"
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Latitude to mark on the plot (optional)"
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Longitude to mark on the plot (optional)"
    )
    parser.add_argument(
        "--mark-seasonal",
        action="store_true",
        help="If set, also mark the location on seasonal snapshots (default: only mark on temporal mean)"
    )
    args = parser.parse_args()
    print("Loading LAI dataset...")
    try:
        ds = load_all_data()
        print(ds)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Get location to mark (if provided)
    marker_lat = None
    marker_lon = None
    if args.lat is not None or args.lon is not None:
        marker_lat, marker_lon = select_single_location(ds, args.lat, args.lon)
        print(f"\nLocation marker: lat={marker_lat:.4f}, lon={marker_lon:.4f}")

    print("Computing temporal mean of LAI...")
    # Compute mean over time dimension
    # This collapses (time, lat, lon) -> (lat, lon)
    mean_lai = ds["LAI"].mean(dim="time", skipna=True)

    print("Plotting mean spatial distribution...")
    fig_mean = plt.figure(figsize=(12, 8))
    ax_mean = fig_mean.add_subplot(111)
    
    # Use xarray's built-in plotting which handles coordinates automatically
    # cmap="YlGn" gives a yellow-to-green gradient suitable for vegetation
    im_mean = mean_lai.plot(
        ax=ax_mean,
        cmap="YlGn",
        add_colorbar=False,
        robust=True  # Handle outliers reasonably
    )
    
    # Add colorbar with larger font
    cbar_mean = fig_mean.colorbar(im_mean, ax=ax_mean, label="Average LAI")
    cbar_mean.set_label("Average LAI", fontsize=24)
    cbar_mean.ax.tick_params(labelsize=20)
    
    ax_mean.set_title("Typical Spatial distribution of LAI\n(Average Canopy Greenness)", fontsize=28)
    ax_mean.set_xlabel("Longitude", fontsize=24)
    ax_mean.set_ylabel("Latitude", fontsize=24)
    ax_mean.tick_params(labelsize=20)
    
    # Add location marker if provided
    if marker_lat is not None and marker_lon is not None:
        ax_mean.plot(
            marker_lon, marker_lat,
            marker='*', markersize=15, color='red', markeredgecolor='white',
            markeredgewidth=2, label=f'Selected location\n({marker_lat:.2f}, {marker_lon:.2f})',
            zorder=10, transform=ax_mean.transData
        )
        ax_mean.legend(loc='best', fontsize=14, framealpha=0.9)
    
    # Save the figure
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    
    out_path_mean = out_dir / "spatial_distribution_lai.png"
    plt.tight_layout()
    plt.savefig(out_path_mean, dpi=300, bbox_inches="tight")
    print(f"Mean figure saved to: {out_path_mean}")
    plt.close(fig_mean)

    # ---------------------------------------------------------
    # Select 4 typical time points in 2001 (approx. seasonal)
    # ---------------------------------------------------------
    # Filter for year 2001
    ds_2001 = ds.sel(time="2001")
    if len(ds_2001.time) == 0:
        print("No data found for year 2001.")
        return

    # Attempt to find indices roughly at 15%, 40%, 65%, 90% through the year 
    # to represent Winter, Spring, Summer, Autumn
    indices = np.linspace(0, len(ds_2001.time) - 1, 5, dtype=int)
    indices = indices[0:-1]
    indices = [idx + 2 for idx in indices]
    selected_times = ds_2001.time.isel(time=indices).values
    
    print(f"Selected 4 time points in 2001: {selected_times}")

    # Create a 2x2 grid figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    # Determine common vmin/vmax for shared color mapping
    # We can use robust quantile-based limits from the first slice or global min/max of the 4 slices
    # To be robust, let's check the min/max of these 4 slices combined
    combined_4_slices = ds["LAI"].sel(time=selected_times)
    # vmin = combined_4_slices.min().item()
    # vmax = combined_4_slices.max().item()
    # Or use robust quantiles if outliers are an issue:
    vmin = combined_4_slices.quantile(0.01).item()
    vmax = combined_4_slices.quantile(0.99).item()
    
    print(f"Shared Color Scale -> vmin: {vmin:.2f}, vmax: {vmax:.2f}")

    for i, t_val in enumerate(selected_times):
        ax = axes[i]
        t_str = str(t_val).split("T")[0]  # YYYY-MM-DD
        
        print(f"Plotting subplot for {t_str}...")
        lai_t = ds["LAI"].sel(time=t_val)
        
        # Plot on the specific axis
        # We turn off the individual colorbar (add_colorbar=False) and add a shared one later
        im = lai_t.plot(
            ax=ax,
            cmap="YlGn",
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False
        )
        
        ax.set_title(f"{t_str}", fontsize=28, pad=8)
        ax.set_xlabel("Longitude", fontsize=24, labelpad=8)
        ax.set_ylabel("Latitude", fontsize=24, labelpad=5)
        # Increase tick label sizes
        ax.tick_params(labelsize=20)
        
        # Add location marker if requested
        if args.mark_seasonal and marker_lat is not None and marker_lon is not None:
            ax.plot(
                marker_lon, marker_lat,
                marker='*', markersize=12, color='red', markeredgecolor='white',
                markeredgewidth=1.5, zorder=10, transform=ax.transData
            )

    # Add a shared colorbar
    # Adjust the position: [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label="LAI")
    cbar.set_label("LAI", fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    
    # plt.suptitle("Spatial distribution of LAI in 2001 (Seasonal Snapshots)", fontsize=32, y=0.98)
    
    out_path_grid = out_dir / "spatial_distribution_lai_2001_grid.png"
    # Adjust layout to make room for colorbar and reduce spacing
    # Reduced hspace and wspace to bring subplots closer together
    plt.subplots_adjust(right=0.9, top=0.92, hspace=0.15, wspace=0.15, left=0.05, bottom=0.08)
    plt.savefig(out_path_grid, dpi=300, bbox_inches="tight")
    print(f"Grid figure saved to: {out_path_grid}")
    plt.close()

if __name__ == "__main__":
    main()

