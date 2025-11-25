"""
Plot the effect of forecast length (H) on forecasting accuracy.

This script visualizes RMSE for different forecast lengths (H) comparing
LSTM and Sundial models with a fixed window size of 512. The plot demonstrates
how Sundial's advantage increases as forecast length increases.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import json
from typing import Dict, List, Optional

# Set matplotlib to use LaTeX for text rendering if available
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 14


def load_results_from_json(json_path: Path) -> Optional[Dict]:
    """Load results from a JSON file if it exists."""
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Convert null (None) to NaN for proper handling in plots
            def convert_none_to_nan(obj):
                if obj is None:
                    return np.nan
                elif isinstance(obj, dict):
                    return {k: convert_none_to_nan(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_none_to_nan(item) for item in obj]
                else:
                    return obj
            return convert_none_to_nan(data)
    return None


def extract_data_for_window_size(data: Dict, target_window_size: int = 512) -> Optional[Dict]:
    """
    Extract data for a specific window size from the JSON structure.
    
    Supports two JSON formats:
    1. Direct format:
       {
         "window_size": 512,
         "forecast_lengths": [1, 4, 8, 12],
         "sundial": {"rmse": [...], "mae": [...]},
         "lstm": {"rmse": [...], "mae": [...]}
       }
    
    2. Nested format:
       {
         "512": {
           "forecast_lengths": [1, 4, 8, 12],
           "sundial": {"rmse": [...], "mae": [...]},
           "lstm": {"rmse": [...], "mae": [...]}
         }
       }
    """
    # Check if data is already for the target window size
    if 'window_size' in data and data['window_size'] == target_window_size:
        return data
    
    # Check if data is nested by window size
    if str(target_window_size) in data:
        return data[str(target_window_size)]
    
    # If neither format matches, return None
    return None


def plot_forecast_length_effect(
    results: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
    dpi: int = 300,
    window_size: int = 512
):
    """
    Plot RMSE vs forecast length (H) for LSTM and Sundial.
    
    Parameters
    ----------
    results : Dict
        Dictionary containing forecast lengths and RMSE for each model.
        Expected format:
        {
            'forecast_lengths': [1, 4, 8, 12],
            'sundial': {'rmse': [...], 'mae': [...]},
            'lstm': {'rmse': [...], 'mae': [...]}
        }
    output_path : Optional[Path]
        Path to save the figure. If None, displays the figure.
    figsize : tuple
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figure.
    window_size : int
        Window size used (for title/labeling).
    """
    forecast_lengths = np.array(results['forecast_lengths'])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
    # Define colors and markers for each model
    model_styles = {
        'sundial': {'color': '#2E86AB', 'marker': 'o', 'linestyle': '-', 'label': 'Sundial'},
        'lstm': {'color': '#F18F01', 'marker': '^', 'linestyle': '-.', 'label': 'LSTM'}
    }
    
    # Plot RMSE for each model
    for model_name in ['sundial', 'lstm']:
        if model_name in results:
            style = model_styles[model_name]
            rmse_values = np.array(results[model_name]['rmse'])
            # Filter out NaN values
            valid_mask = ~np.isnan(rmse_values)
            if np.any(valid_mask):
                valid_forecast_lengths = forecast_lengths[valid_mask]
                valid_rmse = rmse_values[valid_mask]
                ax.plot(
                    valid_forecast_lengths,
                    valid_rmse,
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    label=style['label'],
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8
                )
    
    # Configure axis
    ax.set_xlabel('Forecast Horizon H', fontweight='bold', fontsize=14)
    ax.set_ylabel('RMSE', fontweight='bold', fontsize=14)
    # ax.set_title(f'RMSE vs Forecast Length (Window Size = {window_size})', 
    #              fontweight='bold', fontsize=16, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9, fontsize=12)
    
    # Set x-axis ticks to match forecast_lengths
    ax.set_xticks(forecast_lengths)
    ax.set_xticklabels([str(int(x)) for x in forecast_lengths])
    
    # Set reasonable y-axis limits
    all_rmse = []
    for model_name in ['sundial', 'lstm']:
        if model_name in results:
            rmse_values = np.array(results[model_name]['rmse'])
            valid_rmse = rmse_values[~np.isnan(rmse_values)]
            if len(valid_rmse) > 0:
                all_rmse.extend(valid_rmse.tolist())
    
    if all_rmse:
        y_min = max(0, min(all_rmse) * 0.9)
        y_max = max(all_rmse) * 1.1
        ax.set_ylim(bottom=y_min, top=y_max)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to generate and plot forecast length effects."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Plot the effect of forecast length (H) on forecasting accuracy'
    )
    parser.add_argument(
        '--input-json',
        type=str,
        required=True,
        help='Path to JSON file containing results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='forecast_length_effect.png',
        help='Output path for the figure (default: forecast_length_effect.png)'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[5, 3],
        help='Figure size in inches (width height)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution for saved figure (default: 300)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=512,
        help='Window size to extract from JSON (default: 512)'
    )
    
    args = parser.parse_args()
    
    # Load results from JSON
    json_path = Path(args.input_json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return
    
    data = load_results_from_json(json_path)
    if data is None:
        print(f"Error: Could not load data from {json_path}")
        return
    
    # Extract data for the specified window size
    results = extract_data_for_window_size(data, args.window_size)
    if results is None:
        print(f"Error: Could not find data for window_size={args.window_size} in JSON file")
        print(f"Available keys: {list(data.keys())}")
        return
    
    # Validate required fields
    if 'forecast_lengths' not in results:
        print("Error: 'forecast_lengths' not found in results")
        return
    
    if 'sundial' not in results or 'lstm' not in results:
        print("Error: 'sundial' or 'lstm' data not found in results")
        return
    
    # Generate output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    
    # Create the plot
    plot_forecast_length_effect(
        results,
        output_path=output_path,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        window_size=args.window_size
    )


if __name__ == '__main__':
    main()

