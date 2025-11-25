"""
Plot the effect of input window size on forecasting accuracy.

This script visualizes RMSE and MAE metrics for different window sizes
across different models (Sundial, ARIMA, LSTM). The plot demonstrates that
larger input windows consistently improve Sundial performance, while ARIMA
and LSTM saturate at moderate window lengths.
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


def generate_synthetic_data() -> Dict:
    """
    Generate synthetic data that demonstrates the described behavior:
    - Sundial: improves with larger windows
    - ARIMA: saturates at moderate window lengths
    - LSTM: saturates at moderate window lengths
    """
    window_sizes = np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    
    # Sundial: improves with larger windows (exponential decay)
    sundial_rmse = 0.8 * np.exp(-window_sizes / 25) + 0.15 + np.random.normal(0, 0.02, len(window_sizes))
    sundial_mae = 0.7 * np.exp(-window_sizes / 25) + 0.12 + np.random.normal(0, 0.015, len(window_sizes))
    
    # ARIMA: saturates around window size 16-20
    arima_rmse = 0.5 * np.exp(-window_sizes / 10) + 0.25 + np.random.normal(0, 0.015, len(window_sizes))
    arima_mae = 0.45 * np.exp(-window_sizes / 10) + 0.22 + np.random.normal(0, 0.012, len(window_sizes))
    # Make it saturate (no further improvement after window size 20)
    saturation_idx = window_sizes >= 20
    arima_rmse[saturation_idx] = arima_rmse[window_sizes == 20][0] + np.random.normal(0, 0.01, np.sum(saturation_idx))
    arima_mae[saturation_idx] = arima_mae[window_sizes == 20][0] + np.random.normal(0, 0.008, np.sum(saturation_idx))
    
    # LSTM: saturates around window size 16-20
    lstm_rmse = 0.55 * np.exp(-window_sizes / 12) + 0.28 + np.random.normal(0, 0.015, len(window_sizes))
    lstm_mae = 0.5 * np.exp(-window_sizes / 12) + 0.25 + np.random.normal(0, 0.012, len(window_sizes))
    # Make it saturate (no further improvement after window size 20)
    saturation_idx = window_sizes >= 20
    lstm_rmse[saturation_idx] = lstm_rmse[window_sizes == 20][0] + np.random.normal(0, 0.01, np.sum(saturation_idx))
    lstm_mae[saturation_idx] = lstm_mae[window_sizes == 20][0] + np.random.normal(0, 0.008, np.sum(saturation_idx))
    
    # Ensure values are positive
    sundial_rmse = np.maximum(sundial_rmse, 0.1)
    sundial_mae = np.maximum(sundial_mae, 0.08)
    arima_rmse = np.maximum(arima_rmse, 0.2)
    arima_mae = np.maximum(arima_mae, 0.18)
    lstm_rmse = np.maximum(lstm_rmse, 0.25)
    lstm_mae = np.maximum(lstm_mae, 0.22)
    
    return {
        'window_sizes': window_sizes.tolist(),
        'sundial': {
            'rmse': sundial_rmse.tolist(),
            'mae': sundial_mae.tolist()
        },
        'arima': {
            'rmse': arima_rmse.tolist(),
            'mae': arima_mae.tolist()
        },
        'lstm': {
            'rmse': lstm_rmse.tolist(),
            'mae': lstm_mae.tolist()
        }
    }


def plot_window_size_effect(
    results: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    mode: str = 'both',
    exclude_arima: bool = False
):
    """
    Plot RMSE and MAE vs window size for different models.
    
    Parameters
    ----------
    results : Dict
        Dictionary containing window sizes and metrics for each model.
        Expected format:
        {
            'window_sizes': [4, 8, 12, ...],
            'sundial': {'rmse': [...], 'mae': [...]},
            'arima': {'rmse': [...], 'mae': [...]},
            'lstm': {'rmse': [...], 'mae': [...]}
        }
    output_path : Optional[Path]
        Path to save the figure. If None, displays the figure.
    figsize : tuple
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figure.
    """
    window_sizes = np.array(results['window_sizes'])
    
    # Create figure with one or two subplots based on mode
    if mode == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        axes = {'rmse': ax1, 'mae': ax2}
    elif mode == 'rmse':
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]/2, figsize[1]), dpi=dpi)
        axes = {'rmse': ax1}
        ax2 = None
    else:  # mode == 'mae'
        fig, ax2 = plt.subplots(1, 1, figsize=(figsize[0]/2, figsize[1]), dpi=dpi)
        axes = {'mae': ax2}
        ax1 = None
    
    # Define colors and markers for each model
    model_styles = {
        'sundial': {'color': '#2E86AB', 'marker': 'o', 'linestyle': '-', 'label': 'Sundial'},
        'arima': {'color': '#A23B72', 'marker': 's', 'linestyle': '--', 'label': 'ARIMA'},
        'lstm': {'color': '#F18F01', 'marker': '^', 'linestyle': '-.', 'label': 'LSTM'}
    }
    
    # Helper function to configure an axis
    def configure_axis(ax, ylabel, title, tick_locations):
        ax.set_xlabel('Input Window Size', fontweight='bold', fontsize=14)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
        # ax.set_title(title, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.legend(loc='best', framealpha=0.9, fontsize=12)
        # Set x-axis limits based on valid window sizes (avoid 0 for log scale)
        valid_window_min = window_sizes[window_sizes > 0].min() if np.any(window_sizes > 0) else 1
        ax.set_xlim(left=valid_window_min * 0.9, right=window_sizes.max() * 1.1)
        # Set custom tick locations and labels for better readability
        ax.set_xticks(tick_locations)
        ax.set_xticklabels([str(int(x)) for x in tick_locations])
    
    tick_locations = window_sizes[window_sizes > 0]
    
    # Determine which models to plot
    models_to_plot = ['sundial', 'arima', 'lstm']
    if exclude_arima:
        models_to_plot = ['sundial', 'lstm']
    
    # Plot RMSE if requested
    if 'rmse' in axes:
        ax1 = axes['rmse']
        for model_name in models_to_plot:
            if model_name in results:
                style = model_styles[model_name]
                rmse_values = np.array(results[model_name]['rmse'])
                # Filter out NaN values
                valid_mask = ~np.isnan(rmse_values)
                if np.any(valid_mask):
                    valid_window_sizes = window_sizes[valid_mask]
                    valid_rmse = rmse_values[valid_mask]
                    ax1.plot(
                        valid_window_sizes,
                        valid_rmse,
                        color=style['color'],
                        marker=style['marker'],
                        linestyle=style['linestyle'],
                        label=style['label'],
                        linewidth=2,
                        markersize=6,
                        alpha=0.8
                    )
        configure_axis(ax1, 'RMSE', 'RMSE vs Window Size', tick_locations)
    
    # Plot MAE if requested
    if 'mae' in axes:
        ax2 = axes['mae']
        for model_name in models_to_plot:
            if model_name in results:
                style = model_styles[model_name]
                mae_values = np.array(results[model_name]['mae'])
                # Filter out NaN values
                valid_mask = ~np.isnan(mae_values)
                if np.any(valid_mask):
                    valid_window_sizes = window_sizes[valid_mask]
                    valid_mae = mae_values[valid_mask]
                    ax2.plot(
                        valid_window_sizes,
                        valid_mae,
                        color=style['color'],
                        marker=style['marker'],
                        linestyle=style['linestyle'],
                        label=style['label'],
                        linewidth=2,
                        markersize=6,
                        alpha=0.8
                    )
        configure_axis(ax2, 'MAE', 'MAE vs Window Size', tick_locations)
    
    # # Add overall title
    # fig.suptitle(
    #     'Effect of input window size on forecasting accuracy.',
    #     fontsize=18,
    #     fontweight='bold',
    #     y=0.98
    # )
    
    
    fig.text(0.5, 0.02,s = '', ha='center', fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    
    # Save or display
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to generate and plot window size effects."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Plot the effect of input window size on forecasting accuracy'
    )
    parser.add_argument(
        '--input-json',
        type=str,
        default=None,
        help='Path to JSON file containing results (if not provided, synthetic data will be used)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='window_size_effect.png',
        help='Output path for the figure (default: window_size_effect.png)'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[10, 3],
        help='Figure size in inches (width height)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=450,
        help='Resolution for saved figure (default: 300)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['both', 'rmse', 'mae'],
        default='both',
        help='Which metric(s) to plot: "both" (default), "rmse" only, or "mae" only'
    )
    parser.add_argument(
        '--no-arima',
        action='store_true',
        help='Exclude ARIMA from the plot'
    )
    
    args = parser.parse_args()
    
    # Load results or generate synthetic data
    if args.input_json:
        json_path = Path(args.input_json)
        results = load_results_from_json(json_path)
        if results is None:
            print(f"Warning: Could not load {json_path}. Using synthetic data.")
            results = generate_synthetic_data()
        else:
            print(f"Loaded results from {json_path}")
    else:
        print("No input JSON provided. Using synthetic data.")
        results = generate_synthetic_data()
    
    # Generate output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    
    # Create the plot
    plot_window_size_effect(
        results,
        output_path=output_path,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        mode=args.mode,
        exclude_arima=args.no_arima
    )


if __name__ == '__main__':
    main()

