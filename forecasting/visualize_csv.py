#!/usr/bin/env python3
"""
Simple CSV visualization script for time series data.
Usage: python visualize_csv.py <csv_file> [--column value]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def visualize_csv(csv_path, column='value', title=None, show_stats=True):
    """
    Visualize time series data from a CSV file.
    
    Args:
        csv_path: Path to CSV file with timestamp,value columns
        column: Name of the value column to plot
        title: Custom title for the plot
        show_stats: Whether to show statistics
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Print statistics
    if show_stats:
        print(f"\n{'='*60}")
        print(f"CSV File: {csv_path}")
        print(f"{'='*60}")
        print(f"Total rows: {len(df)}")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}")
        
        # Calculate average interval
        time_diffs = df['timestamp'].diff().dropna()
        avg_interval = time_diffs.mean()
        print(f"Average interval: {avg_interval}")
        
        print(f"\nValue statistics:")
        print(f"  Mean: {df[column].mean():.4f}")
        print(f"  Std:  {df[column].std():.4f}")
        print(f"  Min:  {df[column].min():.4f}")
        print(f"  Max:  {df[column].max():.4f}")
        print(f"  NaN:  {df[column].isna().sum()}")
        print(f"{'='*60}\n")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot data
    ax.plot(df['timestamp'], df[column], linewidth=1, alpha=0.8)
    
    # Formatting
    if title is None:
        title = f"Time Series: {Path(csv_path).stem}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel(column.capitalize(), fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    
    # Add statistics text box
    if show_stats:
        stats_text = (
            f"Points: {len(df)}\n"
            f"Mean: {df[column].mean():.4f}\n"
            f"Std: {df[column].std():.4f}\n"
            f"Range: [{df[column].min():.4f}, {df[column].max():.4f}]"
        )
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9,
                family='monospace')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize time series CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_csv.py csv/wg-disk-use.csv
  python visualize_csv.py csv/moderate_uptrend_10day.csv --column value
  python visualize_csv.py csv/my_data.csv --title "My Custom Title" --no-stats
        """
    )
    
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--column', default='value', help='Column name to plot (default: value)')
    parser.add_argument('--title', help='Custom plot title')
    parser.add_argument('--no-stats', action='store_true', help='Hide statistics')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        return 1
    
    visualize_csv(
        args.csv_file,
        column=args.column,
        title=args.title,
        show_stats=not args.no_stats
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
