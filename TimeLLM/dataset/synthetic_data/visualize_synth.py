import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_ts(filename, portion_size=608, start_idx=0, save_dir='.'):
    """
    Visualize the entire time series and a specified portion of it.

    Parameters:
    - filename: str, path to the CSV file containing the time series data.
    - portion_size: int, number of time steps to plot in the portion (default: 608).
    - start_idx: int, starting index for the portion to plot (default: 0).
    - save_dir: str, directory to save the PNG files (default: current directory).
    """
    # Load the CSV file
    df = pd.read_csv(filename, parse_dates=['date'])

    # Plot the entire time series
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['synth'])
    plt.title('Overall Time Series')
    plt.xlabel('Date')
    plt.ylabel('Synthetic Value')
    overall_plot_path = os.path.join(save_dir, os.path.basename(filename).replace('.csv', '_overall.png'))
    plt.savefig(overall_plot_path)
    plt.close()

    # Plot the specified portion of the time series
    portion_df = df.iloc[start_idx:start_idx + portion_size]
    plt.figure(figsize=(12, 6))
    plt.plot(portion_df['date'], portion_df['synth'])
    plt.title(f'Portion of Time Series (indices {start_idx} to {start_idx + portion_size - 1})')
    plt.xlabel('Date')
    plt.ylabel('Synthetic Value')
    portion_plot_path = os.path.join(save_dir, os.path.basename(filename).replace('.csv', '_portion.png'))
    plt.savefig(portion_plot_path)
    plt.close()

    print(f"Saved overall plot to {overall_plot_path}")
    print(f"Saved portion plot to {portion_plot_path}")

# Example usage
if __name__ == "__main__":
    # Replace with your desired filename
    visualize_ts('region14.csv')