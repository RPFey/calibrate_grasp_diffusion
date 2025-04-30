import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Set seaborn style for better aesthetics
sns.set(style="whitegrid", context="talk")

def plot_yaxis_from_npz(files):
    """
    Reads 'yaxis' from multiple .npz files and plots them.
    
    Args:
        files (list): List of .npz file paths.
    """
    plt.figure(figsize=(10, 6))  # Set figure size
    for file in files:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue
        
        try:
            data = np.load(file)
            if 'yaxis' in data:
                yaxis = data['yaxis']
                xaxis = data.get('xaxis', np.arange(len(yaxis)))  # Default x-axis if not provided
                base_name = os.path.basename(file)
                plt.plot(xaxis, yaxis, label=base_name.split('-')[0], linewidth=2)
            else:
                print(f"'yaxis' not found in {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Y-axis Value', fontsize=14)
    # Set x limit to 0-1
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Calibration Results', fontsize=16)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig("calibrate.png", dpi=300)  # Save with higher resolution
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 'yaxis' from multiple .npz files.")
    parser.add_argument('files', nargs='+', help="Paths to .npz files")
    args = parser.parse_args()
    
    plot_yaxis_from_npz(args.files)