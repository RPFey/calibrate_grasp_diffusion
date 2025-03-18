# This script is used to summarize multiple PR curves from different experiments.
# It reads the PR curves from the npz files and plots them in a single figure.
# It also finds the recall value where precision >= target_precision and annotates it on the plot.
# The plot is saved as pr_curve.png in the current directory.
# Usage:
# python pr_summarize.py --src path/to/experiment1 path/to/experiment2 --p 0.9
#

import numpy as np
import matplotlib.pyplot as plt
import argparse

def summarize_pr_curves(pr_curves, target_precision=0.9):
    plt.figure(figsize=(14, 12))

    # Define colors for different models
    colors = ['b', 'g', 'm']

    # Loop over each PR curve
    for (label, (precision, recall)), color in zip(pr_curves.items(), colors):
        # Find recall where precision >= target_precision
        valid_recalls = recall[precision >= target_precision]  # Filter recalls where precision >= 0.9
        
        if len(valid_recalls) > 0:
            target_recall = max(valid_recalls)  # Get the maximum recall
        else:
            target_recall = None  # If no valid recall found

        # Plot PR curve
        plt.plot(recall, precision, linestyle='-', color=color, label=label)

        if target_recall is not None:
            # Draw vertical dashed line at target recall
            plt.axvline(x=target_recall, color=color, linestyle='--')

            # Add text annotation for recall value
            plt.text(target_recall + 0.02, target_precision + 0.02, f"{target_recall:.2f}",
                    color=color, fontsize=8, fontweight='bold')

    # Labels and title
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid()
    plt.savefig('pr_curve.png')
    
if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--src", type=str, nargs='+', required=True, help="Paths to TensorBoard log directories")
    arg.add_argument("--p", type=float, default=0.9, help="Target precision value")
    opt = arg.parse_args()
    
    pr_curves = {}
    for stat_npz in opt.src:
        f = np.load(stat_npz)
        exp_name = stat_npz.split('/')[-2]
        pr_curves[exp_name] = (f['precision'], f['recall'])
    
    summarize_pr_curves(pr_curves, opt.p)
