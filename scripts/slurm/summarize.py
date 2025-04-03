# This script is used to summarize multiple tensorboard logs from different experiments.
# It reads the logs from the tensorboard log directories and plots the metric in a single figure.
# The plot is saved as <metric_name>.png in the current directory.
# Usage:
# python summarize.py --metric_name loss --src path/to/experiment1/summarizes path/to/experiment2/summarizes
#


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_metric_from_events(log_dir, metric_name):
    metric_data = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                file_path = os.path.join(root, file)
                ea = event_accumulator.EventAccumulator(file_path)
                ea.Reload()
                
                if metric_name in ea.Tags()["scalars"]:
                    for event in ea.Scalars(metric_name):
                        metric_data.append((event.step, event.value))
    return metric_data

def plot_metrics(src_dirs, metric_name, smoothing=0):
    plt.figure(figsize=(10, 6))
    
    for log_dir in src_dirs:
        experiment_name = log_dir.strip('/').split("/")[-2]
        metric_values = extract_metric_from_events(log_dir, metric_name)
        
        if metric_values:
            steps, values = zip(*sorted(metric_values))
            
            if smoothing > 1:
                # do 1-d smoothing for valus
                # 1-d convolution
                kernel = np.ones(smoothing) / smoothing
                values = np.convolve(values, kernel, mode='same')
                
            plt.plot(steps, values, label=experiment_name)
        else:
            print(f"No data found for {metric_name} in {log_dir}")
    
    plt.xlabel("Steps")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Time")
    plt.legend()
    plt.grid()
    plt.savefig(f"{metric_name}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics from TensorBoard logs.")
    parser.add_argument("--metric_name", type=str, required=True, help="Name of the metric to plot")
    parser.add_argument("--src", type=str, nargs='+', required=True, help="Paths to TensorBoard log directories")
    parser.add_argument("--smoothing", type=int, default=1, help="Smoothing factor for the plot")
    args = parser.parse_args()
    
    plot_metrics(args.src, args.metric_name, args.smoothing)