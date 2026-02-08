
import json
import matplotlib.pyplot as plt
import os
import pandas as pd

log_file = "/home/guch1/logdir/dreamerv3_origin_bankheist_100k/metrics.jsonl"
output_file = "learning_curve.png"

data = []
try:
    with open(log_file, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
except FileNotFoundError:
    print(f"Error: {log_file} not found.")
    exit(1)

if not data:
    print("No data found in metrics.jsonl")
    exit(1)

df = pd.DataFrame(data)

# Check available columns
print("Available keys:", df.columns.tolist())

# Select relevant metrics
metric_name = "episode/score"
if metric_name not in df.columns:
    if "episode/return" in df.columns:
        metric_name = "episode/return"
    else:
        print(f"Neither episode/score nor episode/return found.")
        metric_name = None

if metric_name:
    # Filter out NaNs for the metric
    df_plot = df.dropna(subset=[metric_name, "step"])
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_plot["step"].values, df_plot[metric_name].values, label=metric_name)
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title(f"Learning Curve: {metric_name}")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
else:
    print("Could not find a score metric to plot.")
