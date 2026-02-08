
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Plot detailed training metrics from metrics.jsonl")
parser.add_argument("--logdir", type=str, required=True, help="Path to the log directory containing metrics.jsonl")
args = parser.parse_args()

log_file = os.path.join(args.logdir, "metrics.jsonl")
output_file = "detailed_metrics.png"
print(f"Reading from: {log_file}")

# Metric groups to plot
metrics_to_plot = [
    # World Model
    {"key": "train/loss/image", "title": "Image Recon Loss (WM)", "group": "World Model"},
    {"key": "train/loss/dyn", "title": "Dynamics Loss (WM)", "group": "World Model"},
    {"key": "train/loss/rep", "title": "Representation Loss (WM)", "group": "World Model"},
    
    # Actor-Critic
    {"key": "train/loss/policy", "title": "Policy Loss (Actor)", "group": "Actor-Critic"},
    {"key": "train/loss/value", "title": "Value Loss (Critic)", "group": "Actor-Critic"},
    {"key": "train/ent/action", "title": "Action Entropy", "group": "Actor-Critic"},
    
    # Performance & Value
    {"key": "episode/score", "title": "Episode Score", "group": "Performance"},
    {"key": "train/val", "title": "Value Estimate (train/val)", "group": "Performance"},
    {"key": "train/loss/rew", "title": "Reward Loss", "group": "World Model"},
]

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
    print("No data found.")
    exit(1)

df = pd.DataFrame(data)

# Create a figure with subplots (3x3 grid)
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle("DreamerV3 Training Metrics", fontsize=16)
axes = axes.flatten()

for i, item in enumerate(metrics_to_plot):
    key = item["key"]
    title = item["title"]
    ax = axes[i]
    
    if key in df.columns:
        # Check for non-null data
        df_subset = df.dropna(subset=[key, "step"])
        if not df_subset.empty:
            # Use .values to avoid pandas/matplotlib version issues
            ax.plot(df_subset["step"].values, df_subset[key].values)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3, which="both", ls="-") # Grid for both major and minor ticks
            
            # Apply Log Scale for Losses
            if "loss" in key or "ent" in key:
                ax.set_yscale('log')
                
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.set_title(title)
    else:
        ax.text(0.5, 0.5, f"Key not found:\n{key}", ha='center', va='center', color='red')
        ax.set_title(title)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_file)
print(f"Saved detailed metrics to {os.path.abspath(output_file)}")
