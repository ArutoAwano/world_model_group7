
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def plot_metrics(logdir):
    log_file = os.path.join(logdir, "metrics.jsonl")
    output_file = "debug_metrics.png"
    
    data = []
    if not os.path.exists(log_file):
        print(f"File not found: {log_file}")
        return

    with open(log_file, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    if not data:
        print("No data")
        return

    df = pd.DataFrame(data)
    
    # Filter for relevant keys
    keys = [
        "episode/score",
        "train/loss/value_ext", "train/loss/value_intr",
        "train/val_ext", "train/val_intr",
        "train/rew_ext", "train/rew_intr",
        "train/lpm_delta", "train/lpm_eps", "train/lpm_pred"
    ]
    
    available_keys = [k for k in keys if k in df.columns]
    
    n = len(available_keys)
    cols = 3
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    for i, key in enumerate(available_keys):
        ax = axes[i]
        subset = df.dropna(subset=[key, "step"])
        if not subset.empty:
            ax.plot(subset["step"].values, subset[key].values)
            ax.set_title(key)
            ax.grid(True)
            if "loss" in key:
                ax.set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    args = parser.parse_args()
    plot_metrics(args.logdir)
