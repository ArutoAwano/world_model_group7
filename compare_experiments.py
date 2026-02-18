import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import math
import matplotlib.ticker as ticker
import argparse

def load_data(logdir):
    log_file = os.path.join(logdir, "metrics.jsonl")
    data = []
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found.")
        return None
    with open(log_file, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not data:
        return None
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Compare multiple experiments")
    parser.add_argument("--logdirs", nargs="+", help="List of log directories to compare")
    parser.add_argument("--labels", nargs="+", help="Labels for the legend")
    parser.add_argument("--output", type=str, default="comparison_metrics.png", help="Output file name")
    parser.add_argument("--max_step", type=int, default=None, help="Max step to plot")
    args = parser.parse_args()

    if not args.logdirs:
        # Default experiments as requested
        args.logdirs = [
            "logdir/dreamerv3_recon_single_bankheist_400k",
            "logdir/dreamerv3_lpm_bankheist_100k",
            "logdir/dreamerv3_lpm_add_Dquote_bankheist_100k"
        ]
        args.labels = ["Baseline", "LPM Separated", "LPM Integrated"]

    if not args.labels:
        args.labels = [os.path.basename(os.path.normpath(d)) for d in args.logdirs]

    # Define metric groups based on logically related metrics (Total 14 as in reference)
    metric_groups = [
        # Group 1: World Model & Policy (4 plots)
        {
            "title": "World Model & Policy",
            "metrics": [
                {"key": "train/loss/image", "title": "Image Recon Loss (WM)", "log_scale": True},
                {"key": "train/loss/dyn", "title": "Dynamics Loss (WM)", "log_scale": True},
                {"key": "train/loss/rep", "title": "Representation Loss (WM)", "log_scale": True},
                {"key": "train/loss/policy", "title": "Policy Loss (Actor)", "log_scale": True},
            ]
        },
        # Group 2: LPM Performance (5 plots)
        {
            "title": "LPM Performance",
            "metrics": [
                {"key": "train/lpm_delta", "title": "LPM Surprise (Delta)", "log_scale": True},
                {"key": "train/lpm_eps", "title": "LPM Actual Error", "log_scale": True},
                {"key": "train/lpm_pred", "title": "LPM Predicted Error", "log_scale": True},
                {"key": "train/loss/lpm", "title": "LPM Predictor Loss", "log_scale": True},
                {"key": "train/loss/intr", "title": "Intrinsic Model Loss (Surprise)", "log_scale": True},
            ]
        },
        # Group 3: Rewards & Values (4 plots)
        {
            "title": "Rewards & Values",
            "metrics": [
                {"key": "train/loss/rew", "title": "Reward Loss", "log_scale": True},
                {"key": "train/loss/value", "title": "Value Loss (Combined)", "log_scale": True},
                {"key": "train/val", "title": "Value Estimate (Combined)", "log_scale": False},
                {"key": "train/ent/action", "title": "Action Entropy", "log_scale": True},
            ]
        },
        # Group 4: Episode Score (1 plot)
        {
            "title": "Episode Score",
            "metrics": [
                {"key": "episode/score", "title": "Episode Score", "log_scale": False},
            ]
        }
    ]

    # Load all dataframes
    exp_data = []
    for label, logdir in zip(args.labels, args.logdirs):
        df = load_data(logdir)
        if df is not None:
            if args.max_step:
                df = df[df['step'] <= args.max_step]
            exp_data.append({"label": label, "df": df})

    if not exp_data:
        print("No valid data found.")
        return

    # Find the minimum of the maximum steps to align all plots
    max_steps = min(d["df"]["step"].max() for d in exp_data)
    if args.max_step:
        max_steps = min(max_steps, args.max_step)
    
    print(f"Aligning all plots to max step: {max_steps}")
    for data in exp_data:
        data["df"] = data["df"][data["df"]["step"] <= max_steps]

    # Iterate over metric groups and create a figure for each
    output_base, output_ext = os.path.splitext(args.output)
    
    for group_idx, group in enumerate(metric_groups):
        group_metrics = group["metrics"]
        group_title = group["title"]
        
        # Filter metrics that exist in at least one dataframe
        available_metrics = []
        for m in group_metrics:
            if any(m["key"] in d["df"].columns for d in exp_data):
                available_metrics.append(m)

        if not available_metrics:
            continue

        num_plots = len(available_metrics)
        # Choose layout based on number of plots
        if num_plots <= 2: cols, rows = num_plots, 1
        elif num_plots <= 4: cols, rows = 2, 2
        elif num_plots <= 6: cols, rows = 3, 2
        else: cols, rows = 3, math.ceil(num_plots / 3)

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows + 1), squeeze=False)
        fig.suptitle(f"{group_title}", fontsize=20)
        
        axes_flat = axes.flatten()

        colors = plt.cm.tab10(np.linspace(0, 1, len(exp_data)))

        for i, metric in enumerate(available_metrics):
            key = metric["key"]
            ax = axes_flat[i]
            
            for j, data in enumerate(exp_data):
                df = data["df"]
                label = data["label"]
                
                if key in df.columns:
                    df_subset = df.dropna(subset=[key, "step"])
                    if not df_subset.empty:
                        steps = df_subset["step"].values
                        values = df_subset[key].values
                        
                        # Apply smoothing
                        if len(values) > 10:
                            window = min(50, len(values) // 5)
                            if window > 1:
                                smoothed = pd.Series(values).rolling(window=window, min_periods=1, center=True).mean()
                                ax.plot(steps, values, color=colors[j], alpha=0.1)
                                ax.plot(steps, smoothed.values, color=colors[j], label=label if i == 0 else "", linewidth=2)
                            else:
                                ax.plot(steps, values, color=colors[j], label=label if i == 0 else "")
                        else:
                            ax.plot(steps, values, color=colors[j], label=label if i == 0 else "")

            ax.set_title(metric["title"], fontsize=14)
            if metric["log_scale"]:
                # Use symlog to handle potentially very small or non-positive values gracefully
                ax.set_yscale('symlog', linthresh=1e-4)
                # Ensure the view includes at least a reasonable range
                y_min, y_max = ax.get_ylim()
                ax.set_ylim(max(y_min, 1e-6) if y_min > 0 else None, y_max * 1.1)
            ax.grid(True, alpha=0.3)
            
            def k_formatter(x, pos):
                return f'{int(x/1000)}k' if x >= 1000 else f'{x}'
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))

        # Hide unused subplots
        for j in range(num_plots, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        # Shared xlabel and legend with adjusted positions to avoid overlap
        fig.text(0.5, 0.08, 'Step', ha='center', fontsize=14)
        
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=len(exp_data), 
                       bbox_to_anchor=(0.5, 0.02), fontsize=12)

        # Use subplots_adjust for finer control to avoid title/label clipping
        plt.tight_layout(rect=[0, 0.12, 1, 0.95])
        
        group_output = f"{output_base}_{group_idx + 1}{output_ext}"
        plt.savefig(group_output)
        print(f"Saved {group_title} to {group_output}")
        plt.close(fig)


if __name__ == "__main__":
    main()
