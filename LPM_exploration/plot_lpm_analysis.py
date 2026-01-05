import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_lpm_analysis(csv_file="lpm_stats.csv", output_dir="plots"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run training first.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading data from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check for required columns
    required_cols = ["step", "actual_error", "pred_error", "intr_reward"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in CSV.")
            return

    # Handle 'is_noisy' if present (for Noisy-TV analysis), otherwise default to False
    if "is_noisy" not in df.columns:
        df["is_noisy"] = False
    else:
        # Convert string 'True'/'False' to boolean if needed
        df["is_noisy"] = df["is_noisy"].astype(str) == "True"

    # 1. Reward over Time
    plt.figure(figsize=(12, 6))
    
    noisy_df = df[df["is_noisy"]]
    clean_df = df[~df["is_noisy"]]

    plt.plot(clean_df["step"].values, clean_df["intr_reward"].values, label="Clean State Reward", color="blue", alpha=0.6, linewidth=1)
    if not noisy_df.empty:
        plt.plot(noisy_df["step"].values, noisy_df["intr_reward"].values, label="Noisy State Reward", color="red", alpha=0.6, linewidth=1)
    
    plt.xlabel("Step")
    plt.ylabel("Intrinsic Reward")
    plt.title("LPM Intrinsic Reward over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "reward_over_time.png"))
    print(f"Saved {output_dir}/reward_over_time.png")
    plt.close()

    # 2. Error Correlation (Predicted vs Actual)
    plt.figure(figsize=(8, 8))
    plt.scatter(clean_df["actual_error"], clean_df["pred_error"], alpha=0.3, label="Clean", color="blue", s=10)
    if not noisy_df.empty:
        plt.scatter(noisy_df["actual_error"], noisy_df["pred_error"], alpha=0.3, label="Noisy", color="red", s=10)
    
    # Draw diagonal line
    max_val = max(df["actual_error"].max(), df["pred_error"].max())
    min_val = min(df["actual_error"].min(), df["pred_error"].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Perfect Prediction")
    
    plt.xlabel("Actual Error (MSE)")
    plt.ylabel("Predicted Error (ErrorPredictor)")
    plt.title("Error Predictor Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "error_correlation.png"))
    print(f"Saved {output_dir}/error_correlation.png")
    plt.close()

    # 3. Error Distribution (Histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(clean_df["actual_error"], bins=50, alpha=0.6, label="Clean States", color="blue", density=True)
    if not noisy_df.empty:
        plt.hist(noisy_df["actual_error"], bins=50, alpha=0.6, label="Noisy States", color="red", density=True)
    
    plt.xlabel("Actual Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.title("Distribution of Reconstruction Errors")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "error_distribution.png"))
    print(f"Saved {output_dir}/error_distribution.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="lpm_stats.csv", help="Path to CSV file")
    parser.add_argument("--out", default="plots", help="Output directory")
    args = parser.parse_args()
    
    plot_lpm_analysis(args.csv, args.out)
