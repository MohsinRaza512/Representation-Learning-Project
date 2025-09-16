import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV file generated from prompt_type_comparison.py
CSV_PATH = "results/prompt_type_comparison/prompt_type_comparison_results.csv"
OUT_DIR = "results/prompt_type_comparison"
os.makedirs(OUT_DIR, exist_ok=True)  # Ensure output directory exists

# Read CSV into DataFrame
df = pd.read_csv(CSV_PATH)

# Drop rows where DICE or HD values are missing (in case prediction failed)
df = df.dropna(subset=["dice", "hd"])

# Group by dx, dy, and mode (point, box, both) and calculate mean and std
summary = df.groupby(["dx", "dy", "mode"]).agg({
    "dice": ["mean", "std"],
    "hd": ["mean", "std"]
}).reset_index()

# Rename columns for easier use
summary.columns = ["dx", "dy", "mode", "dice_mean", "dice_std", "hd_mean", "hd_std"]

# Save summary to CSV
summary.to_csv(os.path.join(OUT_DIR, "summary_by_shift.csv"), index=False)

# List of prompt modes
modes = ["point", "box", "both"]

# Loop over each prompt type and generate heatmaps
for mode in modes:
    # Filter the summary for current mode
    df_mode = summary[summary["mode"] == mode]

    # Pivot DICE values for heatmap (dy as rows, dx as columns)
    pivot_dice = df_mode.pivot(index="dy", columns="dx", values="dice_mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_dice, annot=True, fmt=".4f", cmap="viridis")
    plt.title(f"Mean DICE by (dx, dy) Shift - {mode}")
    plt.xlabel("dx")
    plt.ylabel("dy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"dice_heatmap_{mode}.png"))
    plt.close()

    # Pivot HD values for heatmap
    pivot_hd = df_mode.pivot(index="dy", columns="dx", values="hd_mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_hd, annot=True, fmt=".2f", cmap="magma")
    plt.title(f"Mean HD95 by (dx, dy) Shift - {mode}")
    plt.xlabel("dx")
    plt.ylabel("dy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"hd_heatmap_{mode}.png"))
    plt.close()

print("\u2705 Heatmaps for each prompt type saved in:", OUT_DIR)
