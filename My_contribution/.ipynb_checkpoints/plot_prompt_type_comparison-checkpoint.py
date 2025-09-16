# -*- coding: utf-8 -*-
"""
Plot Prompt Type Comparison (Original vs Fine-Tuned Epochs)
- Scans all *_thr* subfolders in results/prompt_type_comparison_heart
- Aggregates mean Dice and HD95 for each prompt type
- Produces bar charts comparing Original and Fine-tuned epochs
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
RESULTS_ROOT = "results/prompt_type_comparison_heart"
OUTPUT_DIR = os.path.join(RESULTS_ROOT, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Collect all results
# -----------------------
all_data = []

for subdir in sorted(os.listdir(RESULTS_ROOT)):
    csv_path = os.path.join(RESULTS_ROOT, subdir, "prompt_type_comparison_results.csv")
    if not os.path.isfile(csv_path):
        continue
    df = pd.read_csv(csv_path)
    df["subdir"] = subdir
    all_data.append(df)

if not all_data:
    raise FileNotFoundError(f"No results found in {RESULTS_ROOT}")

df_all = pd.concat(all_data, ignore_index=True)

# Derive version name from folder (strip _thrX)
df_all["version"] = df_all["subdir"].apply(lambda s: s.split("_thr")[0])

# -----------------------
# Aggregate
# -----------------------
agg = df_all.groupby(["version", "mode"])[["dice", "hd95"]].mean().reset_index()

# -----------------------
# Plotting utility
# -----------------------
def plot_metric(metric: str, ylabel: str, title: str, fname: str):
    modes = ["point", "box", "both"]
    versions = agg["version"].unique()
    x = range(len(modes))

    width = 0.15  # bar width
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, version in enumerate(versions):
        vals = [agg[(agg["version"] == version) & (agg["mode"] == m)][metric].mean()
                for m in modes]
        positions = [p + (i - len(versions)/2) * width for p in x]
        ax.bar(positions, vals, width, label=version)
        for pos, val in zip(positions, vals):
            ax.text(pos, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(modes, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot: {out_path}")

# -----------------------
# Generate plots
# -----------------------
plot_metric("dice", "Mean Dice", "Prompt Type Comparison – Dice", "prompt_type_dice.png")
plot_metric("hd95", "Mean HD95", "Prompt Type Comparison – HD95", "prompt_type_hd95.png")

print(f"\n📊 Plots saved in: {OUTPUT_DIR}")
