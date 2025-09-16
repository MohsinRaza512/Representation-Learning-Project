# -*- coding: utf-8 -*-
"""
Extended Plotting for Prompt Type Comparison
- Reads all subfolders from results/prompt_type_comparison_heart/
- Aggregates mean Dice & HD95 across prompt types, thresholds, and epochs
- Produces grouped bar charts for quick comparison
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
RESULTS_ROOT = "results/prompt_type_comparison_heart"
OUTPUT_DIR = os.path.join(RESULTS_ROOT, "plots_extended")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load all results
# ---------------------------
all_data = []

for subdir in sorted(os.listdir(RESULTS_ROOT)):
    csv_path = os.path.join(RESULTS_ROOT, subdir, "prompt_type_comparison_results.csv")
    if not os.path.isfile(csv_path):
        continue
    try:
        df = pd.read_csv(csv_path)
        df["subdir"] = subdir
        all_data.append(df)
    except Exception as e:
        print(f"⚠️ Skipping {csv_path}: {e}")

if not all_data:
    raise FileNotFoundError(f"No result CSVs found in {RESULTS_ROOT}")

df_all = pd.concat(all_data, ignore_index=True)

# ---------------------------
# Parse version/threshold properly
# ---------------------------
def parse_info(subdir):
    if "_thr" in subdir:
        version, thr = subdir.split("_thr")
        try:
            thr = float(thr)
        except:
            thr = None
        return version, thr
    return subdir, None

df_all[["version", "threshold"]] = df_all["subdir"].apply(lambda s: pd.Series(parse_info(s)))

# ---------------------------
# Aggregate metrics
# ---------------------------
agg = df_all.groupby(["version", "threshold", "mode"])[["dice", "hd95"]].mean().reset_index()

# Debug print to verify results
print("\n📊 Aggregated metrics (first 20 rows):")
print(agg.head(20))

# ---------------------------
# Plotting functions
# ---------------------------
def plot_metric(metric, ylabel, fname):
    for mode in agg["mode"].unique():
        subset = agg[agg["mode"] == mode]
        pivot = subset.pivot(index="version", columns="threshold", values=metric)

        plt.figure(figsize=(12, 6))
        pivot.plot(kind="bar", width=0.8, ax=plt.gca())

        plt.title(f"{metric.upper()} by Version/Threshold – {mode} prompts")
        plt.ylabel(ylabel)
        plt.xlabel("Model Version (Original / Fine-tuned Epochs)")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Threshold")

        # Ensure scales make sense
        if metric == "dice":
            plt.ylim(0, 1.0)
        if metric == "hd95":
            plt.ylim(0, max(agg[metric].dropna()) * 1.2)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{fname}_{mode}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved {out_path}")

# ---------------------------
# Generate plots
# ---------------------------
plot_metric("dice", "Mean Dice", "dice_comparison")
plot_metric("hd95", "Mean HD95", "hd95_comparison")

print(f"\n🎉 Extended plots generated in {OUTPUT_DIR}")
