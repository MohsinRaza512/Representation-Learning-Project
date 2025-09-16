# -*- coding: utf-8 -*-
"""
Find the best fine-tuned epoch from summary.csv
- Reads summary.csv from prompt_type_comparison_debug
- Computes mean Dice and HD95 per version (Original + each epoch)
- Prints a ranking table
- Generates a plot showing mean Dice across epochs
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
SUMMARY_CSV = "results/prompt_type_comparison_debug/summary.csv"
OUTPUT_DIR = "results/prompt_type_comparison_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load Data
# ---------------------------
if not os.path.isfile(SUMMARY_CSV):
    raise FileNotFoundError(f"❌ Could not find {SUMMARY_CSV}")

df = pd.read_csv(SUMMARY_CSV)

# ---------------------------
# Aggregate by version
# ---------------------------
agg = df.groupby("version")[["dice", "hd95"]].mean().reset_index()

# Sort by mean Dice (descending)
agg_sorted = agg.sort_values("dice", ascending=False)

print("\n📊 Mean Dice & HD95 by Model Version (averaged across modes & thresholds):\n")
print(agg_sorted.to_string(index=False))

# ---------------------------
# Best Fine-tuned Epoch
# ---------------------------
fine_tuned_only = agg_sorted[agg_sorted["version"].str.contains("finetuned")]
if not fine_tuned_only.empty:
    best_row = fine_tuned_only.iloc[0]
    print("\n🏆 Best Fine-tuned Epoch:")
    print(f"  Version   : {best_row['version']}")
    print(f"  Mean Dice : {best_row['dice']:.4f}")
    print(f"  Mean HD95 : {best_row['hd95']:.2f}")
else:
    print("\n⚠️ No fine-tuned epochs found in summary.csv")

# ---------------------------
# Plot mean Dice across epochs
# ---------------------------
# Extract epoch number from version string
def extract_epoch(v):
    if "finetuned_epoch" in v:
        return int(v.replace("finetuned_epoch", ""))
    return None

epochs = []
dice_vals = []
for _, row in agg.iterrows():
    ep = extract_epoch(row["version"])
    if ep is not None:
        epochs.append(ep)
        dice_vals.append(row["dice"])

if epochs:
    plt.figure(figsize=(8,5))
    plt.plot(epochs, dice_vals, marker="o", label="Fine-tuned Dice")
    plt.axhline(
        agg[agg["version"]=="original"]["dice"].values[0],
        color="red", linestyle="--", label="Original baseline"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice (avg over modes & thresholds)")
    plt.title("Fine-tuned Epoch Performance vs Original")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "best_epoch_dice_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n✅ Saved Dice vs Epoch plot: {out_path}")
