import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
CSV_PATH = "results/test_prompt_robustness/test_prompt_robustness_results.csv"
OUT_DIR = "results/test_prompt_robustness"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load data ---
df = pd.read_csv(CSV_PATH)

# Remove NaNs if any
df = df.dropna(subset=['dice', 'hd'])

# Group and average results by mode (prompt type), dx, dy
grouped = df.groupby(["mode", "dx", "dy"]).agg({"dice": "mean", "hd": "mean"}).reset_index()

# Save summary table
grouped.to_csv(os.path.join(OUT_DIR, "summary_by_prompt_mode.csv"), index=False)

# --- Visualization ---
for mode in grouped["mode"].unique():
    sub_df = grouped[grouped["mode"] == mode]

    # DICE heatmap
    pivot_dice = sub_df.pivot(index="dy", columns="dx", values="dice")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_dice, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"DICE Heatmap - {mode} Prompt")
    plt.xlabel("dx")
    plt.ylabel("dy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"dice_heatmap_{mode}.png"))
    plt.close()

    # HD95 heatmap
    pivot_hd = sub_df.pivot(index="dy", columns="dx", values="hd")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_hd, annot=True, fmt=".2f", cmap="magma")
    plt.title(f"HD95 Heatmap - {mode} Prompt")
    plt.xlabel("dx")
    plt.ylabel("dy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"hd_heatmap_{mode}.png"))
    plt.close()

print("✅ Heatmaps for each prompt type generated and saved.")
