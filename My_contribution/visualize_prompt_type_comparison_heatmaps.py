import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the input/output directories
ORGANS = ["heart", "pancreas"]
BASE_DIR = "results"
FILENAME = "prompt_type_comparison_results.csv"

# Loop through each organ folder
for organ in ORGANS:
    organ_dir = os.path.join(BASE_DIR, f"prompt_type_comparison_{organ}")
    csv_path = os.path.join(organ_dir, FILENAME)

    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        continue

    print(f"✅ Generating heatmaps for {organ.title()}...")

    # Load the results
    df = pd.read_csv(csv_path)

    # Sanity check for required columns
    if not all(col in df.columns for col in ['mode', 'dx', 'dy', 'dice', 'hd']):
        print(f"❌ Required columns missing in {csv_path}")
        continue

    # Loop over prompt types (point, box, both)
    for mode in df["mode"].unique():
        sub_df = df[df["mode"] == mode]

        # --- DICE Heatmap ---
        pivot_dice = sub_df.pivot_table(values="dice", index="dy", columns="dx", aggfunc="mean")
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_dice, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"{organ.title()} - DICE Heatmap ({mode})")
        plt.xlabel("dx")
        plt.ylabel("dy")
        plt.tight_layout()
        plt.savefig(os.path.join(organ_dir, f"dice_heatmap_{mode}.png"))
        plt.close()

        # --- HD95 Heatmap ---
        pivot_hd = sub_df.pivot_table(values="hd", index="dy", columns="dx", aggfunc="mean")
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_hd, annot=True, fmt=".2f", cmap="magma")
        plt.title(f"{organ.title()} - HD95 Heatmap ({mode})")
        plt.xlabel("dx")
        plt.ylabel("dy")
        plt.tight_layout()
        plt.savefig(os.path.join(organ_dir, f"hd95_heatmap_{mode}.png"))
        plt.close()

    print(f"✅ Heatmaps generated and saved for {organ.title()}.\n")

print("✅ All prompt type comparison heatmaps generated.")
