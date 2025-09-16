import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Organ result folders
organs = {
    "heart": "results/test_prompt_robustness_heart/test_prompt_robustness_results.csv",
    "pancreas": "results/test_prompt_robustness_pancreas/test_prompt_robustness_results.csv"
}

for organ, csv_path in organs.items():
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found for {organ}: {csv_path}")
        continue

    print(f"✅ Generating heatmaps for {organ.title()}...")

    df = pd.read_csv(csv_path)

    if not all(col in df.columns for col in ["prompt_type", "dx", "dy", "dice", "hd95"]):
        print(f"❌ Required columns missing in {csv_path}")
        continue

    # Group by prompt type, dx, dy
    grouped = df.groupby(["prompt_type", "dx", "dy"]).agg({
        "dice": "mean",
        "hd95": "mean"
    }).reset_index()

    result_dir = os.path.dirname(csv_path)

    for metric in ["dice", "hd95"]:
        for prompt in ["point", "box", "both"]:
            subset = grouped[grouped["prompt_type"] == prompt]

            if subset.empty:
                continue

            heatmap_data = subset.pivot(index="dy", columns="dx", values=metric)
            heatmap_data = heatmap_data.sort_index(ascending=False)

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".3f",
                cmap="viridis" if metric == "dice" else "magma",
                cbar_kws={"label": metric.upper()}
            )
            plt.title(f"{organ.title()} - {prompt} - {metric}")
            plt.xlabel("X Shift (dx)")
            plt.ylabel("Y Shift (dy)")
            plt.tight_layout()

            out_path = os.path.join(result_dir, f"{metric}_heatmap_{prompt}.png")
            plt.savefig(out_path)
            plt.close()
            print(f"✅ Saved: {out_path}")

print("\n✅ All heatmaps generated.")
