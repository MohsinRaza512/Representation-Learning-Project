# -*- coding: utf-8 -*-
"""
Debug Prompt Type Comparison for Fine-Tuned SAM (Fixed)
- Evaluates Original + fine-tuned epochs 1..10
- Tests thresholds [0.2, 0.3, 0.5]
- Computes Dice & HD95 for point/box/both prompts
- Normalizes prompt coords to SAM 1024x1024 space (CRITICAL FIX)
- Saves per-version CSVs and aggregated summary + plots
"""
import os
import csv
import math
import numpy as np
import torch
from tqdm import tqdm
from skimage.measure import regionprops, label
from scipy.spatial.distance import directed_hausdorff
import pandas as pd
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry

# ---------------------------
# Metrics & helpers
# ---------------------------
def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum() + 1e-8
    return float(2.0 * inter / denom)

def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_pts = np.argwhere(pred)
    gt_pts = np.argwhere(gt)
    if pred_pts.size == 0 or gt_pts.size == 0:
        return float("nan")
    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return float(max(d1, d2))

def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    lbl = label(mask.astype(np.uint8))
    if lbl.max() == 0:
        return mask.astype(np.uint8)
    largest = 1 + np.argmax([(lbl == i).sum() for i in range(1, lbl.max() + 1)])
    return (lbl == largest).astype(np.uint8)

def clamp_box(x0, y0, x1, y1, w, h):
    x0, x1 = float(x0), float(x1)
    y0, y1 = float(y0), float(y1)
    x0, x1 = max(0.0, min(x0, w - 1)), max(0.0, min(x1, w - 1))
    y0, y1 = max(0.0, min(y0, h - 1)), max(0.0, min(y1, h - 1))
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    return x0, y0, x1, y1

# ---------------------------
# Config
# ---------------------------
MODEL_TYPE = "vit_b"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use embeddings computed with the (fine-tuned compatible) encoder
NPZ_DIR = "data/precompute_vit_b_ft/test/22_Heart"
OUTPUT_ROOT = "results/prompt_type_comparison_debug"

MODEL_PATHS = {"original": "checkpoints/medsam_box_best_vitb.pth"}
for i in range(1, 11):
    MODEL_PATHS[f"finetuned_epoch{i}"] = f"checkpoints/finetuned_heart/sam_heart_epoch_{i}.pth"

MODES = ["point", "box", "both"]
THRESHOLDS = [0.2, 0.3, 0.5]
SAM_INPUT_SIZE = (1024, 1024)              # (H, W) space for prompts
USE_LCC = True                             # post-processing toggle

# ---------------------------
# Main
# ---------------------------
def main():
    npz_files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files in {NPZ_DIR}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    summary_records = []

    for version, ckpt in MODEL_PATHS.items():
        if not os.path.isfile(ckpt):
            print(f"⚠️  Skipping {version}, checkpoint not found: {ckpt}")
            continue

        print(f"\n🔄 Evaluating {version.upper()} with {ckpt}")
        sam = sam_model_registry[MODEL_TYPE](checkpoint=ckpt)
        sam.to(DEVICE).eval()
        pe = sam.prompt_encoder
        decoder = sam.mask_decoder
        image_pe = pe.get_dense_pe()

        version_rows = []

        with torch.no_grad():
            for fname in tqdm(npz_files):
                data = np.load(os.path.join(NPZ_DIR, fname), allow_pickle=True)
                img_embed = torch.tensor(data["img_embeddings"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                gts = data["gts"]  # (C, H, W)

                for cls_idx in range(gts.shape[0]):
                    gt = gts[cls_idx].astype(np.uint8)
                    if gt.sum() == 0:
                        continue

                    props = regionprops(gt)
                    if not props:
                        continue

                    r = props[0]
                    y0, x0, y1, x1 = r.bbox
                    cy, cx = r.centroid

                    # ---- Normalize to 1024×1024 prompt space (CRITICAL) ----
                    gh, gw = gt.shape
                    H, W = SAM_INPUT_SIZE
                    sx, sy = W / float(gw), H / float(gh)

                    bx0, by0, bx1, by1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
                    bx0, by0, bx1, by1 = clamp_box(bx0, by0, bx1, by1, W, H)
                    if (bx1 - bx0) < 1e-3 or (by1 - by0) < 1e-3:
                        continue  # degenerate

                    boxes_t  = torch.tensor([[bx0, by0, bx1, by1]], dtype=torch.float32, device=DEVICE)
                    px, py   = max(0.0, min(cx * sx, W - 1)), max(0.0, min(cy * sy, H - 1))
                    points_t = torch.tensor([[[px, py]]], dtype=torch.float32, device=DEVICE)
                    p_labels = torch.tensor([[1]], dtype=torch.int64, device=DEVICE)

                    for mode in MODES:
                        if mode == "point":
                            pts, bxs = (points_t, p_labels), None
                        elif mode == "box":
                            pts, bxs = None, boxes_t
                        else:
                            pts, bxs = (points_t, p_labels), boxes_t

                        sparse, dense = pe(points=pts, boxes=bxs, masks=None)
                        low_res_masks, _ = decoder(
                            image_embeddings=img_embed,
                            image_pe=image_pe,
                            sparse_prompt_embeddings=sparse,
                            dense_prompt_embeddings=dense,
                            multimask_output=False,
                        )
                        mask_prob = torch.nn.functional.interpolate(
                            low_res_masks, size=gt.shape, mode="bilinear", align_corners=False
                        ).sigmoid().cpu().numpy()[0, 0]

                        for thr in THRESHOLDS:
                            mask_bin = (mask_prob > thr).astype(np.uint8)
                            if USE_LCC:
                                mask_bin = largest_connected_component(mask_bin)

                            dice = compute_dice(mask_bin, gt)
                            hd   = compute_hd95(mask_bin, gt)
                            if math.isinf(dice): dice = float("nan")
                            if math.isinf(hd):   hd   = float("nan")

                            row = [version, fname, cls_idx, mode, thr, dice, hd]
                            version_rows.append(row)
                            summary_records.append([version, mode, thr, dice, hd])

        # Save per-version CSV
        out_dir = os.path.join(OUTPUT_ROOT, version)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "results.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["version", "file", "class", "mode", "threshold", "dice", "hd95"])
            w.writerows(version_rows)
        print(f"✅ Saved {csv_path}")

    # ---- Save summary + quick means to console ----
    summary_df = pd.DataFrame(summary_records, columns=["version", "mode", "threshold", "dice", "hd95"])
    summary_csv = os.path.join(OUTPUT_ROOT, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n📊 Summary saved to {summary_csv}")

    means = (
        summary_df.groupby(["version", "mode"])[["dice", "hd95"]]
        .mean()
        .reset_index()
        .sort_values(["mode", "version"])
    )
    print("\n📌 Mean metrics across thresholds (by version & prompt type):")
    print(means.to_string(index=False))

    # ---- Plots per threshold ----
    for thr in THRESHOLDS:
        thr_df = summary_df[summary_df["threshold"] == thr]
        pivot = thr_df.groupby(["version", "mode"])[["dice", "hd95"]].mean().reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Dice
        for mode in MODES:
            sub = pivot[pivot["mode"] == mode]
            axes[0].bar(sub["version"], sub["dice"], label=mode, alpha=0.8)
        axes[0].set_title(f"DICE @ thr={thr}")
        axes[0].set_ylim(0, 1.0)
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].legend(title="Prompt Type")

        # HD95 (auto max with headroom)
        for mode in MODES:
            sub = pivot[pivot["mode"] == mode]
            axes[1].bar(sub["version"], sub["hd95"], label=mode, alpha=0.8)
        axes[1].set_title(f"HD95 @ thr={thr}")
        ymax = max(10.0, float(np.nanmax(pivot["hd95"])) * 1.2)
        axes[1].set_ylim(0, ymax)
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        out_png = os.path.join(OUTPUT_ROOT, f"comparison_thr{thr}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"✅ Saved {out_png}")

if __name__ == "__main__":
    main()
