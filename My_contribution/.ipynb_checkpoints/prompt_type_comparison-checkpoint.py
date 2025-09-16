# -*- coding: utf-8 -*-
"""
Prompt Type Comparison with Multiple Epochs, Thresholds, and Post-processing
- Compares Original SAM vs multiple Fine-tuned epochs
- Evaluates point, box, both prompts
- Sweeps thresholds [0.2, 0.3, 0.5]
- Applies Largest Connected Component post-processing

Fixes vs original:
- Normalize BOTH box and point coordinates to SAM's 1024×1024 input space
- Clamp boxes to valid range; skip degenerate boxes
- Consistently write 'hd95' to CSV
"""

import os
import csv
import math
import numpy as np
import cv2
import torch
from tqdm import tqdm
from skimage.measure import regionprops, label
from scipy.spatial.distance import directed_hausdorff
from segment_anything import sam_model_registry

# ---------------------------
# Metrics
# ---------------------------
def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum() + 1e-8
    return float(2.0 * inter / denom)

def compute_hd95(pred, gt):
    pred_pts = np.argwhere(pred)
    gt_pts = np.argwhere(gt)
    if pred_pts.size == 0 or gt_pts.size == 0:
        return float("nan")
    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return float(max(d1, d2))

def largest_connected_component(mask):
    """Keep only the largest connected component."""
    lbl = label(mask)
    if lbl.max() == 0:
        return mask
    largest = 1 + np.argmax([(lbl == i).sum() for i in range(1, lbl.max() + 1)])
    return (lbl == largest).astype(np.uint8)

def clamp_box(x0, y0, x1, y1, w, h):
    """Clamp and sort xyxy to the valid image range [0,w)×[0,h)."""
    x0, x1 = float(x0), float(x1)
    y0, y1 = float(y0), float(y1)
    x0, x1 = max(0.0, min(x0, w - 1)), max(0.0, min(x1, w - 1))
    y0, y1 = max(0.0, min(y0, h - 1)), max(0.0, min(y1, h - 1))
    # ensure x0<=x1 and y0<=y1
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    return x0, y0, x1, y1

# ---------------------------
# Config
# ---------------------------
MODEL_TYPE = "vit_b"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use *recomputed* embeddings for fine-tuned runs to avoid mismatch with the encoder
NPZ_DIR = "data/precompute_vit_b_ft/test/22_Heart"
OUTPUT_ROOT = "results/prompt_type_comparison_heart"

# Models to evaluate
MODEL_PATHS = {
    "original": "checkpoints/medsam_box_best_vitb.pth",
}
# Add multiple fine-tuned epochs
FINETUNED_EPOCHS = [5, 7, 10]
for ep in FINETUNED_EPOCHS:
    MODEL_PATHS[f"finetuned_epoch{ep}"] = f"checkpoints/finetuned_heart/sam_heart_epoch_{ep}.pth"

# Prompt modes
MODES = ["point", "box", "both"]

# Thresholds to sweep
THRESHOLDS = [0.2, 0.3, 0.5]

# SAM expected input image size for prompts (matches how embeddings were computed)
SAM_INPUT_SIZE = (1024, 1024)  # (H, W)

# ---------------------------
# Main
# ---------------------------
def main():
    npz_files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    if not npz_files:
        raise FileNotFoundError(f"No NPZs found in {NPZ_DIR}")

    for version, ckpt in MODEL_PATHS.items():
        if not os.path.isfile(ckpt):
            print(f"⚠️ Skipping {version}, checkpoint not found: {ckpt}")
            continue

        print(f"\n🔄 Evaluating {version.upper()} with {ckpt}")
        sam = sam_model_registry[MODEL_TYPE](checkpoint=ckpt)
        sam.to(DEVICE).eval()

        pe = sam.prompt_encoder
        decoder = sam.mask_decoder
        image_pe = pe.get_dense_pe()

        for thr in THRESHOLDS:
            print(f"   ➡️ Threshold = {thr}")
            out_dir = os.path.join(OUTPUT_ROOT, f"{version}_thr{thr}")
            os.makedirs(out_dir, exist_ok=True)

            results = []

            with torch.no_grad():
                for fname in tqdm(npz_files):
                    data = np.load(os.path.join(NPZ_DIR, fname), allow_pickle=True)
                    img_embed = torch.tensor(data["img_embeddings"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    gts = data["gts"]

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

                        # For visualization overlays (in GT coordinate space)
                        box_xyxy_vis = [int(x0), int(y0), int(x1), int(y1)]
                        point_xy_vis = [int(cx), int(cy)]

                        # ---- Coordinate normalization to 1024×1024 for SAM prompts ----
                        gh, gw = gt.shape  # GT mask resolution
                        H, W = SAM_INPUT_SIZE
                        scale_x = W / float(gw)
                        scale_y = H / float(gh)

                        # Scale and clamp box
                        bx0 = x0 * scale_x
                        by0 = y0 * scale_y
                        bx1 = x1 * scale_x
                        by1 = y1 * scale_y
                        bx0, by0, bx1, by1 = clamp_box(bx0, by0, bx1, by1, W, H)
                        # Skip degenerate boxes (zero area)
                        if (bx1 - bx0) < 1e-3 or (by1 - by0) < 1e-3:
                            continue

                        boxes_t = torch.tensor([[bx0, by0, bx1, by1]], dtype=torch.float32, device=DEVICE)

                        # Scale and clamp point
                        px = max(0.0, min(cx * scale_x, W - 1))
                        py = max(0.0, min(cy * scale_y, H - 1))
                        points_t = torch.tensor([[[px, py]]], dtype=torch.float32, device=DEVICE)
                        p_labels = torch.tensor([[1]], dtype=torch.int64, device=DEVICE)

                        for mode in MODES:
                            if mode == "point":
                                pts, bxs = (points_t, p_labels), None
                            elif mode == "box":
                                pts, bxs = None, boxes_t
                            else:  # both
                                pts, bxs = (points_t, p_labels), boxes_t

                            # Encode prompts and decode mask
                            sparse, dense = pe(points=pts, boxes=bxs, masks=None)
                            low_res_masks, _ = decoder(
                                image_embeddings=img_embed,
                                image_pe=image_pe,
                                sparse_prompt_embeddings=sparse,
                                dense_prompt_embeddings=dense,
                                multimask_output=False,
                            )

                            # Upsample to GT size, then threshold
                            mask_prob = torch.nn.functional.interpolate(
                                low_res_masks, size=gt.shape, mode="bilinear", align_corners=False
                            ).sigmoid().cpu().numpy()[0, 0]

                            mask_bin = (mask_prob > thr).astype(np.uint8)
                            mask_bin = largest_connected_component(mask_bin)

                            dice = compute_dice(mask_bin, gt)
                            hd = compute_hd95(mask_bin, gt)

                            # Sanity: replace inf with nan (rare numerical edge cases)
                            if math.isinf(dice): dice = float("nan")
                            if math.isinf(hd):   hd = float("nan")

                            results.append([version, fname, cls_idx, mode, thr, dice, hd])

                            # Save qualitative overlay (GT space)
                            vis = (gt * 255).astype(np.uint8)
                            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                            if 0 <= point_xy_vis[1] < vis.shape[0] and 0 <= point_xy_vis[0] < vis.shape[1]:
                                vis[point_xy_vis[1], point_xy_vis[0]] = [0, 255, 0]
                            cv2.rectangle(
                                vis,
                                (box_xyxy_vis[0], box_xyxy_vis[1]),
                                (box_xyxy_vis[2], box_xyxy_vis[3]),
                                (255, 0, 0),
                                1,
                            )
                            pred_bgr = cv2.cvtColor(mask_bin * 255, cv2.COLOR_GRAY2BGR)
                            overlay = cv2.addWeighted(vis, 0.6, pred_bgr, 0.4, 0)
                            out_png = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_class{cls_idx}_{mode}.png")
                            cv2.imwrite(out_png, overlay)

            # Save CSV
            csv_path = os.path.join(out_dir, "prompt_type_comparison_results.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["version", "file", "class", "mode", "threshold", "dice", "hd95"])
                writer.writerows(results)

            print(f"✅ Results saved for {version} @ thr={thr}: {csv_path}")

if __name__ == "__main__":
    main()
