import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import csv
from skimage.measure import regionprops
from scipy.spatial.distance import directed_hausdorff
from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer

# --- Metric Functions ---
def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)

def compute_hd95(pred, gt):
    pred_pts = np.argwhere(pred)
    gt_pts = np.argwhere(gt)
    if pred_pts.size == 0 or gt_pts.size == 0:
        return np.nan
    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return max(d1, d2)

# --- Config ---
MODEL_TYPE = "vit_b"
CHECKPOINT = "/home/hpc/rlvl/rlvl139v/Segment-Anything-Model-for-Medical-Images/checkpoints/finetune_data/medsam_box_best.pth"
DEVICE = "cuda:0"

# Folder name mapping for each organ
ORGANS = {
    "Heart": "22_Heart",
    "Pancreas": "6_Pancreas",
    "Liver": "3_Liver"
}

DATA_ROOT = "data/precompute_vit_b/test"
SHIFTS = [(0, 0), (5, 0), (-5, 0), (0, 5), (0, -5), (10, 10), (-10, -10)]

# --- Load model ---
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)

# --- Helper ---
def get_box_and_point(mask):
    props = regionprops(mask.astype(np.uint8))
    if len(props) == 0:
        return None, None
    y0, x0, y1, x1 = props[0].bbox
    center = props[0].centroid
    return [x0, y0, x1, y1], [int(center[1]), int(center[0])]

# --- Main Loop for Each Organ ---
for organ, folder in ORGANS.items():
    NPZ_DIR = os.path.join(DATA_ROOT, folder)
    OUT_DIR = f"results/prompt_type_comparison_{organ.lower()}"
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    for fname in sorted(os.listdir(NPZ_DIR)):
        if not fname.endswith(".npz"): continue

        data = np.load(os.path.join(NPZ_DIR, fname), allow_pickle=True)
        image_embedding = torch.tensor(data['img_embeddings']).unsqueeze(0).to(DEVICE)
        label_stack = data['gts']

        for class_index in range(label_stack.shape[0]):
            gt = label_stack[class_index].astype(np.uint8)
            if gt.sum() == 0:
                continue

            base_box, base_point = get_box_and_point(gt)
            if base_box is None or base_point is None:
                continue

            for dx, dy in SHIFTS:
                box = [base_box[0]+dx, base_box[1]+dy, base_box[2]+dx, base_box[3]+dy]
                point = [base_point[0]+dx, base_point[1]+dy]

                for mode in ["point", "box", "both"]:
                    try:
                        pe = PromptEncoder(
                            embed_dim=256,
                            image_embedding_size=(64, 64),
                            input_image_size=(1024, 1024),
                            mask_in_chans=16
                        )
                        transformer = TwoWayTransformer(
                            depth=2,
                            embedding_dim=256,
                            mlp_dim=2048,
                            num_heads=8
                        )
                        decoder = MaskDecoder(
                            num_multimask_outputs=1,
                            transformer=transformer,
                            transformer_dim=256
                        )
                        pe.to(DEVICE)
                        decoder.to(DEVICE)

                        sparse_embeddings, dense_embeddings = pe(
                            points=(torch.tensor([[[point[0], point[1]]]], device=DEVICE).float(),
                                    torch.tensor([[1]], device=DEVICE).long()) if mode != "box" else None,
                            boxes=torch.tensor([box], device=DEVICE).float() if mode != "point" else None,
                            masks=None,
                        )
                        low_res_masks, _ = decoder(
                            image_embeddings=image_embedding,
                            image_pe=pe.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False
                        )
                        masks = torch.nn.functional.interpolate(
                            low_res_masks,
                            size=gt.shape,
                            mode="bilinear",
                            align_corners=False
                        ).sigmoid().detach().cpu().numpy() > 0.5

                        mask_bin = masks[0, 0]

                    except Exception as e:
                        print(f"Error in {fname} class {class_index} shift ({dx},{dy}) mode {mode}: {e}")
                        continue

                    dice = compute_dice(mask_bin, gt)
                    hd = compute_hd95(mask_bin, gt)
                    results.append([organ, fname, class_index, dx, dy, mode, dice, hd])

                    vis = (gt * 255).astype(np.uint8)
                    if vis.ndim == 2:
                        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                    if 0 <= point[1] < vis.shape[0] and 0 <= point[0] < vis.shape[1]:
                        vis[point[1], point[0]] = [0,255,0]
                    cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (255,0,0), 1)
                    pred_overlay = (mask_bin*255).astype(np.uint8)
                    pred_overlay_color = cv2.cvtColor(pred_overlay, cv2.COLOR_GRAY2BGR)
                    overlay = cv2.addWeighted(vis, 0.6, pred_overlay_color, 0.4, 0)
                    cv2.imwrite(os.path.join(OUT_DIR, f"{fname[:-4]}_{class_index}_{dx}_{dy}_{mode}.png"), overlay)

    # --- Save CSV ---
    with open(os.path.join(OUT_DIR, f"prompt_type_comparison_results.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["organ", "file", "class", "dx", "dy", "mode", "dice", "hd"])
        writer.writerows(results)

    print(f"✅ Prompt type comparison test complete for {organ}. Results saved to: {OUT_DIR}")
