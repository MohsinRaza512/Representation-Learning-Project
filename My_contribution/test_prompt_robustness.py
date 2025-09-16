# This script tests prompt shift robustness for Heart and Pancreas using SAM-Med

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

# --- Configuration ---
MODEL_TYPE = "vit_b"
CHECKPOINT = "/home/hpc/rlvl/rlvl139v/Segment-Anything-Model-for-Medical-Images/checkpoints/finetune_data/medsam_box_best.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SHIFTS = [(0, 0), (5, 0), (-5, 0), (0, 5), (0, -5), (10, 10), (-10, -10)]

# Define organs and data paths (Heart + Pancreas only)
ORGANS = {
    "heart": "data/precompute_vit_b/test/22_Heart",
    "pancreas": "data/precompute_vit_b/test/6_Pancreas"
}

# Load and map model checkpoint
from segment_anything import sam_model_registry
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(DEVICE)

# --- Loop through each organ ---
for organ_name, data_dir in ORGANS.items():
    output_dir = f"results/test_prompt_robustness_{organ_name}"
    os.makedirs(output_dir, exist_ok=True)
    results = []

    if not os.path.exists(data_dir):
        print(f"❌ Skipping {organ_name}: directory not found.")
        continue

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue

        try:
            data = np.load(os.path.join(data_dir, fname), allow_pickle=True)
            image_embedding = torch.tensor(data['img_embeddings']).unsqueeze(0).to(DEVICE)
            label_stack = data['gts']
        except Exception as e:
            print(f"⚠️ Skipping file {fname}: {e}")
            continue

        for class_index in range(label_stack.shape[0]):
            gt = label_stack[class_index].astype(np.uint8)
            if gt.sum() == 0:
                continue

            props = regionprops(gt.astype(np.uint8))
            if len(props) == 0:
                continue
            y0, x0, y1, x1 = props[0].bbox
            center = props[0].centroid
            base_box = [x0, y0, x1, y1]
            base_point = [int(center[1]), int(center[0])]

            for dx, dy in SHIFTS:
                box = [base_box[0]+dx, base_box[1]+dy, base_box[2]+dx, base_box[3]+dy]
                point = [base_point[0]+dx, base_point[1]+dy]

                for mode in ["point", "box", "both"]:
                    try:
                        # Build PromptEncoder and Decoder
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
                            num_heads=8  # 8 divides 256
                        )
                        decoder = MaskDecoder(
                            num_multimask_outputs=1,
                            transformer=transformer,
                            transformer_dim=256
                        )
                        pe.to(DEVICE)
                        decoder.to(DEVICE)

                        # Prepare inputs based on mode
                        point_coords = torch.tensor([[[point[0], point[1]]]], device=DEVICE).float()
                        point_labels = torch.tensor([[1]], device=DEVICE).long()
                        boxes = torch.tensor([box], device=DEVICE).float()

                        # Choose prompts
                        use_point = mode in ["point", "both"]
                        use_box = mode in ["box", "both"]

                        points = (point_coords, point_labels) if use_point else None
                        boxes_input = boxes if use_box else None

                        # Encode prompts
                        sparse_embeddings, dense_embeddings = pe(
                            points=points,
                            boxes=boxes_input,
                            masks=None
                        )

                        # Decode
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
                        dice = compute_dice(mask_bin, gt)
                        hd = compute_hd95(mask_bin, gt)

                        results.append([fname, class_index, dx, dy, mode, dice, hd])

                        # Visualization
                        vis = (gt * 255).astype(np.uint8)
                        if vis.ndim == 2:
                            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                        if 0 <= point[1] < vis.shape[0] and 0 <= point[0] < vis.shape[1]:
                            vis[point[1], point[0]] = [0, 255, 0]
                        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
                        pred_overlay = (mask_bin * 255).astype(np.uint8)
                        overlay = cv2.addWeighted(vis, 0.6, cv2.cvtColor(pred_overlay, cv2.COLOR_GRAY2BGR), 0.4, 0)
                        cv2.imwrite(os.path.join(output_dir, f"{fname[:-4]}_{class_index}_{dx}_{dy}_{mode}.png"), overlay)

                    except Exception as e:
                        print(f"Error in {fname} class {class_index} shift ({dx},{dy}) mode {mode}: {e}")
                        continue

    # Save results as CSV
    csv_path = os.path.join(output_dir, "test_prompt_robustness_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file", "class", "dx", "dy", "prompt_type", "dice", "hd95"])
        writer.writerows(results)

    print(f"✅ Prompt robustness test complete for {organ_name.title()}. Results saved to: {output_dir}")
