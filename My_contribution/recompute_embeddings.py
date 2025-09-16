# -*- coding: utf-8 -*-
"""
Recompute NPZ embeddings using a (fine-tuned) SAM checkpoint.
This ensures that img_embeddings match the encoder weights.

Usage:
    export PYTHONPATH=. && python My_contribution/recompute_embeddings.py

It will:
- Load images + labels from PNGs
- Run them through the SAM encoder
- Save new NPZs with updated img_embeddings + GT masks
"""

import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry

# ---------------------------
# Config
# ---------------------------

MODEL_TYPE = "vit_b"
CHECKPOINT = "checkpoints/finetuned_heart/sam_heart_epoch_10.pth"  # update if needed

# Input raw data (images + labels)
IMAGE_DIR = "data/test_data/22_Heart/images"
LABEL_DIR = "data/test_data/22_Heart/labels"

# Output NPZ directory (new embeddings)
OUT_DIR = "data/precompute_vit_b_ft/test/22_Heart"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load model
# ---------------------------

print(f"🔄 Loading SAM model from {CHECKPOINT}...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(DEVICE)
sam.eval()

# ---------------------------
# Process dataset
# ---------------------------

for fname in tqdm(sorted(os.listdir(IMAGE_DIR))):
    if not fname.endswith(".png"):
        continue

    img_path = os.path.join(IMAGE_DIR, fname)
    lbl_path = os.path.join(LABEL_DIR, fname)

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

    if image is None or label is None:
        print(f"⚠️ Skipping {fname} (missing image or label)")
        continue

    # Convert grayscale -> 3-channel RGB (SAM expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    import torch.nn.functional as F

    # Convert to tensor [1,3,H,W]
    img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(DEVICE)
    
    # Resize to 1024×1024 as required by SAM
    img_tensor = F.interpolate(img_tensor, size=(1024, 1024), mode="bilinear", align_corners=False)


    with torch.no_grad():
        # Run encoder
        img_embeddings = sam.image_encoder(img_tensor)  # [1,256,64,64]

    # Build GT stack (split label into class masks)
    class_values = np.unique(label)
    class_values = class_values[class_values != 0]  # exclude background
    gts = np.stack([(label == val).astype(np.uint8) for val in class_values])

    # Save NPZ
    out_path = os.path.join(OUT_DIR, fname.replace(".png", ".npz"))
    np.savez_compressed(
        out_path,
        img_embeddings=img_embeddings.squeeze(0).cpu().numpy(),
        gts=gts,
        label_except_bk=label,
        image_shape=image.shape,
        resized_size_before_padding=image.shape,
    )

    print(f"✅ Saved: {out_path}")

print("\n🎉 Embedding recomputation complete!")
