# This script converts raw PNG image and label files into .npz format
# compatible with SAM-Med experiment scripts

import os
import numpy as np
import cv2

# Set paths to your data folders (adjust if needed)
IMAGE_DIR = "data/test_data/6_Pancreas/images"
LABEL_DIR = "data/test_data/6_Pancreas/labels"
OUT_DIR = "data/precompute_vit_b/test/6_Pancreas"

# Make sure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Loop through all PNG files in image directory
for fname in sorted(os.listdir(IMAGE_DIR)):
    if not fname.endswith(".png"):
        continue

    image_path = os.path.join(IMAGE_DIR, fname)
    label_path = os.path.join(LABEL_DIR, fname)

    # Load image and label as grayscale arrays
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    if image is None or label is None:
        print(f"Skipping {fname} (missing image or label)")
        continue

    # Dummy image embedding for now (will be replaced during real experiment)
    img_embeddings = np.zeros((256, 64, 64), dtype=np.float32)

    # Expand label into a fake gts stack
    # For SAM-Med, different classes were separated; here we use 3 values arbitrarily
    class_values = [85, 170, 255]
    gts = np.stack([(label == val).astype(np.uint8) for val in class_values])

    # Save to NPZ file
    out_path = os.path.join(OUT_DIR, fname.replace(".png", ".npz"))
    np.savez_compressed(out_path,
                        img_embeddings=img_embeddings,
                        gts=gts,
                        label_except_bk=label,
                        image_shape=image.shape,
                        resized_size_before_padding=image.shape)

    print(f"✅ Saved: {out_path}")

print("\n✅ Conversion complete!")
