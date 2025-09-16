import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from segment_anything import sam_model_registry
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

# --- Config ---
IMAGE_DIR = "data/train_data/22_Heart/images"
LABEL_DIR = "data/train_data/22_Heart/labels"
CHECKPOINT = "checkpoints/finetune_data/medsam_box_best.pth"  # Your starting SAM-Med checkpoint
SAVE_DIR = "checkpoints/finetuned_heart"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-5

# --- Dataset ---
class HeartSegDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024, 1024))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        img_tensor = self.transform(img).repeat(3, 1, 1)  # Convert to 3-channel
        label_tensor = self.transform(label).squeeze(0)
        return img_tensor, label_tensor

# --- Load Model ---
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(DEVICE)

# --- Freeze strategy ---
# Freeze the entire image encoder so embeddings remain consistent with precomputed NPZs
for name, param in sam.image_encoder.named_parameters():
    param.requires_grad = False

# Allow training of prompt encoder and mask decoder only
for name, param in sam.prompt_encoder.named_parameters():
    param.requires_grad = True

for name, param in sam.mask_decoder.named_parameters():
    param.requires_grad = True


# --- Loss and Optimizer ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, sam.parameters()), lr=LR)

# --- DataLoader ---
dataset = HeartSegDataset(IMAGE_DIR, LABEL_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Training Loop ---
for epoch in range(1, EPOCHS + 1):
    sam.train()
    total_loss = 0

    for images, masks in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Dummy center point for training
        h, w = masks.shape[1:]
        point = torch.tensor([[[w // 2, h // 2]]], device=DEVICE).float()
        point_label = torch.tensor([[1]], device=DEVICE).long()

        with torch.no_grad():
            image_embedding = sam.image_encoder(images)  # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(point, point_label),
            boxes=None,
            masks=None
        )

        image_pe = sam.prompt_encoder.get_dense_pe()
        low_res_logits, _ = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # Resize GT masks to 256x256 to match logits
        masks_resized = nn.functional.interpolate(masks.unsqueeze(1).float(), size=(256, 256), mode='bilinear', align_corners=False).squeeze(1)
        loss = criterion(low_res_logits.squeeze(1), masks_resized)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"✅ Epoch {epoch}: Loss = {total_loss:.4f}")
    torch.save(sam.state_dict(), os.path.join(SAVE_DIR, f"sam_heart_epoch_{epoch}.pth"))

print("✅ Fine-tuning complete.")
