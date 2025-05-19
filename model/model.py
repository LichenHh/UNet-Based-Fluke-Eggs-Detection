import os
import json
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Map folder names to class indices
CLASS_MAP = {
    'Clonorchis sinensis egg': 0,
    'Fasciolopsis buski eggs': 1,
    'Paragonimus westermani eggs': 2,
    'Schistosoma eggs': 3
}

class EggDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.samples = []
        for cls_name, cls_idx in CLASS_MAP.items():
            folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(folder):
                continue
            for img_path in glob(os.path.join(folder, '*.*')):
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    base = os.path.splitext(img_path)[0]
                    json_path = base + '.json'
                    if os.path.isfile(json_path):
                        self.samples.append((img_path, json_path, cls_idx))
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in '{root_dir}'. Please check your data directory and class folder names.")
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, cls_idx = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        # load bbox
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        label = data['labels'][0]
        x1, y1, x2, y2 = label['x1'], label['y1'], label['x2'], label['y2']
        # create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = cls_idx + 1  # classes 1-4

        if self.transforms:
            augmented = self.transforms(image=np.array(img), mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            img = T.ToTensor()(img)
            mask = torch.from_numpy(mask).long()

        return img, mask

# Define transforms (using albumentations)
import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

# Initialize dataset and data loader
root_dir = r'D:\UNet-Based-Fluke-Eggs-Detection\data'  # change if your data path is different
dataset = EggDataset(root_dir=root_dir, transforms=transforms)
print(f"Found {len(dataset)} samples for training.")

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model: UNet with pretrained encoder
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=5  # 0 background + 4 classes
)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    epoch_loss = 0.0
    for batch_idx, (imgs, masks) in enumerate(loop, start=1):
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=epoch_loss / batch_idx)

# Evaluation on training set
model.eval()
ious = {c: [] for c in range(1, 5)}
with torch.no_grad():
    for imgs, masks in tqdm(dataloader, desc='Eval'):
        imgs = imgs.to(device)
        masks = masks.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        for c in range(1, 5):
            pred_c = (preds == c)
            true_c = (masks == c)
            inter = (pred_c & true_c).sum().item()
            uni = (pred_c | true_c).sum().item()
            if uni > 0:
                ious[c].append(inter / uni)

print("Class IoUs:")
for c, vals in ious.items():
    mean_iou = np.nan if not vals else sum(vals) / len(vals)
    print(f"Class {c}: Mean IoU = {mean_iou:.4f}")

# Extract bounding boxes from predicted masks
import cv2

def mask_to_bbox(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, x+w, y+h

# Show some sample results
for idx in range(min(5, len(dataset))):
    img, mask = dataset[idx]
    with torch.no_grad():
        out = model(img.unsqueeze(0).to(device))
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
    bbox = mask_to_bbox(pred == (mask.numpy()))
    print(f"Sample {idx}: True bbox = {mask.numpy().nonzero()}, Pred bbox = {bbox}")
