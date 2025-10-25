import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ============================
# 1. Dataset
# ============================

class CubeDataset(Dataset):
    def __init__(self, csv_path, img_root, split="train", transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.img_dir = os.path.join(img_root, split)
        self.transform = transform

        self.label_map = {"None": 0, "Red_Cube": 1, "Blue_Cube": 2, "Both": 3}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Assign default label "None" if missing
        row_label = row["label"] if pd.notna(row["label"]) else "None"
        label = self.label_map[row_label]

        # Get positions, defaulting NaN to 0.0
        red_x = row.get("red_x", np.nan)
        red_z = row.get("red_z", np.nan)
        blue_x = row.get("blue_x", np.nan)
        blue_z = row.get("blue_z", np.nan)

        pos = np.nan_to_num([red_x, red_z, blue_x, blue_z], nan=0.0)
        pos = torch.tensor(pos, dtype=torch.float32)

        return image, torch.tensor(label, dtype=torch.long), pos



# ============================
# 2. Model
# ============================

class CubeDetector(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)
        self.regressor = nn.Linear(512, 4)

    def forward(self, x):
        feats = self.features(x).squeeze(-1).squeeze(-1)
        cls_logits = self.classifier(feats)
        pos_preds = self.regressor(feats)
        return cls_logits, pos_preds


# ============================
# 3. Masked regression loss
# ============================

def masked_mse(pred, target, cls_labels):
    mask_red = (cls_labels == 1) | (cls_labels == 3)
    mask_blue = (cls_labels == 2) | (cls_labels == 3)

    loss_red = ((pred[:, 0:2] - target[:, 0:2]) ** 2).sum(dim=1)
    loss_blue = ((pred[:, 2:4] - target[:, 2:4]) ** 2).sum(dim=1)

    loss_red = loss_red * mask_red.float()
    loss_blue = loss_blue * mask_blue.float()

    denom = mask_red.float().sum() + mask_blue.float().sum()
    if denom == 0:
        return torch.tensor(0.0, device=pred.device)
    return (loss_red.sum() + loss_blue.sum()) / denom


# ============================
# 4. Training loop
# ============================

def train_model(csv_path, img_root, num_epochs=10, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_ds = CubeDataset(csv_path, img_root, split="train", transform=transform)
    test_ds = CubeDataset(csv_path, img_root, split="test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = CubeDetector().to(device)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, cls_labels, pos_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, cls_labels, pos_labels = imgs.to(device), cls_labels.to(device), pos_labels.to(device)
            cls_logits, pos_preds = model(imgs)

            loss_cls = criterion_cls(cls_logits, cls_labels)
            loss_reg = masked_mse(pos_preds, pos_labels, cls_labels)
            loss = loss_cls + 0.5 * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, cls_labels, pos_labels in test_loader:
                imgs, cls_labels = imgs.to(device), cls_labels.to(device)
                cls_logits, _ = model(imgs)
                preds = cls_logits.argmax(dim=1)
                correct += (preds == cls_labels).sum().item()
                total += cls_labels.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Test Acc={val_acc:.3f}")

    torch.save(model.state_dict(), "advanced_cube_detector.pth")
    print("\n✅ Training complete! Model saved to advanced_cube_detector.pth")


# ============================
# 5. Run training
# ============================

if __name__ == "__main__":
    train_model("dataset/dataset_3d/labels.csv", "dataset/dataset_3d", num_epochs=10, batch_size=32)
