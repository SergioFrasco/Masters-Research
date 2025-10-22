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
        
        # Calculate normalization statistics for positions
        self._calculate_position_stats()

    def _calculate_position_stats(self):
        """Calculate mean and std for position normalization"""
        all_positions = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            red_x = row.get("red_x", np.nan)
            red_z = row.get("red_z", np.nan)
            blue_x = row.get("blue_x", np.nan)
            blue_z = row.get("blue_z", np.nan)
            
            # Only include valid positions
            if pd.notna(red_x) and pd.notna(red_z):
                all_positions.extend([red_x, red_z])
            if pd.notna(blue_x) and pd.notna(blue_z):
                all_positions.extend([blue_x, blue_z])
        
        if all_positions:
            self.pos_mean = np.mean(all_positions)
            self.pos_std = np.std(all_positions)
            # Avoid division by zero
            if self.pos_std < 1e-6:
                self.pos_std = 1.0
        else:
            self.pos_mean = 0.0
            self.pos_std = 1.0
        
        print(f"Position normalization: mean={self.pos_mean:.2f}, std={self.pos_std:.2f}")

    def normalize_position(self, pos):
        """Normalize position to zero mean and unit variance"""
        return (pos - self.pos_mean) / self.pos_std
    
    def denormalize_position(self, pos_norm):
        """Convert normalized position back to original scale"""
        return pos_norm * self.pos_std + self.pos_mean

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
        # In CubeDataset.__getitem__, change to:
        red_x = row.get("red_dx", np.nan)  # Changed from red_x
        red_z = row.get("red_dz", np.nan)  # Changed from red_z
        blue_x = row.get("blue_dx", np.nan)  # Changed from blue_x
        blue_z = row.get("blue_dz", np.nan)  # Changed from blue_z

        # Convert to array, keeping NaNs for now
        pos = np.array([red_x, red_z, blue_x, blue_z], dtype=np.float32)
        
        # Normalize valid positions, keep 0 for invalid
        pos_normalized = np.zeros(4, dtype=np.float32)
        for i in range(4):
            if not np.isnan(pos[i]):
                pos_normalized[i] = self.normalize_position(pos[i])
        
        pos_tensor = torch.tensor(pos_normalized, dtype=torch.float32)

        return image, torch.tensor(label, dtype=torch.long), pos_tensor


# ============================
# 2. Model
# ============================

class CubeDetector(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        self.classifier = nn.Linear(512, num_classes)
        
        # Improved regression head with additional layers
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        feats = self.features(x).squeeze(-1).squeeze(-1)
        cls_logits = self.classifier(feats)
        pos_preds = self.regressor(feats)
        return cls_logits, pos_preds


# ============================
# 3. Masked regression loss
# ============================

def masked_mse(pred, target, cls_labels):
    """
    Compute MSE loss only for positions that should be predicted
    """
    mask_red = (cls_labels == 1) | (cls_labels == 3)
    mask_blue = (cls_labels == 2) | (cls_labels == 3)

    # Calculate per-coordinate losses
    loss_red = ((pred[:, 0:2] - target[:, 0:2]) ** 2).mean(dim=1)  # Changed sum to mean
    loss_blue = ((pred[:, 2:4] - target[:, 2:4]) ** 2).mean(dim=1)  # Changed sum to mean

    # Apply masks
    loss_red = loss_red * mask_red.float()
    loss_blue = loss_blue * mask_blue.float()

    # Count valid predictions
    num_red = mask_red.float().sum()
    num_blue = mask_blue.float().sum()
    
    # IMPORTANT: Return per-coordinate loss, not per-cube
    total_valid = num_red + num_blue
    
    if total_valid == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    total_loss = (loss_red.sum() + loss_blue.sum()) / total_valid
    
    # Add small epsilon to ensure it's not exactly zero
    return total_loss + 1e-8

# ============================
# 4. Training loop
# ============================

def train_model(csv_path, img_root, num_epochs=10, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    train_ds = CubeDataset(csv_path, img_root, split="train", transform=transform)
    test_ds = CubeDataset(csv_path, img_root, split="test", transform=transform)
    
    # Store normalization stats for later use
    pos_mean = train_ds.pos_mean
    pos_std = train_ds.pos_std

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = CubeDetector().to(device)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0

        # In the training loop, add this debugging:
        # if epoch == 0:  # First epoch
        #     print(f"Sample predictions (normalized): {pos_preds[0].detach().cpu().numpy()}")
        #     print(f"Sample targets (normalized): {pos_labels[0].detach().cpu().numpy()}")
        #     print(f"Sample class labels: {cls_labels[0].item()}")
        
        for imgs, cls_labels, pos_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, cls_labels, pos_labels = imgs.to(device), cls_labels.to(device), pos_labels.to(device)
            cls_logits, pos_preds = model(imgs)

            loss_cls = criterion_cls(cls_logits, cls_labels)
            loss_reg = masked_mse(pos_preds, pos_labels, cls_labels)
            
            # Balance the losses better - increase regression weight
            loss = loss_cls + 2.0 * loss_reg  # Increased from 0.5 to 2.0

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        correct, total = 0, 0
        test_reg_loss = 0
        
        with torch.no_grad():
            for imgs, cls_labels, pos_labels in test_loader:
                imgs, cls_labels, pos_labels = imgs.to(device), cls_labels.to(device), pos_labels.to(device)
                cls_logits, pos_preds = model(imgs)
                
                preds = cls_logits.argmax(dim=1)
                correct += (preds == cls_labels).sum().item()
                total += cls_labels.size(0)
                
                test_reg_loss += masked_mse(pos_preds, pos_labels, cls_labels).item()
        
        val_acc = correct / total
        avg_test_reg_loss = test_reg_loss / len(test_loader)
        
        scheduler.step(avg_train_loss)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} (Cls={avg_cls_loss:.4f}, Reg={avg_reg_loss:.4f}) | "
              f"Test Acc={val_acc:.3f} | Test Reg Loss={avg_test_reg_loss:.4f}")
        
        # Save best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'pos_mean': pos_mean,
                'pos_std': pos_std,
                'epoch': epoch,
            }, "advanced_cube_detector.pth")
            print(f"  → Saved best model")

    print("\n✅ Training complete! Model saved to advanced_cube_detector.pth")


# ============================
# 5. Inference function
# ============================

def load_model_and_predict(model_path, image_path):
    """
    Load trained model and make prediction on a single image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    pos_mean = checkpoint['pos_mean']
    pos_std = checkpoint['pos_std']
    
    # Load model
    model = CubeDetector().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        cls_logits, pos_preds = model(img_tensor)
        
        # Classification
        probs = torch.softmax(cls_logits, dim=1)
        confidence, pred_class = probs.max(dim=1)
        
        label_names = ["None", "Red_Cube", "Blue_Cube", "Both"]
        pred_label = label_names[pred_class.item()]
        
        # Denormalize regression output
        pos_denorm = pos_preds.cpu().numpy()[0] * pos_std + pos_mean
        
        # Round to nearest integer if close enough (within 0.3)
        pos_rounded = np.where(np.abs(pos_denorm - np.round(pos_denorm)) < 0.3, 
                               np.round(pos_denorm), 
                               pos_denorm)
    
    return {
        'label': pred_label,
        'confidence': confidence.item(),
        'regression': pos_rounded.astype(np.float32),
        'regression_raw': pos_denorm.astype(np.float32)
    }


# ============================
# 6. Run training
# ============================

if __name__ == "__main__":
    train_model("dataset/dataset_3d/labels.csv", "dataset/dataset_3d", num_epochs=15, batch_size=32, lr=1e-4)