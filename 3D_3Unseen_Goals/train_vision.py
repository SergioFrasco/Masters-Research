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
        
        # 6 independent binary labels
        self.label_cols = ["red_box", "blue_box", "green_box", "red_sphere", "blue_sphere", "green_sphere"]
        
        # Calculate normalization statistics for positions
        self._calculate_position_stats()

    def _calculate_position_stats(self):
        """Calculate mean and std for position normalization"""
        all_positions = []
        
        position_cols = [
            "red_box_dx", "red_box_dz", 
            "blue_box_dx", "blue_box_dz",
            "green_box_dx", "green_box_dz",
            "red_sphere_dx", "red_sphere_dz",
            "blue_sphere_dx", "blue_sphere_dz",
            "green_sphere_dx", "green_sphere_dz"
        ]
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            for col in position_cols:
                val = row.get(col, np.nan)
                if pd.notna(val):
                    all_positions.append(val)
        
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

        # Multi-label classification: 6 binary labels
        labels = np.zeros(6, dtype=np.float32)
        for i, col in enumerate(self.label_cols):
            labels[i] = float(row.get(col, 0))
        
        # Get positions for all 6 object types (12 values total)
        position_cols = [
            "red_box_dx", "red_box_dz", 
            "blue_box_dx", "blue_box_dz",
            "green_box_dx", "green_box_dz",
            "red_sphere_dx", "red_sphere_dz",
            "blue_sphere_dx", "blue_sphere_dz",
            "green_sphere_dx", "green_sphere_dz"
        ]
        
        pos = np.array([row.get(col, np.nan) for col in position_cols], dtype=np.float32)
        
        # Normalize valid positions, keep 0 for invalid
        pos_normalized = np.zeros(12, dtype=np.float32)
        for i in range(12):
            if not np.isnan(pos[i]):
                pos_normalized[i] = self.normalize_position(pos[i])
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        pos_tensor = torch.tensor(pos_normalized, dtype=torch.float32)

        return image, labels_tensor, pos_tensor


# ============================
# 2. Model
# ============================

class CubeDetector(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Multi-label classification head (6 independent binary predictions)
        self.classifier = nn.Linear(512, num_labels)
        
        # Regression head for 12 position values (6 objects × 2 coordinates each)
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 12)  # red_box, blue_box, green_box, red_sphere, blue_sphere, green_sphere (dx/dz each)
        )

    def forward(self, x):
        feats = self.features(x).squeeze(-1).squeeze(-1)
        cls_logits = self.classifier(feats)  # [batch, 6]
        pos_preds = self.regressor(feats)     # [batch, 12]
        return cls_logits, pos_preds


# ============================
# 3. Masked regression loss
# ============================

def masked_mse(pred, target, labels):
    """
    Compute MSE loss only for positions that should be predicted
    pred: [batch, 12] - predicted positions
    target: [batch, 12] - target positions
    labels: [batch, 6] - binary labels for [red_box, blue_box, green_box, red_sphere, blue_sphere, green_sphere]
    """
    # Create masks for each object type
    mask_red_box = labels[:, 0]       # [batch]
    mask_blue_box = labels[:, 1]      # [batch]
    mask_green_box = labels[:, 2]     # [batch]
    mask_red_sphere = labels[:, 3]    # [batch]
    mask_blue_sphere = labels[:, 4]   # [batch]
    mask_green_sphere = labels[:, 5]  # [batch]
    
    # Calculate per-object losses (mean over dx and dz for each object)
    loss_red_box = ((pred[:, 0:2] - target[:, 0:2]) ** 2).mean(dim=1)         # [batch]
    loss_blue_box = ((pred[:, 2:4] - target[:, 2:4]) ** 2).mean(dim=1)        # [batch]
    loss_green_box = ((pred[:, 4:6] - target[:, 4:6]) ** 2).mean(dim=1)       # [batch]
    loss_red_sphere = ((pred[:, 6:8] - target[:, 6:8]) ** 2).mean(dim=1)      # [batch]
    loss_blue_sphere = ((pred[:, 8:10] - target[:, 8:10]) ** 2).mean(dim=1)   # [batch]
    loss_green_sphere = ((pred[:, 10:12] - target[:, 10:12]) ** 2).mean(dim=1) # [batch]
    
    # Apply masks
    loss_red_box = loss_red_box * mask_red_box
    loss_blue_box = loss_blue_box * mask_blue_box
    loss_green_box = loss_green_box * mask_green_box
    loss_red_sphere = loss_red_sphere * mask_red_sphere
    loss_blue_sphere = loss_blue_sphere * mask_blue_sphere
    loss_green_sphere = loss_green_sphere * mask_green_sphere
    
    # Count valid predictions
    total_valid = (mask_red_box.sum() + mask_blue_box.sum() + mask_green_box.sum() +
                   mask_red_sphere.sum() + mask_blue_sphere.sum() + mask_green_sphere.sum())
    
    if total_valid == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # Average over all valid object predictions
    total_loss = (loss_red_box.sum() + loss_blue_box.sum() + loss_green_box.sum() +
                  loss_red_sphere.sum() + loss_blue_sphere.sum() + loss_green_sphere.sum()) / total_valid
    
    return total_loss + 1e-8


# ============================
# 4. Training loop
# ============================

def train_model(csv_path, img_root, num_epochs=30, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = CubeDataset(csv_path, img_root, split="train", transform=transform)
    test_ds = CubeDataset(csv_path, img_root, split="test", transform=transform)
    
    # Store normalization stats for later use
    pos_mean = train_ds.pos_mean
    pos_std = train_ds.pos_std

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CubeDetector().to(device)
    
    # BCE with logits for multi-label classification
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        
        for imgs, labels, pos_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            pos_labels = pos_labels.to(device)
            
            cls_logits, pos_preds = model(imgs)

            loss_cls = criterion_cls(cls_logits, labels)
            loss_reg = masked_mse(pos_preds, pos_labels, labels)
            
            # Balance the losses
            loss = loss_cls + 2.0 * loss_reg

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
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
        test_reg_loss = 0
        
        # For multi-label: track per-label accuracy
        correct_per_label = np.zeros(6)
        total_per_label = np.zeros(6)
        
        with torch.no_grad():
            for imgs, labels, pos_labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                pos_labels = pos_labels.to(device)
                
                cls_logits, pos_preds = model(imgs)
                
                # Multi-label predictions (threshold at 0.5)
                preds = (torch.sigmoid(cls_logits) > 0.5).float()
                
                # Per-label accuracy
                for i in range(6):
                    correct_per_label[i] += (preds[:, i] == labels[:, i]).sum().item()
                    total_per_label[i] += labels.size(0)
                
                test_reg_loss += masked_mse(pos_preds, pos_labels, labels).item()
        
        # Calculate accuracies
        label_names = ["red_box", "blue_box", "green_box", "red_sphere", "blue_sphere", "green_sphere"]
        acc_str = " | ".join([f"{name}={correct_per_label[i]/total_per_label[i]:.3f}" 
                              for i, name in enumerate(label_names)])
        avg_test_reg_loss = test_reg_loss / len(test_loader)
        
        scheduler.step(avg_train_loss)

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss={avg_train_loss:.4f} (Cls={avg_cls_loss:.4f}, Reg={avg_reg_loss:.4f})")
        print(f"  Test Accuracies: {acc_str}")
        print(f"  Test Reg Loss={avg_test_reg_loss:.4f}")
        
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

def load_model_and_predict(model_path, image_path, threshold=0.5):
    """
    Load trained model and make prediction on a single image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint with weights_only=False for PyTorch 2.6+
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
        
        # Multi-label classification
        probs = torch.sigmoid(cls_logits)
        predictions = (probs > threshold).cpu().numpy()[0]
        
        label_names = ["red_box", "blue_box", "green_box", "red_sphere", "blue_sphere", "green_sphere"]
        detected_objects = [label_names[i] for i in range(6) if predictions[i]]
        
        # Denormalize regression output
        pos_denorm = pos_preds.cpu().numpy()[0] * pos_std + pos_mean
        
        # Round to nearest integer if close enough
        pos_rounded = np.where(np.abs(pos_denorm - np.round(pos_denorm)) < 0.3, 
                               np.round(pos_denorm), 
                               pos_denorm)
        
        # Package positions by object
        positions = {}
        pos_labels = ["red_box", "blue_box", "green_box", "red_sphere", "blue_sphere", "green_sphere"]
        for i, obj in enumerate(pos_labels):
            if predictions[i]:  # Only include positions for detected objects
                positions[obj] = {
                    'dx': float(pos_rounded[i*2]),
                    'dz': float(pos_rounded[i*2 + 1])
                }
    
    return {
        'detected_objects': detected_objects,
        'none': len(detected_objects) == 0,
        'probabilities': {label_names[i]: float(probs[0, i]) for i in range(6)},
        'positions': positions,
        'positions_raw': pos_rounded.astype(np.float32)
    }


# ============================
# 6. Run training
# ============================

if __name__ == "__main__":
    # Train the model
    # train_model("dataset/dataset_3d/labels.csv", "dataset/dataset_3d", num_epochs=30, batch_size=32, lr=1e-4)
    
    # uncomment after training
    result = load_model_and_predict("advanced_cube_detector.pth", "dataset/dataset_3d/test/img_00002.png")
    print(f"\nInference Results:")
    print(f"  Detected objects: {result['detected_objects']}")
    print(f"  None present: {result['none']}")
    print(f"  Probabilities: {result['probabilities']}")
    print(f"  Positions: {result['positions']}")