import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

class CubeDataset(Dataset):
    """Dataset for cube detection"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load cube images (label = 1)
        cube_dir = os.path.join(root_dir, 'cube')
        if os.path.exists(cube_dir):
            for img_name in os.listdir(cube_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(cube_dir, img_name))
                    self.labels.append(1)
        
        # Load not_cube images (label = 0)
        not_cube_dir = os.path.join(root_dir, 'not_cube')
        if os.path.exists(not_cube_dir):
            for img_name in os.listdir(not_cube_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(not_cube_dir, img_name))
                    self.labels.append(0)
        
        print(f"Loaded {len(self.samples)} images:")
        print(f"  - Cube: {sum(self.labels)}")
        print(f"  - Not cube: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CubeDetector(nn.Module):
    """Lightweight CNN for cube detection using MobileNetV2"""
    def __init__(self, pretrained=True):
        super(CubeDetector, self).__init__()
        # Use MobileNetV2 as backbone (lightweight and fast)
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        # Replace final classifier
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, 2)
    
    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=10, device='cuda', save_path='models/cube_detector.pth'):
    """Train the model"""
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)')
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model and show detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    print("\n=== Confusion Matrix ===")
    print("                Predicted")
    print("              Not Cube  Cube")
    print(f"Actual Not Cube   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"       Cube       {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Not Cube', 'Cube']))
    
    # Overall accuracy
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    return accuracy, cm

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✓ Training history saved to {save_path}")

def main():
    # Configuration
    DATA_DIR = 'dataset'
    MODEL_SAVE_PATH = 'models/cube_detector.pth'
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    print("Loading dataset...")
    full_dataset = CubeDataset(DATA_DIR, transform=transform)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = int(VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images\n")
    
    # Calculate class weights for imbalanced dataset
    num_cube = sum(full_dataset.labels)
    num_not_cube = len(full_dataset.labels) - num_cube
    class_weights = torch.tensor([1.0 / num_not_cube, 1.0 / num_cube]).to(device)
    class_weights = class_weights / class_weights.sum()  # Normalize
    
    print(f"Class weights: Not Cube={class_weights[0]:.4f}, Cube={class_weights[1]:.4f}\n")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create model
    print("Creating model...")
    model = CubeDetector(pretrained=True).to(device)
    
    # Loss and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("\nStarting training...\n")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=NUM_EPOCHS, device=device, save_path=MODEL_SAVE_PATH
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # Evaluate on test set
    print("\n=== Final Test Set Evaluation ===")
    test_acc, cm = evaluate_model(model, test_loader, device=device)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best model saved to: {MODEL_SAVE_PATH}")
    print(f"✓ Test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()