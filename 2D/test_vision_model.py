# evaluate_autoencoder.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Paths
dataset_path = 'datasets/grid_dataset.npy'
model_path = 'results/current/vision_model.h5'

# Load dataset
print("Loading dataset...")
data = np.load(dataset_path)
print("Dataset shape:", data.shape)

# Normalize (if not already normalized)
# In your case it's already 0 and 1 for input.

# Load model
print("Loading model...")
model = load_model(model_path, compile=False)
print("Model loaded.")

# Predict (reconstruct)
print("Generating reconstructions...")
reconstructions = model.predict(data)

# Visualize some inputs and their reconstructions
def plot_comparisons(originals, reconstructions, n=10):
    plt.figure(figsize=(n * 2, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(originals[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis("off")

        # Reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("results/current/reconstruction")

plot_comparisons(data, reconstructions, n=20)
