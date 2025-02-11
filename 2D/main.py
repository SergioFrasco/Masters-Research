import sys
sys.path.append(".")
import numpy as np
import matplotlib as plt
from env import SimpleEnv, collect_data
from models import load_trained_autoencoder
from models import overlay_values_on_grid
from models import focal_mse_loss

# Collecting sample images from the environment
# collect_data()

# Running the AE on the collected samples to build a model
# 
def main():
    dataset = np.load("datasets/grid_dataset.npy")
    autoencoder = load_trained_autoencoder()
    autoencoder.compile(optimizer="adam", loss=focal_mse_loss)
    reconstructed = autoencoder.predict(dataset)

    # Visualize the original and reconstructed grids (first sample)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Original grid
    ax1.imshow(dataset[0, ..., 0], cmap='gray')
    ax1.set_title("Original Grid")
    overlay_values_on_grid(dataset[0, ..., 0], ax1)

    # Reconstructed grid
    ax2.imshow(reconstructed[0, ..., 0], cmap='gray')
    ax2.set_title("Reconstructed Grid")
    overlay_values_on_grid(reconstructed[0, ..., 0], ax2)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure to a file (e.g., 'comparison.png')
    plt.savefig('comparison.png')

if __name__ == "__main__":
    main()