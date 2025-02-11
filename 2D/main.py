import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import absl.logging
import tensorflow as tf

from env import SimpleEnv, collect_data
from models import load_trained_autoencoder
from models import focal_mse_loss
from utils.plotting import overlay_values_on_grid, visualize_sr
from models.construct_sr import constructSR


# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Disable GPU if not needed
tf.config.set_visible_devices([], "GPU")

# Suppress absl warnings
absl.logging.set_verbosity(absl.logging.ERROR)
sys.path.append(".")


def main():

     # Collecting sample images from the environment
    # collect_data()
    
    # --------------------- Vision Based Reward Model ------------------------
  
    # dataset = np.load("datasets/grid_dataset.npy")
    # autoencoder = load_trained_autoencoder()
    # autoencoder.compile(optimizer="adam", loss=focal_mse_loss)
    # reconstructed = autoencoder.predict(dataset)

    # # Visualize the original and reconstructed grids (first sample)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # # Original grid
    # ax1.imshow(dataset[0, ..., 0], cmap='gray')
    # ax1.set_title("Original Grid")
    # overlay_values_on_grid(dataset[0, ..., 0], ax1)

    # # Reconstructed grid
    # ax2.imshow(reconstructed[0, ..., 0], cmap='gray')
    # ax2.set_title("Reconstructed Grid")
    # overlay_values_on_grid(reconstructed[0, ..., 0], ax2)

    # # Adjust layout for better spacing
    # plt.tight_layout()

    # # Save the figure to a file (e.g., 'comparison.png')
    # plt.savefig('results/comparison.png')

    # ------------------Successor Representation Map------------------
    # Retrieve the specific SR for the Given environment

    # Build the SR map for the given Environment (Saved to results)
    # constructSR()
    visualize_sr()
    



if __name__ == "__main__":
    main()