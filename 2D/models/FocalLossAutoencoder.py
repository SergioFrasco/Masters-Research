import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from utils.plotting import overlay_values_on_grid


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU, forcing CPU usage


def load_trained_autoencoder(model_path="models/autoencoder_model.h5"):
    """Loads and returns the trained autoencoder model."""
    return load_model(model_path, custom_objects={"focal_mse_loss": focal_mse_loss})


# Main autoencoder creation
def build_autoencoder(input_shape):

    '''
    Encoder:

        Conv2D: 32 filters, (3,3), stride 1

        Conv2D: 64 filters, (3,3), stride 1

        Conv2D: 64 filters, (2,2), stride 1 ← New extra layer

    Decoder:

        Conv2DTranspose: 64 filters, (2,2), stride 1 ← Mirror of new layer

        Conv2DTranspose: 64 filters, (3,3), stride 1

        Conv2DTranspose: 32 filters, (3,3), stride 1

        Final Conv2DTranspose to reconstruct output shape
    '''
    # Later
    # TODO override default initialization especially with guadratic loss, small gaussian disc with var 1e-3
    # init decode layers larger than encode layers
    # New Model
    
    # Structure from before meeting
    # # Encoder
    # inputs = layers.Input(shape=input_shape)
    # x = layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(inputs)
    # x = layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    # x = layers.Conv2D(64, (2, 2), strides=1, activation='relu', padding='same')(x)  # <-- Added new layer

    # # Decoder (mirror of encoder)
    # x = layers.Conv2DTranspose(64, (2, 2), strides=1, activation='relu', padding='same')(x)  # <-- Match new layer
    # x = layers.Conv2DTranspose(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    # x = layers.Conv2DTranspose(32, (3, 3), strides=1, activation='relu', padding='same')(x)

    # # CHANGED the structure - From Meeting
    # inputs = layers.Input(shape=input_shape)
    # x1 = layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(inputs)
    # x2 = layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x1)
    # x3 = layers.Conv2D(64, (2, 2), strides=1, activation='relu', padding='same')(x2)  # <-- Added new layer

    # # Decoder (mirror of encoder)
    # x4 = layers.Conv2DTranspose(64, (2, 2), strides=1, activation='relu', padding='same')(x3)  # <-- Match new layer
    # x5 = layers.Conv2DTranspose(64, (3, 3), strides=1, activation='relu', padding='same')(np.dstack([x2,x4]))
    # x6 = layers.Conv2DTranspose(32, (3, 3), strides=1, activation='relu', padding='same')(np.dstack([x1,x5]))
    # # TODO try quadratic loss instead of focal, no activation
    # # Output layer — match input channels (no sigmoid)
    # outputs = layers.Conv2DTranspose(input_shape[-1], (3, 3), activation=None, padding='same')(np.dstack([input,x6]))
    
    # AE Training wasnt being triggered enough without sigmoid
    # outputs = layers.Conv2DTranspose(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(np.dstack([input,x6]))

    # Structure to fix keras errors
    
    inputs = layers.Input(shape=input_shape)
    # Encoder
    x1 = layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(inputs)
    x2 = layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x1)
    x3 = layers.Conv2D(64, (2, 2), strides=1, activation='relu', padding='same')(x2)  # New deeper encoding layer

    # Decoder
    x4 = layers.Conv2DTranspose(64, (2, 2), strides=1, activation='relu', padding='same')(x3)
    x5 = layers.Concatenate(axis=-1)([x2, x4])  # Skip connection
    x5 = layers.Conv2DTranspose(64, (3, 3), strides=1, activation='relu', padding='same')(x5)

    x6 = layers.Concatenate(axis=-1)([x1, x5])  # Skip connection
    x6 = layers.Conv2DTranspose(32, (3, 3), strides=1, activation='relu', padding='same')(x6)

    x7 = layers.Concatenate(axis=-1)([inputs, x6])  # Final merge with input
    outputs = layers.Conv2DTranspose(input_shape[-1], (3, 3), activation=None, padding='same')(x7)


    autoencoder = models.Model(inputs, outputs)

    # print(f"Input shape: {inputs.shape}")
    # print(f"Output shape: {outputs.shape}")

    return autoencoder

def weighted_focal_mse_loss(y_true, y_pred, gamma=2.0, reward_weight=10.0):
    """
    Weighted focal MSE loss that emphasizes reward pixels (1s).
    """
    # Compute standard MSE
    mse_loss = tf.square(y_true - y_pred)
    
    # Add higher weight to reward pixels (1s)
    pixel_weights = 1.0 + (reward_weight - 1.0) * y_true
    
    # Compute the focal weighting term
    focal_weight = tf.pow(1.0 - tf.exp(-mse_loss), gamma)
    
    # Apply both weightings to the MSE loss
    weighted_focal_loss = pixel_weights * focal_weight * mse_loss
    
    return tf.reduce_mean(weighted_focal_loss)

def focal_mse_loss(y_true, y_pred, gamma=2.0):
    """
    Focal MSE loss function for autoencoders.
    
    Parameters:
    - y_true: Ground truth grid.
    - y_pred: Reconstructed grid.
    - gamma: Focusing parameter (higher = more focus on difficult errors).
    
    Returns:
    - Weighted MSE loss that emphasizes harder-to-learn regions.
    """
    # Compute standard MSE
    mse_loss = tf.square(y_true - y_pred)
    
    # Compute the focal weighting term: (1 - exp(-MSE))^gamma
    focal_weight = tf.pow(1.0 - tf.exp(-mse_loss), gamma)
    
    # Apply weighting to the MSE loss
    focal_loss = focal_weight * mse_loss
    
    # Return the mean focal loss
    return tf.reduce_mean(focal_loss)

# ------------------------------------------------------------------------------------------------
def run_autoencoder():
    # Load the dataset from the file
    dataset = np.load('grid_dataset.npy')

    # Define input shape
    input_shape = dataset.shape[1:]
    autoencoder = build_autoencoder(input_shape)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss=focal_mse_loss)

    # Train the autoencoder
    autoencoder.fit(dataset, dataset, epochs=1000, batch_size=32, validation_split=0.1)

    # Evaluate the model on the dataset (use a separate test set if you have one)
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

    # Save the trained model
    autoencoder.save('autoencoder_model.h5')

if __name__ == "__main__":
    run_autoencoder()




