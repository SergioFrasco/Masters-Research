import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define a function to overlay values on the grid
def overlay_values_on_grid(grid, ax):
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', color='red', fontsize=8)

# Main autoencoder creation
def build_autoencoder(input_shape):
    # Encoder
    inputs = layers.Input(shape=input_shape)
    
    # Calculate padding to ensure output size matches input size
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(inputs)  # Add padding
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Add a cropping layer to get back to the original dimensions
    x = layers.Cropping2D(cropping=((1, 1), (1, 1)))(x)
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = models.Model(inputs, outputs)
    
    # Print the input and output shapes to verify they match
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    
    return autoencoder

def positional_loss(y_true, y_pred):
    # Calculate MSE loss as usual
    mse_loss = tf.square(y_true - y_pred)
    
    # Define a higher weight for rewards being in incorrect positions
    reward_positions = tf.equal(y_true, 1.0)  # Find where the rewards are
    incorrect_rewards = tf.logical_and(reward_positions, tf.not_equal(y_true, y_pred))
    
    # Apply a penalty for incorrect reward positions (higher penalty for errors)
    penalty = tf.cast(incorrect_rewards, tf.float32) * 10.0  # Assigning a penalty weight of 10
    
    # Sum up the loss and penalties
    total_loss = tf.reduce_mean(mse_loss + penalty)
    
    return total_loss


# ------------------------------------------------------------------------------------------------
# Load the dataset from the file
dataset = np.load('grid_dataset.npy')

# Define input shape
input_shape = dataset.shape[1:]
autoencoder = build_autoencoder(input_shape)

# Compile the model
autoencoder.compile(optimizer='adam', loss=positional_loss)

# Train the autoencoder
autoencoder.fit(dataset, dataset, epochs=100, batch_size=32, validation_split=0.1)

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



