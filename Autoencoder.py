import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

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
    
    # Add assertions to verify shapes
    tf.debugging.assert_equal(tf.shape(inputs)[1:3], tf.shape(outputs)[1:3], 
                            message="Output spatial dimensions don't match input dimensions")
    
    autoencoder = models.Model(inputs, outputs)
    return autoencoder

# ------------------------------------------------------------------------------------------------
# Load the dataset from the file
dataset = np.load('grid_dataset.npy')

# Define input shape
input_shape = dataset.shape[1:]
autoencoder = build_autoencoder(input_shape)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(dataset, dataset, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate the model on the dataset (use a separate test set if you have one)
reconstructed = autoencoder.predict(dataset)

# print the first reconstructed grid
import matplotlib.pyplot as plt

# Visualize the original and reconstructed grids (first sample)
plt.subplot(1, 2, 1)
plt.imshow(dataset[0, ..., 0], cmap='gray')
plt.title("Original Grid")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed[0, ..., 0], cmap='gray')
plt.title("Reconstructed Grid")

plt.show()

# Save the trained model
autoencoder.save('autoencoder_model.h5')



