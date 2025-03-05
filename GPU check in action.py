import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import time

# Check GPU availability
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
print("Using GPU:", tf.config.experimental.list_logical_devices('GPU'))

# Generate minimal dummy image data (10 samples, 8x8 grayscale)
x_train = np.random.rand(10, 8, 8, 1).astype(np.float32)
y_train = np.random.randint(0, 2, size=(10,))
y_train = tf.keras.utils.to_categorical(y_train, 2)

# Define the simplest CNN possible
model = Sequential([
    Conv2D(4, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    Flatten(),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model on GPU
start_time = time.time()
with tf.device('/GPU:0'):
    model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=1)

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Confirm GPU usage
print("GPU used during training:", tf.config.experimental.list_logical_devices('GPU'))
