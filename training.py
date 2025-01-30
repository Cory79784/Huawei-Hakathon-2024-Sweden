import os
import numpy as np
from matplotlib import pyplot as plt
from Dataset0InputData1.data0try1 import build_improved_model, load_augmented_data_from_file


# The fixed model building function that supports three-dimensional input
def build_improved_model(input_shape):
    from keras.models import Sequential
    from keras.layers import Dense, InputLayer, BatchNormalization, Dropout, Conv1D, Flatten, Reshape
    from keras.regularizers import l2

    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape,)))

    # Convert two-dimensional input to three-dimensional
    model.add(Reshape((input_shape // 128, 128)))  # Assume each sample has 128 features

    # Convolutional layer
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())

    # Add more fully connected layers
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Initialize the model
model = build_improved_model(input_shape=104448)

# Load augmented data in batches and train the model
batch_size = 1000  # Load 1000 samples at a time
epochs_per_batch = 1  # Train one epoch per batch, repeat for multiple cycles
augmented_X_file = r"D:\augmented_data\augmented_X.npy"
augmented_Y_file = r"D:\augmented_data\augmented_Y.npy"

# Outer loop to control the number of times the entire augmented dataset is covered
for epoch in range(10):  # Train for a total of 10 cycles
    print(f"Epoch {epoch + 1}")
    # Load data from disk in batches and train the model
    for X_batch, Y_batch in load_augmented_data_from_file(augmented_X_file, augmented_Y_file, batch_size):
        # Train the model
        history = model.fit(X_batch, Y_batch, epochs=epochs_per_batch, batch_size=32, verbose=1)

# Visualize the training process
plt.plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss vs Validation Loss')
plt.show()

# Save the model
model.save('final_model.h5')  # Save the model to the current directory
print("The model has been saved to 'final_model.h5'")
