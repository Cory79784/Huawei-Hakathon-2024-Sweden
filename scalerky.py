import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Convert features to polar coordinates
def convert_to_polar(H):
    amplitude = np.abs(H)
    phase = np.angle(H)
    return np.concatenate([amplitude, phase], axis=-1)

if __name__ == "__main__":
    print("<<< Start processing Dataset3 data and save the normalizer >>>")

    # File path configuration
    output_dir = r"D:\slice3\slice3-finish"
    slice_num = 20 # Number of slice files in Dataset0
    batch_size = 5000 # Number of samples processed at a time

    # Initialize the normalizer
    scaler = StandardScaler()

    # Initialize an empty list to store the data after batch processing
    X_batches = []

    for slice_idx in tqdm(range(slice_num), desc="Processing Slices"):
        # Load slice file
        slice_file = f'{output_dir}/H_slice_{slice_idx}.npy'
        H_tmp = np.load(slice_file).astype(np.complex64) # Load complex data
        H_polar = convert_to_polar(H_tmp) # Convert to polar coordinates

        # Flatten 4D array to 2D array
        total_samples, port_num, sc_num, ant_num = H_polar.shape
        X_tmp = H_polar.reshape(total_samples, port_num * sc_num * ant_num).astype(np.float32)

        # Standardize data by batch
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            X_batch_scaled = scaler.fit_transform(X_tmp[start:end]) # Normalize the current batch
            X_batches.append(X_batch_scaled) # Store the normalized batch

    # Merge all batches into a complete feature matrix
    X = np.concatenate(X_batches, axis=0)

    # Save the normalizer
    scaler_path = r"D:\slice3\slice3-finish\scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"The normalizer has been successfully saved to: {scaler_path}")
