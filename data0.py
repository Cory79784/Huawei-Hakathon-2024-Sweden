import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization, Dropout, Conv1D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

def convert_to_polar(H):
    amplitude = np.abs(H)
    phase = np.angle(H)
    return np.concatenate([amplitude, phase], axis=-1)

def augment_data_to_file(X, Y, augmentation_factor, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    augmented_X_file = os.path.join(output_dir, "augmented_X.npy")
    augmented_Y_file = os.path.join(output_dir, "augmented_Y.npy")

    with open(augmented_X_file, 'wb') as fX, open(augmented_Y_file, 'wb') as fY:
        for i in tqdm(range(len(X)), desc="Augmenting Data"):
            for _ in range(augmentation_factor):
                noise = np.random.normal(0, 0.01, X[i].shape).astype(np.float32)  
                augmented_X = X[i] + noise
                augmented_Y = Y[i]

                np.save(fX, augmented_X)
                np.save(fY, augmented_Y)

    print(f"增强数据已保存到目录: {output_dir}")
    return augmented_X_file, augmented_Y_file

def load_augmented_data_from_file(augmented_X_file, augmented_Y_file, batch_size):
    with open(augmented_X_file, 'rb') as fX, open(augmented_Y_file, 'rb') as fY:
        while True:
            X_batch = []
            Y_batch = []
            try:
                for _ in range(batch_size):
                    X_batch.append(np.load(fX))
                    Y_batch.append(np.load(fY))
                yield np.array(X_batch), np.array(Y_batch)
            except EOFError:
                if X_batch and Y_batch:
                    yield np.array(X_batch), np.array(Y_batch)
                break

def build_improved_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape,)))

    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())

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

if __name__ == "__main__":
    print("<<< 欢迎参加 2024 无线算法大赛！ >>>\n")

    cfg_path = r"D:\data0\Dataset0CfgData1.txt"
    anch_pos_path = r"D:\data0\Dataset0InputPos1.txt"
    output_dir = r"D:\data0"

    slice_num = 20
    batch_size = 5000
    scaler = StandardScaler()

    X_batches = []

    for slice_idx in tqdm(range(slice_num), desc="Processing Slices"):
        slice_file = f'{output_dir}/H_slice_{slice_idx}.npy'
        H_tmp = np.load(slice_file).astype(np.complex64)

        H_polar = convert_to_polar(H_tmp)

        total_samples, port_num, sc_num, ant_num = H_polar.shape
        X_tmp = H_polar.reshape(total_samples, port_num * sc_num * ant_num).astype(np.float32)

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            X_batch_scaled = scaler.fit_transform(X_tmp[start:end])
            X_batches.append(X_batch_scaled)

    X = np.concatenate(X_batches, axis=0)

    X = X.astype(np.float32)

    ground_truth_path = r"D:\data0\Dataset0GroundTruth1.txt"
    ground_truth = np.loadtxt(ground_truth_path, delimiter=' ')
    Y = ground_truth[:, 1:]

    augment_output_dir = r"D:\augmented_data"
    augmented_X_file, augmented_Y_file = augment_data_to_file(X, Y, augmentation_factor=2, output_dir=augment_output_dir)

    print("分批加载增强数据...")
    for X_batch, Y_batch in load_augmented_data_from_file(augmented_X_file, augmented_Y_file, batch_size=1000):
        print(f"加载批次数据，X 形状: {X_batch.shape}, Y 形状: {Y_batch.shape}")

    input_shape = X_batch.shape[1]
    model = build_improved_model(input_shape)

    # 模型训练代码 (需要划分训练集和验证集)
    # model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=..., batch_size=..., callbacks=[early_stopping, reduce_lr])