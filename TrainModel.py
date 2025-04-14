import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     BatchNormalization, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau



# constants

CSV_PATH = 'datasets/driving_log.csv'
IMG_PATH = 'datasets/IMG'
MODEL_NAME = 'model.h5'
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
SAMPLES_PER_BIN = 1000
NUM_BINS = 31



# preprocessing

def preprocess_image(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


def augment_image(img_path, steering):
    img = np.copy(mpimg.imread(img_path))

    if np.random.rand() < 0.5:
        img = iaa.Multiply((0.5, 1.2)).augment_image(img)

    if np.random.rand() < 0.5:
        x1, y1 = 0, np.random.randint(0, img.shape[0])
        x2, y2 = img.shape[1], np.random.randint(0, img.shape[0])
        xm, ym = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        mask = np.zeros_like(img[:, :, 1])
        mask[((ym - y1)*(x2 - x1) - (y2 - y1)*(xm - x1)) >= 0] = 1
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(0.2, 0.5)
        img_channel = img[:, :, 0].astype(np.float32)
        img_channel[cond] *= s_ratio
        img[:, :, 0] = img_channel.astype(np.uint8)


    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    if np.random.rand() < 0.5:
        tx = np.random.randint(-20, 20)
        steering += tx * 0.002
        trans_matrix = np.float32([[1, 0, tx], [0, 1, 0]])
        img = cv2.warpAffine(img, trans_matrix, (img.shape[1], img.shape[0]))

    return img, steering


def batch_generator(image_paths, steering_angles, batch_size):
    while True:
        batch_images, batch_steerings = [], []
        for _ in range(batch_size):
            idx = random.randint(0, len(image_paths) - 1)
            img, steer = augment_image(image_paths[idx], steering_angles[idx])
            img = preprocess_image(img)
            batch_images.append(img)
            batch_steerings.append(steer)
        yield np.array(batch_images), np.array(batch_steerings)



# model 

def create_model():
    model = Sequential([
        Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3)),
        BatchNormalization(), Activation('elu'),

        Conv2D(36, (5, 5), strides=(2, 2)),
        BatchNormalization(), Activation('elu'),

        Conv2D(48, (5, 5), strides=(2, 2)),
        BatchNormalization(), Activation('elu'),

        Conv2D(64, (3, 3)),
        BatchNormalization(), Activation('elu'),

        Conv2D(64, (3, 3)),
        BatchNormalization(), Activation('elu'),

        Flatten(),

        Dense(100), BatchNormalization(), Activation('elu'), Dropout(0.5),
        Dense(50), BatchNormalization(), Activation('elu'), Dropout(0.5),
        Dense(10), BatchNormalization(), Activation('elu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model



# data balancing

def load_data():
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    df = pd.read_csv(CSV_PATH, names=columns, skiprows=1)

    image_paths = []
    steering_angles = []

    for i in range(len(df)):
        path = df.iloc[i]['center']
        fname = os.path.basename(path.replace('\\', '/'))
        full_path = os.path.join(IMG_PATH, fname)
        image_paths.append(full_path)
        steering_angles.append(float(df.iloc[i]['steering']))

    return image_paths, steering_angles


def balance_data(image_paths, steering_angles):
    print("[INFO]: Balancing data...")
    hist, bins = np.histogram(steering_angles, NUM_BINS)
    remove_idx = []

    for i in range(NUM_BINS):
        bin_idx = [j for j in range(len(steering_angles)) if bins[i] <= steering_angles[j] <= bins[i+1]]
        if len(bin_idx) > SAMPLES_PER_BIN:
            remove_idx.extend(random.sample(bin_idx, len(bin_idx) - SAMPLES_PER_BIN))

    image_paths = [img for i, img in enumerate(image_paths) if i not in remove_idx]
    steering_angles = [a for i, a in enumerate(steering_angles) if i not in remove_idx]

    print(f"[INFO]: Removed {len(remove_idx)} samples")

    plt.figure(figsize=(10, 5))
    plt.hist(steering_angles, bins=NUM_BINS)
    plt.xlabel('Steering Angle')
    plt.ylabel('Count')
    plt.title('Balanced Data Distribution')
    plt.savefig('balanced_data_distribution.png')

    return image_paths, steering_angles



# training

def train():
    image_paths, steering_angles = load_data()

    plt.figure(figsize=(10, 5))
    plt.hist(steering_angles, bins=NUM_BINS)
    plt.xlabel('Steering Angle')
    plt.ylabel('Count')
    plt.title('Original Data Distribution')
    plt.savefig('original_data_distribution.png')

    image_paths, steering_angles = balance_data(image_paths, steering_angles)

    x_train, x_val, y_train, y_val = train_test_split(
        image_paths, steering_angles, test_size=TEST_SIZE)

    model = create_model()
    model.summary()

    scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    H = model.fit(
        batch_generator(x_train, y_train, BATCH_SIZE),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        validation_data=batch_generator(x_val, y_val, BATCH_SIZE),
        validation_steps=len(x_val) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[scheduler]
    )

    model.save(MODEL_NAME)
    print(f"[INFO]: Model saved as {MODEL_NAME}")

    plt.figure(figsize=(10, 5))
    plt.plot(H.history['loss'], label='Train')
    plt.plot(H.history['val_loss'], label='Validation')
    plt.legend()
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig('loss_curve.png')


if __name__ == '__main__':
    train()
