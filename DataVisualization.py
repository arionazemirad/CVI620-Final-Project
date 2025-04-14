import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import random

# Constants
CSV_PATH = 'datasets/driving_log.csv'
IMG_PATH = 'datasets/IMG'

def displayImageAndSteering(index, data):
    """Displays a center image with its steering angle."""
    filename = os.path.basename(data['center'][index])
    img_path = os.path.join(IMG_PATH, filename)
    img = mpimg.imread(img_path)
    steering = data['steering'][index]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original Image - Steering: {steering:.4f}")
    
    # Preprocessing
    img_processed = preProcessing(img)
    
    # Plot preprocessed image (convert YUV back to RGB for visualization)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_processed, cv2.COLOR_YUV2RGB))
    plt.title("Preprocessed Image")
    
    plt.tight_layout()
    plt.savefig(f"sample_image_{index}.png")
    plt.show()

def preProcessing(img):
    """Preprocesses an image (same as in training/testing)."""
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    return img

def balanceDataHistogram(data):
    """Creates a histogram of steering angles to show data balance."""
    plt.figure(figsize=(12, 6))
    plt.hist(data['steering'], bins=50)
    plt.xlabel('Steering Angle')
    plt.ylabel('Count')
    plt.title('Distribution of Steering Angles')
    plt.grid(True)
    plt.savefig("steering_distribution.png")
    plt.show()

def displayCameraAngles(index, data):
    """Displays the three camera angles (left, center, right) for a single frame."""
    center_file = os.path.basename(data['center'][index])
    left_file = os.path.basename(data['left'][index])
    right_file = os.path.basename(data['right'][index])
    
    center_img = mpimg.imread(os.path.join(IMG_PATH, center_file))
    left_img = mpimg.imread(os.path.join(IMG_PATH, left_file))
    right_img = mpimg.imread(os.path.join(IMG_PATH, right_file))
    
    steering = data['steering'][index]
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(left_img)
    plt.title("Left Camera")
    
    plt.subplot(1, 3, 2)
    plt.imshow(center_img)
    plt.title(f"Center Camera - Steering: {steering:.4f}")
    
    plt.subplot(1, 3, 3)
    plt.imshow(right_img)
    plt.title("Right Camera")
    
    plt.tight_layout()
    plt.savefig(f"camera_angles_{index}.png")
    plt.show()

def dataAugmentationDemo(index, data):
    """Demonstrates data augmentation techniques."""
    filename = os.path.basename(data['center'][index])
    img_path = os.path.join(IMG_PATH, filename)
    original_img = mpimg.imread(img_path)
    steering = data['steering'][index]
    
    # Brightness augmentation
    augmenter = iaa.Multiply((0.5, 1.2))
    brightness_img = augmenter.augment_image(original_img.copy())
    
    # Flip augmentation
    flipped_img = cv2.flip(original_img.copy(), 1)
    flipped_steering = -steering
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title(f"Original - Steering: {steering:.4f}")
    
    plt.subplot(1, 3, 2)
    plt.imshow(brightness_img)
    plt.title("Brightness Adjusted")
    
    plt.subplot(1, 3, 3)
    plt.imshow(flipped_img)
    plt.title(f"Flipped - Steering: {flipped_steering:.4f}")
    
    plt.tight_layout()
    plt.savefig(f"augmentation_demo_{index}.png")
    plt.show()

def main():
    # Load data
    print("Loading data...")
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv(CSV_PATH, names=columns)
    
    # Print data summary
    print(f"Total images: {len(data)}")
    print(f"Steering angle range: {data['steering'].min()} to {data['steering'].max()}")
    print(f"Mean steering angle: {data['steering'].mean()}")
    print(f"Median steering angle: {data['steering'].median()}")
    print(f"Steering angle std dev: {data['steering'].std()}")
    
    # Balance check
    zero_steering = len(data[data['steering'] == 0])
    percent_zero = (zero_steering / len(data)) * 100
    print(f"\nZero steering angle frames: {zero_steering} ({percent_zero:.2f}%)")
    
    # Randomly select a few indices to visualize
    random_indices = random.sample(range(len(data)), 3)
    
    # Display data distribution
    balanceDataHistogram(data)
    
    # For each random index, display visualizations
    for idx in random_indices:
        print(f"\nVisualization for sample {idx}:")
        displayImageAndSteering(idx, data)
        displayCameraAngles(idx, data)
        
        try:
            from imgaug import augmenters as iaa
            dataAugmentationDemo(idx, data)
        except ImportError:
            print("imgaug not available, skipping augmentation demo")
    
    print("\nData visualization complete. Check the saved PNG files.")

if __name__ == "__main__":
    main() 