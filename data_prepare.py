import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
DATASET_PATH = r"C:\Users\prasa\Documents\ec_it\organized_dataset"  # Input path
PROCESSED_PATH = r"C:\Users\prasa\Documents\ec_it\processed_dataset"  # Output path

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  
    rotation_range=15, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,
    fill_mode="nearest"
)

# Function to process and augment images
def process_images(data_split):
    input_dir = os.path.join(DATASET_PATH, data_split)
    output_dir = os.path.join(PROCESSED_PATH, data_split)
    os.makedirs(output_dir, exist_ok=True)

    categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for category in categories:
        img_dir = os.path.join(input_dir, category)
        save_dir = os.path.join(output_dir, category)
        os.makedirs(save_dir, exist_ok=True)

        images = [img for img in os.listdir(img_dir) if img.endswith((".jpg", ".png", ".jpeg"))]

        for img_name in tqdm(images, desc=f"Processing {len(images)} images in category: {category}"):
            img_path = os.path.join(img_dir, img_name)
            img_arr = cv2.imread(img_path)

            if img_arr is None:
                print(f"Warning: Failed to load image {img_name}")
                continue

            img_resized = cv2.resize(img_arr, (224, 224))
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, img_resized)

            # Apply Augmentation and save extra images
            img_array = np.expand_dims(img_resized, 0)
            for i, aug_img in enumerate(datagen.flow(img_array, batch_size=1)):
                aug_save_path = os.path.join(save_dir, f"aug_{i}_{img_name}")
                cv2.imwrite(aug_save_path, (aug_img[0] * 255.0).astype(np.uint8))  # ✅ Fix applied
                if i >= 4:  # Save 5 augmented images per original
                    break

# Apply processing on all dataset splits
for split in ["train", "valid", "test"]:
    process_images(split)

print("✅ Dataset successfully preprocessed and augmented!")
