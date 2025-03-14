import os
import shutil
import random

# Define paths
DATASET_PATH = r"C:\Users\prasa\Documents\ec_it\dataset"  # Source dataset (current image folders)
OUTPUT_PATH = r"C:\Users\prasa\Documents\ec_it\organized_dataset"  # Target directory for train/test/valid splits

# Create split directories
for split in ["train", "test", "valid"]:
    os.makedirs(os.path.join(OUTPUT_PATH, split), exist_ok=True)

# Get all component categories (folders)
categories = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

# Split and move images
for category in categories:
    category_path = os.path.join(DATASET_PATH, category)
    images = [f for f in os.listdir(category_path) if f.endswith((".jpg", ".png", ".jpeg"))]

    # Shuffle images
    random.shuffle(images)

    # Split dataset
    train_split = int(0.8 * len(images))
    valid_split = int(0.9 * len(images))

    splits = {
        "train": images[:train_split],
        "valid": images[train_split:valid_split],
        "test": images[valid_split:]
    }

    # Create category folders inside train/test/valid
    for split in splits.keys():
        os.makedirs(os.path.join(OUTPUT_PATH, split, category), exist_ok=True)

    # Move images to respective folders
    for split, img_files in splits.items():
        for img in img_files:
            src_path = os.path.join(category_path, img)
            dst_path = os.path.join(OUTPUT_PATH, split, category, img)
            shutil.move(src_path, dst_path)

print("✅ Dataset successfully organized into train, test, and valid splits!")
