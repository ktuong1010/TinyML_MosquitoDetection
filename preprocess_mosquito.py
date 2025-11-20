import cv2
import numpy as np
import os
import random
import shutil
from pathlib import Path
import kagglehub # pip install kagglehub

# ================= CONFIGURATION =================

# 1. KAGGLE DATASET CONFIG
# REPLACE THIS with the specific dataset handle you found on Kaggle.
# Format: "username/dataset-slug"
# Example: "potamitis/mosquito-species-images" (Check your Kaggle URL)

# Download latest version
DATASET_HANDLE = "masud1901/mosquito-dataset-for-classification-cnn"

# 2. OUTPUT CONFIG
OUTPUT_DIR = "mosquito_dataset_processed"

# 3. IMAGE SETTINGS
IMG_SIZE = (160, 160)
INTERPOLATION = cv2.INTER_AREA 
NORMALIZE_RANGE = (0, 1) 
SAVE_AS_NUMPY = False    

# ================= UTILS =================

def normalize_image(img):
    img_float = img.astype(np.float32)
    if NORMALIZE_RANGE == (0, 1):
        return img_float / 255.0
    elif NORMALIZE_RANGE == (-1, 1):
        return (img_float / 127.5) - 1.0
    return img_float

def rotate_image(image):
    # Random 90-degree rotation (Clockwise or Counter-Clockwise)
    rotate_code = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])
    return cv2.rotate(image, rotate_code)

def flip_image(image):
    flip_code = random.choice([0, 1]) # 0=Vertical, 1=Horizontal
    return cv2.flip(image, flip_code)

def process_and_save(img_array, class_name, filename, output_base):
    # Creates subfolder based on 'class_name' automatically
    save_dir = os.path.join(output_base, class_name)
    os.makedirs(save_dir, exist_ok=True)
    
    stem = Path(filename).stem
    
    if SAVE_AS_NUMPY:
        norm_img = normalize_image(img_array)
        np.save(os.path.join(save_dir, f"{stem}.npy"), norm_img)
    else:
        output_path = os.path.join(save_dir, f"{stem}.png")
        cv2.imwrite(output_path, img_array)

def download_dataset():
    print(f"Downloading dataset: {DATASET_HANDLE} via Kaggle Hub...")
    try:
        path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"Download complete. Path: {path}")
        return path
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Make sure you have run 'pip install kagglehub' and are logged in.")
        return None

# ================= MAIN PIPELINE =================

def main():
    # 1. Download Data
    source_dir = download_dataset()
    if not source_dir:
        return

    # 2. Prepare Output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"Starting preprocessing pipeline...")
    print(f"Target Size: {IMG_SIZE} | Grayscale: Yes")

    # 3. Verify Structure
    # Sometimes Kaggle datasets have nested folders. We look for the level containing folders.
    # Heuristic: If source_dir contains only 1 folder, dive into it.
    items = os.listdir(source_dir)
    if len(items) == 1 and os.path.isdir(os.path.join(source_dir, items[0])):
        source_dir = os.path.join(source_dir, items[0])

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    if not classes:
        print(f"Error: No class directories found in '{source_dir}'.")
        return

    total_processed = 0

    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nProcessing Class: {class_name} ({len(images)} images found)")
        
        random.shuffle(images) 
        
        n_total = len(images)
        n_split = n_total // 4
        
        set_original = images[:n_split]
        set_rotate = images[n_split:n_split*2]
        set_flip = images[n_split*2:n_split*3]
        set_both = images[n_split*3:]

        def process_batch(file_list, operations, suffix=""):
            count = 0
            for img_name in file_list:
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None: continue

                img_resized = cv2.resize(img, IMG_SIZE, interpolation=INTERPOLATION)
                
                final_img = img_resized
                if 'rotate' in operations:
                    final_img = rotate_image(final_img)
                if 'flip' in operations:
                    final_img = flip_image(final_img)

                # Pass class_name to ensure it goes into the correct folder
                new_filename = f"{Path(img_name).stem}{suffix}{Path(img_name).suffix}"
                process_and_save(final_img, class_name, new_filename, OUTPUT_DIR)
                count += 1
            return count

        c1 = process_batch(set_original, [], suffix="_orig")
        c2 = process_batch(set_rotate, ['rotate'], suffix="_rot")
        c3 = process_batch(set_flip, ['flip'], suffix="_flip")
        c4 = process_batch(set_both, ['rotate', 'flip'], suffix="_rotflip")
        
        print(f"  -> Saved {c1} Original | {c2} Rot | {c3} Flip | {c4} Both")
        total_processed += (c1 + c2 + c3 + c4)

    print(f"\n Pipeline complete. Total images: {total_processed}")
    print(f"Output saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()