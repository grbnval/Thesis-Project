import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# Constants
IMAGE_SIZE = 300
TRAIN_DATA_DIR = "raw_dataset"
TEST_DATA_DIR = "test_dataset"
PROCESSED_TRAIN_DIR = "processed_train_300x300"
PROCESSED_TEST_DIR = "processed_test_300x300"

def preprocess_image(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Preprocess a single image"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    # Convert to RGB (from BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Resize
    resized_img = cv2.resize(enhanced_img, target_size)
    
    # Normalize to [0,1]
    normalized_img = resized_img / 255.0
    
    return normalized_img.astype(np.float32)

def process_dataset(input_dir, output_dir):
    """Process entire dataset"""
    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Removing...")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)
    
    # Get class names (subdirectories)
    class_names = sorted([d for d in os.listdir(input_dir) 
                          if os.path.isdir(os.path.join(input_dir, d))])
    
    if not class_names:
        raise ValueError(f"No class directories found in {input_dir}")
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Process each class
    for class_name in class_names:
        print(f"Processing class: {class_name}")
        
        # Create output directory for this class
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir)
        
        # Get all image files
        class_dir = os.path.join(input_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"Warning: No image files found in {class_name}/")
            continue
        
        print(f"Processing {len(image_files)} images...")
        
        # Process each image
        for img_file in tqdm(image_files):
            img_path = os.path.join(class_dir, img_file)
            output_path = os.path.join(class_output_dir, 
                                       os.path.splitext(img_file)[0] + '.npy')
            
            try:
                # Preprocess image
                processed_img = preprocess_image(img_path)
                
                if processed_img is not None:
                    # Save as numpy array
                    np.save(output_path, processed_img)
            
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    return class_names

def main():
    print("=== PREPROCESSING DATASETS FOR MOBILENETV2 300x300 ===")
    
    # Check if directories exist
    if not os.path.exists(TRAIN_DATA_DIR):
        raise ValueError(f"Training directory '{TRAIN_DATA_DIR}' not found")
    
    if not os.path.exists(TEST_DATA_DIR):
        raise ValueError(f"Testing directory '{TEST_DATA_DIR}' not found")
    
    # Process training dataset
    print(f"\nProcessing training dataset from: {TRAIN_DATA_DIR}")
    train_classes = process_dataset(TRAIN_DATA_DIR, PROCESSED_TRAIN_DIR)
    
    # Process testing dataset
    print(f"\nProcessing testing dataset from: {TEST_DATA_DIR}")
    test_classes = process_dataset(TEST_DATA_DIR, PROCESSED_TEST_DIR)
    
    # Check if classes match
    if set(train_classes) != set(test_classes):
        print("\nWarning: Training and testing datasets have different classes!")
        print(f"Training classes: {train_classes}")
        print(f"Testing classes: {test_classes}")
    
    # Print summary
    print("\n=== PREPROCESSING COMPLETED ===")
    print(f"Processed training images saved to: {PROCESSED_TRAIN_DIR}")
    print(f"Processed testing images saved to: {PROCESSED_TEST_DIR}")
    
    # Count processed files
    train_count = sum(len(os.listdir(os.path.join(PROCESSED_TRAIN_DIR, cls))) 
                       for cls in train_classes)
    test_count = sum(len(os.listdir(os.path.join(PROCESSED_TEST_DIR, cls))) 
                      for cls in test_classes)
    
    print(f"Total processed training images: {train_count}")
    print(f"Total processed testing images: {test_count}")
    
    print("\nYou can now train the model using: python train_mobilenetv2_separate.py")

if __name__ == "__main__":
    main()
