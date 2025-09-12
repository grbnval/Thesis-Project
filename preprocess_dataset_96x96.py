import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_dataset(input_dir):
    """Check if dataset exists and contains images"""
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist!")
        print("Please create the following folder structure:")
        print(f"{input_dir}/")
        print("├── healthy_leaf/")
        print("├── early_blight_leaf/")
        print("├── late_blight_leaf/")
        print("└── septoria_leaf/")
        return False
        
    image_count = 0
    class_distribution = {}
    
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            num_images = len(images)
            image_count += num_images
            class_distribution[class_name] = num_images
            print(f"Found {num_images} images in {class_name}/")
    
    if image_count == 0:
        print("\nNo images found! Please add images to the appropriate folders:")
        print("1. Put healthy tomato images in 'healthy_leaf/'")
        print("2. Put early blight images in 'early_blight_leaf/'")
        print("3. Put late blight images in 'late_blight_leaf/'")
        print("4. Put septoria leaf images in 'septoria_leaf/'")
        return False
        
    print(f"\nTotal images found: {image_count}")
    
    # Plot class distribution
    print("\nClass distribution:")
    for class_name, count in class_distribution.items():
        print(f"- {class_name}: {count} images ({count/image_count*100:.1f}%)")
    
    return True

def preprocess_image(image_path, target_size=(96, 96), enhance_contrast=True):
    """Enhanced preprocessing for tomato disease images with 96x96 resolution"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size (96x96)
    img = cv2.resize(img, target_size)
    
    # Enhance contrast using CLAHE
    if enhance_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Normalize to [0,1]
    img = img.astype('float32') / 255.0
    
    return img

def process_image_task(args):
    """Process a single image (for parallel processing)"""
    input_path, output_path, target_size = args
    
    try:
        # Preprocess the image
        processed_img = preprocess_image(input_path, target_size)
        
        if processed_img is not None:
            # Save as numpy array
            np.save(output_path, processed_img)
            return True
        return False
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_dataset(input_dir, output_dir, target_size=(96, 96), max_workers=8):
    """Process entire dataset with parallel processing"""
    # Collect all image paths and output paths
    tasks = []
    
    # Count total images and create task list
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Create output class directory
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        
        # Create task for each image
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            input_path = os.path.join(class_path, img_name)
            output_path = os.path.join(output_class_path, 
                                     os.path.splitext(img_name)[0] + '.npy')
            
            tasks.append((input_path, output_path, target_size))
    
    # Process images in parallel
    successful = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image_task, task) for task in tasks]
        
        with tqdm(total=len(tasks), desc="Processing images") as pbar:
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                pbar.update(1)
    
    print(f"Successfully processed {successful} out of {len(tasks)} images")

def visualize_samples(processed_dir, num_samples=5):
    """Visualize random samples from each class after preprocessing"""
    # Create visualization directory
    viz_dir = os.path.join(os.path.dirname(processed_dir), "visualization")
    os.makedirs(viz_dir, exist_ok=True)
    
    class_names = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    
    # Create a figure for all classes
    plt.figure(figsize=(15, 3 * len(class_names)))
    
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(processed_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        if not files:
            continue
            
        # Select random samples
        sample_files = random.sample(files, min(num_samples, len(files)))
        
        for j, file in enumerate(sample_files):
            # Load the numpy array
            img_path = os.path.join(class_dir, file)
            img = np.load(img_path)
            
            # Display the image
            plt.subplot(len(class_names), num_samples, i * num_samples + j + 1)
            plt.imshow(img)
            plt.title(f"{class_name}" if j == 0 else "")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "sample_96x96_images.png"))
    plt.close()
    
    print(f"Sample visualizations saved to: {viz_dir}")

def create_train_test_split(processed_dir, train_dir, test_dir, test_split=0.2):
    """Split the processed dataset into training and testing sets"""
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for class_name in os.listdir(processed_dir):
        class_dir = os.path.join(processed_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Create class directories in train and test
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Get all numpy files
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        random.shuffle(files)
        
        # Calculate split point
        split_idx = int(len(files) * (1 - test_split))
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        print(f"Class {class_name}: {len(train_files)} train, {len(test_files)} test")
        
        # Copy files to train directory
        for file in train_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(train_class_dir, file)
            np.save(dst, np.load(src))
        
        # Copy files to test directory
        for file in test_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(test_class_dir, file)
            np.save(dst, np.load(src))

if __name__ == "__main__":
    # Use correct path to the raw_dataset directory
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(PROJECT_ROOT, "raw_dataset")
    processed_dir = os.path.join(PROJECT_ROOT, "processed_dataset_96x96")
    
    # Directories for train-test split
    train_dir = os.path.join(PROJECT_ROOT, "processed_train_96x96")
    test_dir = os.path.join(PROJECT_ROOT, "processed_test_96x96")
    
    print("=== PREPROCESSING IMAGES TO 96x96 RESOLUTION ===")
    print(f"Looking for dataset in: {raw_dir}")
    print("Checking dataset structure...")
    if not check_dataset(raw_dir):
        exit(1)
    
    # Ask user whether to process or skip
    action = input("\nDo you want to process the images? (y/n, default: y): ").lower()
    if action != 'n':
        print("\nStarting preprocessing...")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Ask user about number of parallel workers
        try:
            max_workers = int(input(f"Enter number of parallel workers (default: 8): ") or 8)
        except ValueError:
            max_workers = 8
        
        process_dataset(raw_dir, processed_dir, target_size=(96, 96), max_workers=max_workers)
        print("\nPreprocessing complete!")
        print(f"Processed images saved to: {processed_dir}")
        
        # Visualize some processed samples
        visualize_samples(processed_dir)
    else:
        print("Skipping image processing...")
    
    # Ask user about creating train-test split
    split_action = input("\nDo you want to create train-test split? (y/n, default: y): ").lower()
    if split_action != 'n':
        try:
            test_split = float(input(f"Enter test split ratio (default: 0.2): ") or 0.2)
            if test_split <= 0 or test_split >= 1:
                print("Invalid split ratio. Using default 0.2")
                test_split = 0.2
        except ValueError:
            test_split = 0.2
            
        print(f"\nCreating train-test split with {test_split*100:.0f}% test data...")
        create_train_test_split(processed_dir, train_dir, test_dir, test_split)
        print(f"Train data saved to: {train_dir}")
        print(f"Test data saved to: {test_dir}")
    
    print("\n=== PREPROCESSING COMPLETED SUCCESSFULLY ===")