# OPTIMIZED Google Colab Inference Code - Much Faster!

import tensorflow as tf
import numpy as np
import cv2
import os
import time
from datetime import datetime

# ============= PERFORMANCE OPTIMIZATIONS =============

# 1. Enable GPU if available
print("🔧 Checking GPU availability...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid memory issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("⚠️ No GPU detected, using CPU")

# 2. Load model once and keep in memory
print("📦 Loading model...")
start_time = time.time()
model = tf.keras.models.load_model('/content/drive/MyDrive/Thesis/model/best_model_new.h5')
load_time = time.time() - start_time
print(f"✅ Model loaded in {load_time:.2f} seconds")

# 3. Pre-compile the model for faster inference
print("🔧 Warming up model...")
dummy_input = np.random.random((1, 96, 96, 3)).astype(np.float32)
_ = model.predict(dummy_input, verbose=0)  # Warm-up prediction
print("✅ Model warmed up")

# Define labels and settings
class_names = ["early_blight_leaf", "healthy_leaf", "late_blight_leaf", "septoria_leaf", "unknown"]
input_shape = model.input_shape[1:3]
CONF_THRESHOLD = 0.7

# 4. Pre-create CLAHE object (reuse for speed)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

def preprocess_image_fast(image_path, target_size=(96, 96)):
    """
    Optimized preprocessing with minimal overhead
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize first (faster on smaller image)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE enhancement (using pre-created object)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0
    
    return np.expand_dims(img, axis=0)

def preprocess_image_ultra_fast(image_path, target_size=(96, 96)):
    """
    Ultra-fast preprocessing - skips CLAHE for speed
    Use this if speed is more important than slight accuracy loss
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Convert and resize in one step
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Simple normalization only
    img = img.astype(np.float32) / 255.0
    
    return np.expand_dims(img, axis=0)

def classify_image_fast(image_path, result_path, use_ultra_fast=False):
    """
    Fast classification with timing
    """
    start_time = time.time()
    
    # Choose preprocessing method
    if use_ultra_fast:
        img = preprocess_image_ultra_fast(image_path, target_size=input_shape)
        preprocess_method = "Ultra-fast (no CLAHE)"
    else:
        img = preprocess_image_fast(image_path, target_size=input_shape)
        preprocess_method = "Fast (with CLAHE)"
    
    if img is None:
        return
    
    preprocess_time = time.time() - start_time
    
    # Make prediction
    pred_start = time.time()
    prediction = model.predict(img, verbose=0)[0]
    pred_time = time.time() - pred_start
    
    # Process results
    class_id = int(np.argmax(prediction))
    confidence = float(prediction[class_id])
    
    if confidence < CONF_THRESHOLD:
        result = "unknown"
    else:
        result = class_names[class_id]
    
    total_time = time.time() - start_time
    
    # Save results with timing info
    base = os.path.basename(image_path).split('.')[0]
    txt_path = os.path.join(result_path, f"{base}_result.txt")
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    
    with open(txt_path, "w") as f:
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write(f"Predicted class: {result} (confidence: {confidence:.4f})\n")
        f.write(f"Preprocessing method: {preprocess_method}\n")
        f.write(f"Timing breakdown:\n")
        f.write(f"  Preprocessing: {preprocess_time:.3f}s\n")
        f.write(f"  Model inference: {pred_time:.3f}s\n")
        f.write(f"  Total time: {total_time:.3f}s\n\n")
        f.write("All class probabilities:\n")
        for cls, prob in zip(class_names, prediction):
            f.write(f"  {cls}: {prob:.4f} ({prob*100:.1f}%)\n")
    
    # Console output with timing
    print(f"⚡ {os.path.basename(image_path)}")
    print(f"   Result: {result} ({confidence:.3f})")
    print(f"   Times: Prep={preprocess_time:.3f}s, Pred={pred_time:.3f}s, Total={total_time:.3f}s")
    print()
    
    return result, confidence, total_time

def batch_classify_images(image_paths, result_path, batch_size=4):
    """
    Process multiple images in batches for even faster processing
    """
    print(f"🚀 Batch processing {len(image_paths)} images (batch size: {batch_size})")
    
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        # Preprocess batch
        for path in batch_paths:
            img = preprocess_image_fast(path)
            if img is not None:
                batch_images.append(img[0])  # Remove batch dimension
                valid_paths.append(path)
        
        if not batch_images:
            continue
            
        # Batch prediction
        start_time = time.time()
        batch_array = np.array(batch_images)
        predictions = model.predict(batch_array, verbose=0)
        batch_time = time.time() - start_time
        
        # Process results
        for path, prediction in zip(valid_paths, predictions):
            class_id = int(np.argmax(prediction))
            confidence = float(prediction[class_id])
            result = class_names[class_id] if confidence >= CONF_THRESHOLD else "unknown"
            
            print(f"📸 {os.path.basename(path)}: {result} ({confidence:.3f}) - {batch_time/len(predictions):.3f}s")
            results.append((path, result, confidence))
    
    return results

# ============= PERFORMANCE TESTING =============

def benchmark_inference_speed():
    """Test inference speed with different methods"""
    print("🏃‍♂️ SPEED BENCHMARK")
    print("=" * 50)
    
    # Create test image
    test_img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    test_path = "/tmp/test_image.jpg"
    cv2.imwrite(test_path, test_img)
    
    methods = [
        ("Standard preprocessing", False),
        ("Ultra-fast preprocessing", True)
    ]
    
    for method_name, use_ultra_fast in methods:
        times = []
        for _ in range(10):  # Test 10 times
            start = time.time()
            if use_ultra_fast:
                img = preprocess_image_ultra_fast(test_path)
            else:
                img = preprocess_image_fast(test_path)
            _ = model.predict(img, verbose=0)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        print(f"{method_name}: {avg_time:.3f}s avg ({1/avg_time:.1f} FPS)")
    
    # Clean up
    os.remove(test_path)

# ============= MAIN EXECUTION =============

# Run benchmark
benchmark_inference_speed()

# Set up paths
images_path = '/content/drive/MyDrive/Thesis/images'
results_path = '/content/drive/MyDrive/Thesis/results'
os.makedirs(results_path, exist_ok=True)

# Choose processing mode
USE_ULTRA_FAST = True  # Set to False for maximum accuracy, True for maximum speed
USE_BATCH_PROCESSING = True  # Set to True for processing multiple images faster

print(f"\n🔍 Starting optimized monitoring...")
print(f"Mode: {'Ultra-fast' if USE_ULTRA_FAST else 'Standard'}")
print(f"Batch processing: {'Enabled' if USE_BATCH_PROCESSING else 'Disabled'}")
print(f"Monitoring: {images_path}")

processed = set()

try:
    while True:
        new_images = []
        
        # Collect new images
        for root, _, files in os.walk(images_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    if image_path not in processed:
                        rel_path = os.path.relpath(root, images_path)
                        save_folder = os.path.join(results_path, rel_path)
                        base = os.path.splitext(file)[0]
                        txt_path = os.path.join(save_folder, f"{base}_result.txt")
                        
                        if not os.path.exists(txt_path):
                            new_images.append((image_path, save_folder))
        
        # Process new images
        if new_images:
            if USE_BATCH_PROCESSING and len(new_images) > 1:
                # Batch processing
                image_paths = [item[0] for item in new_images]
                batch_classify_images(image_paths, results_path)
            else:
                # Individual processing
                for image_path, save_folder in new_images:
                    classify_image_fast(image_path, save_folder, USE_ULTRA_FAST)
            
            # Mark as processed
            for image_path, _ in new_images:
                processed.add(image_path)
        else:
            print(".", end="", flush=True)
        
        time.sleep(2)

except KeyboardInterrupt:
    print(f"\n✅ Stopped. Processed {len(processed)} images total.")