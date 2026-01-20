import tensorflow as tf
import numpy as np
import cv2
import time
import os
import json

def benchmark_specific_image(image_path, model_path="models/mobilenetv2_efficientnet_style_96x96"):
    """
    Detailed timing analysis of each step for a specific image
    """
    print(f"🎯 DETAILED TIMING ANALYSIS")
    print(f"Image: {os.path.basename(image_path)}")
    print("=" * 60)
    
    # Load model
    print("📦 Loading model...")
    model_start = time.time()
    try:
        model = tf.keras.models.load_model(os.path.join(model_path, "best_model.h5"))
    except:
        try:
            model = tf.keras.models.load_model(os.path.join(model_path, "model.keras"))
        except:
            print("❌ Could not load model")
            return
    
    model_load_time = time.time() - model_start
    print(f"   Model loading: {model_load_time:.3f}s")
    
    # Load class names
    with open(os.path.join(model_path, "class_info.json"), 'r') as f:
        class_info = json.load(f)
    class_names = class_info["classes"]
    
    # Warm up model (first prediction is always slower)
    print("🔥 Warming up model...")
    warmup_start = time.time()
    dummy_input = np.random.random((1, 96, 96, 3)).astype(np.float32)
    _ = model.predict(dummy_input, verbose=0)
    warmup_time = time.time() - warmup_start
    print(f"   Warmup time: {warmup_time:.3f}s")
    
    # Now test the actual image multiple times
    print(f"\n🧪 Testing image 5 times...")
    
    all_times = {
        'file_read': [],
        'preprocessing': [],
        'model_prediction': [],
        'post_processing': [],
        'total': []
    }
    
    for i in range(5):
        print(f"\n--- Test {i+1} ---")
        total_start = time.time()
        
        # 1. File reading
        read_start = time.time()
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to read image on attempt {i+1}")
            continue
        read_time = time.time() - read_start
        print(f"File read: {read_time:.4f}s")
        all_times['file_read'].append(read_time)
        
        # 2. Preprocessing
        preprocess_start = time.time()
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        
        # CLAHE enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Normalize
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        preprocess_time = time.time() - preprocess_start
        print(f"Preprocessing: {preprocess_time:.4f}s")
        all_times['preprocessing'].append(preprocess_time)
        
        # 3. Model prediction
        pred_start = time.time()
        prediction = model.predict(img, verbose=0)[0]
        pred_time = time.time() - pred_start
        print(f"Model prediction: {pred_time:.4f}s")
        all_times['model_prediction'].append(pred_time)
        
        # 4. Post-processing
        post_start = time.time()
        class_id = int(np.argmax(prediction))
        confidence = float(prediction[class_id])
        result = class_names[class_id]
        post_time = time.time() - post_start
        print(f"Post-processing: {post_time:.4f}s")
        all_times['post_processing'].append(post_time)
        
        total_time = time.time() - total_start
        print(f"Total time: {total_time:.4f}s")
        print(f"Result: {result} ({confidence:.3f})")
        all_times['total'].append(total_time)
    
    # Calculate averages
    print(f"\n📊 TIMING SUMMARY (5 runs):")
    print("=" * 40)
    for step, times in all_times.items():
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            print(f"{step:15}: {avg_time:.4f}s ± {std_time:.4f}s (min: {min_time:.4f}s, max: {max_time:.4f}s)")
    
    # Identify bottleneck
    avg_times = {step: np.mean(times) for step, times in all_times.items() if times}
    bottleneck = max(avg_times.items(), key=lambda x: x[1])
    
    print(f"\n🐌 BOTTLENECK IDENTIFIED:")
    print(f"   {bottleneck[0]} is taking {bottleneck[1]:.4f}s ({bottleneck[1]/avg_times['total']*100:.1f}% of total time)")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    if avg_times['model_prediction'] > 0.5:
        print("   🚀 Model prediction is slow - consider:")
        print("      - Using TFLite model instead")
        print("      - Enabling GPU acceleration")
        print("      - Using model quantization")
    
    if avg_times['preprocessing'] > 0.2:
        print("   🚀 Preprocessing is slow - consider:")
        print("      - Skipping CLAHE enhancement")
        print("      - Using smaller input images")
        print("      - Pre-creating CLAHE object")
    
    if avg_times['file_read'] > 0.1:
        print("   🚀 File reading is slow - consider:")
        print("      - Moving files to local storage")
        print("      - Using faster storage")
        print("      - Checking file corruption")
    
    return avg_times

if __name__ == "__main__":
    # Test the problematic image
    problem_image = r"C:\Users\Azief\Documents\GitHub\Thesis-Project\raw_dataset\healthy_leaf\Copy of Copy of 20251005-005612(5).jpg"
    
    if os.path.exists(problem_image):
        benchmark_specific_image(problem_image)
    else:
        print(f"❌ Image not found: {problem_image}")
        print("Please check the file path")