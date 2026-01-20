# OPTIMIZED GOOGLE COLAB CLASSIFICATION SCRIPT - Your Actual Code Enhanced
# Copy this entire script into a Google Colab cell

# Install required packages
!pip install opencv-python

# Import libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from google.colab import drive, files
import os
import time
from datetime import datetime

print("🍅 TOMATO DISEASE CLASSIFIER - PRODUCTION VERSION")
print("=" * 60)

# Mount Google Drive
print("📁 Mounting Google Drive...")
drive.mount('/content/drive')

# Load model
model_path = '/content/drive/MyDrive/Thesis/model/best_model_new.h5'
class_names = ['early_blight_leaf', 'healthy_leaf', 'late_blight_leaf', 'septoria_leaf', 'unknown']

print("📦 Loading model...")
try:
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Auto-detect model input size
input_shape = model.input_shape[1:3]  # (height, width)
print(f"📊 Model input shape: {input_shape}")

# Confidence threshold for unknown class
CONF_THRESHOLD = 0.7
print(f"🎯 Confidence threshold: {CONF_THRESHOLD}")

# Pre-create CLAHE object (CRITICAL: Use same parameters as training)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Match training exactly
print(f"🔧 CLAHE configured: clipLimit=2.0, tileGridSize=(8,8)")

def preprocess_image(image_path, target_size=(96, 96)):
    """
    EXACT same preprocessing as training - CRITICAL for accuracy!
    Fixed to match your local training preprocessing exactly
    """
    # Read image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
        
    # Convert BGR to RGB (OpenCV uses BGR by default) - CRITICAL!
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE BEFORE resizing (to match training exactly)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)  # Use global clahe object
    enhanced = cv2.merge((cl, a, b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Resize to target size (use INTER_LINEAR to match training)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0,1] - CRITICAL!
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    return np.expand_dims(img, axis=0)

def classify_image(image_path, result_path):
    """Enhanced classification with proper preprocessing"""
    
    # Preprocess with EXACT same method as training
    img = preprocess_image(image_path, target_size=input_shape)
    if img is None:
        return
    
    # Make prediction
    prediction = model.predict(img, verbose=0)[0]  # single image prediction
    class_id = int(np.argmax(prediction))
    confidence = float(prediction[class_id])

    # Determine result class
    if confidence < CONF_THRESHOLD:
        result = "unknown"
        result_confidence = confidence
    else:
        result = class_names[class_id]
        result_confidence = confidence

    # Top 3 predictions for analysis
    top3_idx = np.argsort(prediction)[-3:][::-1]
    top3 = [(class_names[i], float(prediction[i])) for i in top3_idx]

    # Save detailed result
    base = os.path.basename(image_path).split('.')[0]
    txt_path = os.path.join(result_path, f"{base}_result.txt")
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, "w") as f:
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Predicted class: {result} (confidence: {result_confidence:.4f})\n")
        f.write(f"Raw prediction: {class_names[class_id]} ({confidence:.4f})\n")
        f.write(f"Confidence threshold: {CONF_THRESHOLD}\n\n")
        f.write("All class probabilities:\n")
        for i, (cls, prob) in enumerate(zip(class_names, prediction)):
            f.write(f"  {cls}: {prob:.4f} ({prob*100:.1f}%)\n")
        f.write(f"\nTop 3 predictions:\n")
        for cls, prob in top3:
            f.write(f"  {cls}: {prob:.4f} ({prob*100:.1f}%)\n")

    # Enhanced console output
    print(f"📸 {os.path.basename(image_path)}")
    print(f"   Final: {result} ({result_confidence:.3f})")
    print(f"   Raw: {class_names[class_id]} ({confidence:.3f})")
    print(f"   Top3: {[(cls, f'{prob:.3f}') for cls, prob in top3]}")
    print()

def classify_uploaded_images():
    """Interactive function to upload and classify images"""
    print(f"\n📤 Upload images to classify:")
    uploaded = files.upload()
    
    if uploaded:
        print(f"\n🔍 CLASSIFYING {len(uploaded)} IMAGE(S):")
        print("=" * 60)
        
        # Create temp results directory
        temp_results = "/tmp/colab_results"
        os.makedirs(temp_results, exist_ok=True)
        
        for filename in uploaded.keys():
            print(f"\n📸 Processing: {filename}")
            print("-" * 40)
            classify_image(filename, temp_results)
            
        print(f"\n🎉 Classification complete!")
        print(f"Results saved to: {temp_results}")
        
        # Show results summary
        print(f"\n📊 RESULTS SUMMARY:")
        print("=" * 40)
        for filename in uploaded.keys():
            base = os.path.basename(filename).split('.')[0]
            result_file = os.path.join(temp_results, f"{base}_result.txt")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    lines = f.readlines()
                    predicted_line = [l for l in lines if l.startswith("Predicted class:")][0]
                    print(f"   {filename}: {predicted_line.split(': ')[1].strip()}")
    else:
        print("❌ No images uploaded")

def test_preprocessing():
    """Test function to verify preprocessing works"""
    print("🧪 Testing preprocessing...")
    test_images = []
    
    # Look for test images in common locations
    test_paths = [
        '/content/drive/MyDrive/Thesis/test_images',
        '/content/drive/MyDrive/Thesis/images',
        '/content/drive/MyDrive/Thesis/samples'
    ]
    
    for path in test_paths:
        if os.path.exists(path):
            for file in os.listdir(path)[:3]:  # Test first 3 images
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(path, file))
    
    if test_images:
        print(f"Testing with {len(test_images)} sample images...")
        temp_results = "/tmp/test_results"
        os.makedirs(temp_results, exist_ok=True)
        for img_path in test_images:
            classify_image(img_path, temp_results)
    else:
        print("No test images found in Drive. Use upload function instead.")

def start_monitoring():
    """Start continuous monitoring of image folders"""
    # Paths
    images_path = '/content/drive/MyDrive/Thesis/images'
    results_path = '/content/drive/MyDrive/Thesis/results'
    os.makedirs(results_path, exist_ok=True)

    # Track processed files
    processed = set()

    print("🔍 Watching image folders... (press Stop to end)")
    print(f"Monitoring: {images_path}")
    print(f"Results: {results_path}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print()

    try:
        while True:
            new_files = 0
            if os.path.exists(images_path):
                for root, _, files in os.walk(images_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(root, file)

                            # Relative folder (for date-based structure)
                            rel_path = os.path.relpath(root, images_path)
                            save_folder = os.path.join(results_path, rel_path)
                            os.makedirs(save_folder, exist_ok=True)

                            base = os.path.splitext(file)[0]
                            txt_path = os.path.join(save_folder, f"{base}_result.txt")

                            # Process only if new
                            if image_path not in processed and not os.path.exists(txt_path):
                                classify_image(image_path, save_folder)
                                processed.add(image_path)
                                new_files += 1
            else:
                print(f"⚠️  Images path doesn't exist: {images_path}")
                print("   Creating directory...")
                os.makedirs(images_path, exist_ok=True)

            if new_files == 0:
                print(".", end="", flush=True)  # Show it's still running
            
            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n✅ Stopped monitoring. Processed {len(processed)} images total.")

# Main execution
print(f"\n🚀 CLASSIFIER READY!")
print("=" * 60)
print("Choose an option:")
print("1. Upload and classify images interactively")
print("2. Test with existing images in Drive")
print("3. Start continuous monitoring")
print()

# Test preprocessing first
test_preprocessing()

print(f"\n💡 READY FOR CLASSIFICATION!")
print("Run one of these commands:")
print("  classify_uploaded_images()  # Upload and classify")
print("  start_monitoring()          # Start continuous monitoring")

# For immediate testing, uncomment this:
# classify_uploaded_images()