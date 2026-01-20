# GOOGLE COLAB INFERENCE SCRIPT - Ready to Use
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

print("🍅 TOMATO DISEASE CLASSIFIER - GOOGLE COLAB")
print("=" * 60)

# Mount Google Drive
print("📁 Mounting Google Drive...")
drive.mount('/content/drive')

# Model path (updated with your correct location)
model_path = '/content/drive/MyDrive/Thesis/model/best_model_new.h5'
class_names = ['early_blight_leaf', 'healthy_leaf', 'late_blight_leaf', 'septoria_leaf', 'unknown']

# Check if model exists
if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
    print("Please check if the file is uploaded to the correct location in Google Drive")
else:
    print(f"✅ Model found: {model_path}")

# Load model
print("📦 Loading model...")
try:
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    print(f"📊 Model input shape: {model.input.shape}")
    print(f"📊 Model output shape: {model.output.shape}")
    print(f"🎯 Classes: {class_names}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

def preprocess_image_exact(image_path):
    """Exact preprocessing matching your local training"""
    print(f"🔄 Preprocessing {image_path}...")
    
    # Read image with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"   Original size: {image.shape}")
    
    # Convert BGR to RGB (critical!)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE enhancement (exactly as in training)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    image_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Resize to model input size
    resized = cv2.resize(image_enhanced, (96, 96), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0,1] range
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    final = np.expand_dims(normalized, axis=0)
    
    print(f"   Processed to: {final.shape}")
    return final

def classify_image(image_path):
    """Classify a single image"""
    try:
        # Preprocess
        processed = preprocess_image_exact(image_path)
        
        # Predict
        print("🧠 Running prediction...")
        predictions = model.predict(processed, verbose=0)
        
        # Get results
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = class_names[predicted_idx]
        
        # Display results
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   Class: {predicted_class}")
        print(f"   Confidence: {confidence:.1%}")
        
        print(f"\n📊 All class probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
            marker = " ← PREDICTED" if i == predicted_idx else ""
            print(f"   {class_name:15s}: {prob:.1%}{marker}")
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"❌ Error classifying image: {e}")
        return None, None

# Upload and classify images
print(f"\n📤 Please upload your test images:")
uploaded = files.upload()

if uploaded:
    print(f"\n🔍 CLASSIFYING {len(uploaded)} IMAGE(S):")
    print("=" * 60)
    
    for i, filename in enumerate(uploaded.keys(), 1):
        print(f"\n[{i}/{len(uploaded)}] 📸 Processing: {filename}")
        print("-" * 40)
        
        predicted_class, confidence = classify_image(filename)
        
        if predicted_class:
            if confidence > 0.8:
                print("✅ High confidence prediction")
            elif confidence > 0.5:
                print("⚠️  Medium confidence prediction")
            else:
                print("❌ Low confidence - check image quality")
        
        if i < len(uploaded):
            print()  # Add space between images
            
    print(f"\n🎉 Classification complete!")
    
else:
    print("❌ No images uploaded")

print(f"\n💡 TROUBLESHOOTING TIPS:")
print("=" * 60)
print("• If predictions seem wrong, the issue is likely preprocessing")
print("• Make sure images are clear and well-lit")
print("• For best results, crop images to show mainly the leaf")
print("• Unknown class is for non-leaf objects (soil, tools, etc.)")