# COMPREHENSIVE GOOGLE COLAB DIAGNOSTIC SCRIPT
# Copy and paste this entire script into a Google Colab cell
# This will help identify exactly why Colab gives different results

# First, install required packages
!pip install opencv-python

# Import libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
from google.colab import drive, files
import hashlib

def mount_drive_and_setup():
    """Mount Google Drive and set up paths"""
    print("🔧 SETTING UP GOOGLE COLAB ENVIRONMENT")
    print("=" * 60)
    
    # Mount Google Drive
    print("📁 Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # Check OpenCV version
    print(f"📦 OpenCV version: {cv2.__version__}")
    print(f"📦 TensorFlow version: {tf.__version__}")
    
    return True

def verify_model_integrity():
    """Check if the model file is the same as locally"""
    print("\n🔍 VERIFYING MODEL INTEGRITY")
    print("=" * 60)
    
    # Model path in Google Drive
    model_path = '/content/drive/MyDrive/Thesis/model/best_model_new.h5'
    class_info_path = '/content/drive/MyDrive/Thesis/model/class_info.json'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please check the path and make sure the model is uploaded to Google Drive")
        return None, None
        
    if not os.path.exists(class_info_path):
        print(f"❌ Class info file not found: {class_info_path}")
        print("   Using fallback classes")
        class_names = ['early_blight_leaf', 'healthy_leaf', 'late_blight_leaf', 'septoria_leaf', 'unknown']
    else:
        with open(class_info_path, 'r') as f:
            class_info = json.load(f)
        class_names = class_info['classes']
        print(f"✅ Class info loaded: {class_names}")
    
    # Load model
    print(f"📦 Loading model...")
    try:
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        
        # Check model output
        output_shape = model.outputs[0].shape
        print(f"📊 Model output shape: {output_shape}")
        print(f"🎯 Number of classes: {output_shape[-1]}")
        
        if output_shape[-1] != len(class_names):
            print(f"⚠️  CLASS MISMATCH!")
            print(f"   Model has {output_shape[-1]} outputs")
            print(f"   Class list has {len(class_names)} classes")
        else:
            print(f"✅ Class count matches: {len(class_names)}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None
    
    return model, class_names

def test_preprocessing_methods(image_path, model, class_names):
    """Test different preprocessing methods to find the issue"""
    print(f"\n🧪 TESTING PREPROCESSING METHODS")
    print("=" * 60)
    
    methods = {
        "Method 1 (OpenCV + CLAHE)": preprocess_opencv_clahe,
        "Method 2 (PIL-like)": preprocess_pil_like,
        "Method 3 (No CLAHE)": preprocess_no_clahe,
        "Method 4 (Different resize)": preprocess_different_resize
    }
    
    results = {}
    
    for method_name, preprocess_func in methods.items():
        try:
            print(f"\n🔬 Testing {method_name}...")
            processed = preprocess_func(image_path)
            predictions = model.predict(processed, verbose=0)
            
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            predicted_class = class_names[predicted_idx]
            
            results[method_name] = {
                'class': predicted_class,
                'confidence': confidence,
                'index': predicted_idx,
                'all_probs': predictions[0]
            }
            
            print(f"   Result: {predicted_class} ({confidence:.1%}) [Index: {predicted_idx}]")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[method_name] = {'error': str(e)}
    
    # Show detailed comparison
    print(f"\n📊 DETAILED COMPARISON")
    print("=" * 60)
    
    for method_name, result in results.items():
        if 'error' not in result:
            print(f"\n{method_name}:")
            print(f"  Predicted: {result['class']} (Index {result['index']})")
            print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']:.1%})")
            print(f"  All probabilities:")
            for i, (class_name, prob) in enumerate(zip(class_names, result['all_probs'])):
                marker = " ← WINNER" if i == result['index'] else ""
                print(f"    {i}: {class_name:15s} = {prob:.6f} ({prob:.1%}){marker}")
    
    return results

def preprocess_opencv_clahe(image_path):
    """Method 1: OpenCV with CLAHE (should match local training)"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    image_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    resized = cv2.resize(image_enhanced, (96, 96), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def preprocess_pil_like(image_path):
    """Method 2: PIL-like loading with OpenCV CLAHE"""
    from PIL import Image
    
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    image_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    resized = cv2.resize(image_enhanced, (96, 96), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def preprocess_no_clahe(image_path):
    """Method 3: No CLAHE enhancement"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    resized = cv2.resize(image_rgb, (96, 96), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def preprocess_different_resize(image_path):
    """Method 4: Different resize interpolation"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    image_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    resized = cv2.resize(image_enhanced, (96, 96), interpolation=cv2.INTER_CUBIC)  # Different interpolation
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def run_comprehensive_diagnosis():
    """Run complete diagnostic"""
    print("🍅 COMPREHENSIVE GOOGLE COLAB DIAGNOSTIC")
    print("=" * 60)
    
    # Setup
    mount_drive_and_setup()
    
    # Verify model
    model, class_names = verify_model_integrity()
    if model is None:
        print("❌ Cannot proceed without model")
        return
    
    # Upload test image
    print(f"\n📤 Please upload your test image:")
    uploaded = files.upload()
    
    if not uploaded:
        print("❌ No image uploaded")
        return
    
    # Get uploaded image name
    image_name = list(uploaded.keys())[0]
    print(f"📸 Testing with: {image_name}")
    
    # Test all preprocessing methods
    results = test_preprocessing_methods(image_name, model, class_names)
    
    # Provide recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    unique_results = set()
    for result in results.values():
        if 'class' in result:
            unique_results.add(result['class'])
    
    if len(unique_results) == 1:
        predicted_class = list(unique_results)[0]
        print(f"✅ All methods agree: {predicted_class}")
        print(f"   The model is working consistently in Colab")
    else:
        print(f"⚠️  Different methods give different results:")
        for method_name, result in results.items():
            if 'class' in result:
                print(f"   {method_name}: {result['class']} ({result['confidence']:.1%})")
        
        print(f"\n🔧 Use Method 1 (OpenCV + CLAHE) for best results:")
        print(f"   This should match your local training preprocessing")

# Run the diagnostic
if __name__ == "__main__":
    run_comprehensive_diagnosis()