# Debug Colab Preprocessing - Find the exact difference
# This script helps identify why the same image gives different results locally vs Colab

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
from PIL import Image

def load_model_and_classes():
    """Load the model and class information"""
    model_path = "models/mobilenetv2_efficientnet_style_96x96/best_model.h5"
    model = keras.models.load_model(model_path)
    
    # Try to load class info
    try:
        with open("models/mobilenetv2_efficientnet_style_96x96/class_info.json", 'r') as f:
            class_info = json.load(f)
        class_names = list(class_info.keys())
    except:
        # Fallback to the 5 classes we know
        class_names = ['early_blight_leaf', 'healthy_leaf', 'late_blight_leaf', 'septoria_leaf', 'unknown']
    
    return model, class_names

def preprocess_local_method(image_path):
    """Local preprocessing method (what works correctly)"""
    print(f"\n🏠 LOCAL METHOD for {os.path.basename(image_path)}:")
    
    # Read with OpenCV
    image = cv2.imread(image_path)
    print(f"   Original shape: {image.shape}")
    print(f"   Original dtype: {image.dtype}")
    print(f"   Original range: [{image.min()}, {image.max()}]")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"   After BGR→RGB: shape={image_rgb.shape}, range=[{image_rgb.min()}, {image_rgb.max()}]")
    
    # Apply CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    image_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    print(f"   After CLAHE: shape={image_enhanced.shape}, range=[{image_enhanced.min()}, {image_enhanced.max()}]")
    
    # Resize to 96x96
    resized = cv2.resize(image_enhanced, (96, 96), interpolation=cv2.INTER_LINEAR)
    print(f"   After resize: shape={resized.shape}, range=[{resized.min()}, {resized.max()}]")
    
    # Normalize to [0,1]
    normalized = resized.astype(np.float32) / 255.0
    print(f"   After normalize: shape={normalized.shape}, range=[{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Add batch dimension
    final = np.expand_dims(normalized, axis=0)
    print(f"   Final shape: {final.shape}")
    
    return final

def preprocess_pil_method(image_path):
    """PIL-based preprocessing (what might be happening in Colab)"""
    print(f"\n📱 PIL METHOD for {os.path.basename(image_path)}:")
    
    # Read with PIL
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_array = np.array(image)
    print(f"   Original shape: {image_array.shape}")
    print(f"   Original dtype: {image_array.dtype}")
    print(f"   Original range: [{image_array.min()}, {image_array.max()}]")
    
    # PIL already gives RGB, no conversion needed
    print(f"   Already RGB: shape={image_array.shape}, range=[{image_array.min()}, {image_array.max()}]")
    
    # Try to apply CLAHE (this might be where the difference is)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        image_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        print(f"   After CLAHE: shape={image_enhanced.shape}, range=[{image_enhanced.min()}, {image_enhanced.max()}]")
    except Exception as e:
        print(f"   ❌ CLAHE failed: {e}")
        image_enhanced = image_array
    
    # Resize to 96x96
    resized = cv2.resize(image_enhanced, (96, 96), interpolation=cv2.INTER_LINEAR)
    print(f"   After resize: shape={resized.shape}, range=[{resized.min()}, {resized.max()}]")
    
    # Normalize to [0,1]
    normalized = resized.astype(np.float32) / 255.0
    print(f"   After normalize: shape={normalized.shape}, range=[{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Add batch dimension
    final = np.expand_dims(normalized, axis=0)
    print(f"   Final shape: {final.shape}")
    
    return final

def preprocess_no_clahe_method(image_path):
    """No CLAHE preprocessing (might be what's happening in Colab)"""
    print(f"\n🚫 NO CLAHE METHOD for {os.path.basename(image_path)}:")
    
    # Read with OpenCV
    image = cv2.imread(image_path)
    print(f"   Original shape: {image.shape}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"   After BGR→RGB: shape={image_rgb.shape}, range=[{image_rgb.min()}, {image_rgb.max()}]")
    
    # Skip CLAHE enhancement
    print(f"   Skipping CLAHE...")
    
    # Resize to 96x96
    resized = cv2.resize(image_rgb, (96, 96), interpolation=cv2.INTER_LINEAR)
    print(f"   After resize: shape={resized.shape}, range=[{resized.min()}, {resized.max()}]")
    
    # Normalize to [0,1]
    normalized = resized.astype(np.float32) / 255.0
    print(f"   After normalize: shape={normalized.shape}, range=[{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Add batch dimension
    final = np.expand_dims(normalized, axis=0)
    print(f"   Final shape: {final.shape}")
    
    return final

def compare_preprocessing_methods(image_path):
    """Compare all preprocessing methods and their results"""
    print("🔍 PREPROCESSING COMPARISON")
    print("=" * 60)
    
    # Load model
    model, class_names = load_model_and_classes()
    
    # Test all three methods
    methods = {
        "Local (OpenCV+CLAHE)": preprocess_local_method,
        "PIL-based": preprocess_pil_method, 
        "No CLAHE": preprocess_no_clahe_method
    }
    
    results = {}
    
    for method_name, preprocess_func in methods.items():
        try:
            processed = preprocess_func(image_path)
            predictions = model.predict(processed, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            predicted_class = class_names[predicted_idx]
            
            results[method_name] = {
                'class': predicted_class,
                'confidence': confidence,
                'all_probs': predictions[0]
            }
            
            print(f"\n🎯 {method_name} RESULT:")
            print(f"   Predicted: {predicted_class} ({confidence:.1%})")
            
        except Exception as e:
            print(f"\n❌ {method_name} FAILED: {e}")
            results[method_name] = {'error': str(e)}
    
    # Compare results
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 60)
    
    for method_name, result in results.items():
        if 'error' not in result:
            print(f"{method_name:20s}: {result['class']:15s} ({result['confidence']:.1%})")
        else:
            print(f"{method_name:20s}: ERROR - {result['error']}")
    
    # Check if results differ
    classes = [r['class'] for r in results.values() if 'class' in r]
    if len(set(classes)) > 1:
        print(f"\n⚠️  DIFFERENT RESULTS DETECTED!")
        print(f"   This explains why Colab gives different results!")
        
        # Show detailed probability differences
        print(f"\nDetailed probability comparison:")
        for i, class_name in enumerate(class_names):
            print(f"\n{class_name}:")
            for method_name, result in results.items():
                if 'all_probs' in result:
                    prob = result['all_probs'][i]
                    print(f"   {method_name:20s}: {prob:.4f} ({prob:.1%})")
    else:
        print(f"\n✅ All methods give the same result: {classes[0]}")
    
    return results

def generate_colab_fix_code(image_path):
    """Generate the exact code to use in Colab based on findings"""
    print(f"\n🔧 GENERATING COLAB FIX CODE")
    print("=" * 60)
    
    # Test which method gives the correct local result
    results = compare_preprocessing_methods(image_path)
    
    print(f"\n📝 COPY THIS CODE TO GOOGLE COLAB:")
    print("=" * 60)
    
    colab_code = '''
# EXACT PREPROCESSING FIX FOR GOOGLE COLAB
# This code ensures identical preprocessing to your local environment

import cv2
import numpy as np
from tensorflow import keras
import json

def preprocess_image_exact(image_path):
    """Exact preprocessing matching local environment"""
    # Read image with OpenCV (handles color spaces correctly)
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB (critical step!)
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
    return np.expand_dims(normalized, axis=0)

# Load your model
model = keras.models.load_model('/content/drive/MyDrive/models/mobilenetv2_efficientnet_style_96x96/best_model.h5')
class_names = ['early_blight_leaf', 'healthy_leaf', 'late_blight_leaf', 'septoria_leaf', 'unknown']

# Test with your problematic image
image_path = 'ESP32CAM_540983.jpg'  # Replace with your uploaded image name
processed = preprocess_image_exact(image_path)
predictions = model.predict(processed)

predicted_idx = np.argmax(predictions[0])
confidence = predictions[0][predicted_idx]
predicted_class = class_names[predicted_idx]

print(f"Prediction: {predicted_class} ({confidence:.1%})")
print("All probabilities:")
for i, class_name in enumerate(class_names):
    print(f"  {class_name}: {predictions[0][i]:.4f} ({predictions[0][i]:.1%})")
'''
    
    print(colab_code)

if __name__ == "__main__":
    # Test with the same image that worked locally
    test_image = r"C:\Users\Azief\Documents\GitHub\Thesis-Project\raw_dataset\unknown\ESP32CAM_540983.jpg"
    
    if os.path.exists(test_image):
        print("🔍 DEBUGGING COLAB PREPROCESSING DIFFERENCE")
        print("=" * 60)
        print(f"Testing with: {test_image}")
        
        results = compare_preprocessing_methods(test_image)
        generate_colab_fix_code(test_image)
        
    else:
        print(f"❌ Test image not found: {test_image}")
        print("Please update the test_image path in the script.")