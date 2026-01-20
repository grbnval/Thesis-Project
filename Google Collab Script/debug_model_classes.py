# Debug Model Output - Check actual class mapping
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os

def debug_model_classes():
    """Debug the actual model output to understand class mapping"""
    
    print("🔍 DEBUGGING MODEL CLASS MAPPING")
    print("=" * 50)
    
    # Load model
    model_path = "models/mobilenetv2_efficientnet_style_96x96/best_model.h5"
    model = keras.models.load_model(model_path)
    
    print(f"📦 Model loaded from: {model_path}")
    
    # Get model output info more safely
    try:
        output_shape = model.outputs[0].shape
        print(f"📊 Model output shape: {output_shape}")
        print(f"🎯 Number of output classes: {output_shape[-1]}")
        model_classes = output_shape[-1]
    except:
        # Try to get from model layers
        last_layer = model.layers[-1]
        print(f"📊 Last layer: {last_layer.name}")
        print(f"📊 Last layer config: {last_layer.get_config()}")
        if hasattr(last_layer, 'units'):
            model_classes = last_layer.units
            print(f"🎯 Number of output classes (from units): {model_classes}")
        else:
            model_classes = None
            print(f"❌ Could not determine number of classes")
    
    # Load class info
    class_info_path = "models/mobilenetv2_efficientnet_style_96x96/class_info.json"
    if os.path.exists(class_info_path):
        with open(class_info_path, 'r') as f:
            class_info = json.load(f)
        print(f"📋 Class info loaded from JSON:")
        if 'classes' in class_info:
            expected_classes = class_info['classes']
        else:
            expected_classes = list(class_info.keys())
        print(f"   Expected classes: {expected_classes}")
        print(f"   Expected count: {len(expected_classes)}")
    else:
        expected_classes = ['early_blight_leaf', 'healthy_leaf', 'late_blight_leaf', 'septoria_leaf', 'unknown']
        print(f"📋 Using fallback classes: {expected_classes}")
        print(f"   Fallback count: {len(expected_classes)}")
    
    # Check if counts match
    expected_count = len(expected_classes)
    
    if model_classes is not None:
        if model_classes != expected_count:
            print(f"⚠️  CLASS COUNT MISMATCH!")
            print(f"   Model expects: {model_classes} classes")
            print(f"   We have: {expected_count} classes")
            print(f"   This explains the wrong predictions!")
        else:
            print(f"✅ Class counts match: {model_classes}")
    else:
        print(f"⚠️  Could not verify class count matching")
    
    # Test with a dummy prediction to see actual output
    print(f"\n🧪 Testing with dummy input...")
    dummy_input = np.random.random((1, 96, 96, 3)).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)
    
    print(f"📊 Prediction shape: {predictions.shape}")
    print(f"📊 Prediction values: {predictions[0]}")
    print(f"📊 Prediction sum: {np.sum(predictions[0]):.4f} (should be ~1.0 for softmax)")
    
    # Show which index has highest probability
    max_idx = np.argmax(predictions[0])
    print(f"🎯 Highest probability at index: {max_idx}")
    
    if max_idx < len(expected_classes):
        print(f"🎯 This maps to class: {expected_classes[max_idx]}")
    else:
        print(f"❌ Index {max_idx} is out of range for {len(expected_classes)} classes!")
    
    return model, expected_classes

def test_with_real_image():
    """Test with the actual image to see what's happening"""
    
    print(f"\n🖼️  TESTING WITH REAL IMAGE")
    print("=" * 50)
    
    import cv2
    
    # Load model and classes
    model, expected_classes = debug_model_classes()
    
    # Test image path
    test_image = r"C:\Users\Azief\Documents\GitHub\Thesis-Project\raw_dataset\unknown\ESP32CAM_540983.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📸 Testing with: {os.path.basename(test_image)}")
    
    # Preprocess image (exact same as training)
    image = cv2.imread(test_image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    image_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Resize and normalize
    resized = cv2.resize(image_enhanced, (96, 96), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    final = np.expand_dims(normalized, axis=0)
    
    # Predict
    predictions = model.predict(final, verbose=0)
    
    print(f"\n📊 RAW PREDICTION VALUES:")
    for i, prob in enumerate(predictions[0]):
        class_name = expected_classes[i] if i < len(expected_classes) else f"Unknown_Index_{i}"
        print(f"   Index {i} ({class_name:15s}): {prob:.6f} ({prob:.2%})")
    
    # Find top prediction
    max_idx = np.argmax(predictions[0])
    max_confidence = predictions[0][max_idx]
    
    print(f"\n🎯 TOP PREDICTION:")
    print(f"   Index: {max_idx}")
    print(f"   Confidence: {max_confidence:.6f} ({max_confidence:.2%})")
    
    if max_idx < len(expected_classes):
        predicted_class = expected_classes[max_idx]
        print(f"   Class: {predicted_class}")
        
        # Check if this matches what we expect
        if predicted_class == "unknown":
            print(f"✅ CORRECT! Predicted 'unknown' as expected")
        else:
            print(f"❌ WRONG! Expected 'unknown' but got '{predicted_class}'")
    else:
        print(f"❌ FATAL ERROR: Index {max_idx} out of range!")

if __name__ == "__main__":
    test_with_real_image()