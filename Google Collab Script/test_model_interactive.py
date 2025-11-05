import tensorflow as tf
import numpy as np
import cv2
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_image_for_inference(image_path, target_size=(96, 96)):
    """Preprocess image using the EXACT same method as training"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
        
    # Convert to RGB (from BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    img = cv2.resize(img, target_size)
    
    # Enhance contrast using CLAHE (same as training)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Normalize to [0,1] (CRITICAL - same as training)
    img = img.astype('float32') / 255.0
    
    return img

def load_model_and_classes(model_dir):
    """Load model and class information"""
    # Try to load the best model
    model_path = os.path.join(model_dir, "best_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.keras")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load class information
    class_info_path = os.path.join(model_dir, "class_info.json")
    if os.path.exists(class_info_path):
        with open(class_info_path, 'r') as f:
            class_info = json.load(f)
        class_names = class_info["classes"]
    else:
        # Fallback to common class names
        class_names = ["early_blight_leaf", "healthy_leaf", "late_blight_leaf", "septoria_leaf", "unknown"]
    
    return model, class_names

def predict_image(model, class_names, image_path, show_probabilities=True):
    """Predict class of a single image with detailed output"""
    # Preprocess image
    processed_img = preprocess_image_for_inference(image_path)
    if processed_img is None:
        return None, None
    
    # Add batch dimension
    input_img = np.expand_dims(processed_img, axis=0)
    
    # Make prediction
    predictions = model.predict(input_img, verbose=0)[0]
    
    # Get predicted class
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    # Display results
    print(f"\n=== PREDICTION RESULTS ===")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    if show_probabilities:
        print(f"\nAll class probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, predictions)):
            print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Show image
    plt.figure(figsize=(10, 4))
    
    # Original image
    plt.subplot(1, 2, 1)
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    plt.imshow(original_img)
    plt.title(f"Original Image")
    plt.axis('off')
    
    # Processed image (as seen by model)
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img)
    plt.title(f"Processed Image\nPrediction: {predicted_class}\nConfidence: {confidence:.2f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return predicted_class, confidence

def test_model_robustness(model, class_names, image_path):
    """Test model with slight variations to check robustness"""
    print(f"\n=== ROBUSTNESS TEST ===")
    
    # Test with different preprocessing variations
    variations = [
        ("Original preprocessing", lambda img: img),
        ("Slightly brighter", lambda img: np.clip(img * 1.1, 0, 1)),
        ("Slightly darker", lambda img: np.clip(img * 0.9, 0, 1)),
        ("No CLAHE enhancement", lambda img: preprocess_without_clahe(image_path)),
    ]
    
    for variation_name, preprocess_func in variations:
        try:
            if variation_name == "No CLAHE enhancement":
                processed_img = preprocess_func(image_path)
            else:
                processed_img = preprocess_image_for_inference(image_path)
                if processed_img is not None:
                    processed_img = preprocess_func(processed_img)
            
            if processed_img is not None:
                input_img = np.expand_dims(processed_img, axis=0)
                predictions = model.predict(input_img, verbose=0)[0]
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions)
                print(f"{variation_name}: {predicted_class} ({confidence:.3f})")
        except Exception as e:
            print(f"{variation_name}: Error - {e}")

def preprocess_without_clahe(image_path):
    """Preprocess without CLAHE to test if enhancement is causing issues"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96))
    # Skip CLAHE enhancement
    img = img.astype('float32') / 255.0
    return img

def main():
    """Main testing function"""
    print("=== MODEL TESTING TOOL ===")
    print("This tool tests your model with proper preprocessing")
    
    # Model directory
    model_dir = "models/mobilenetv2_efficientnet_style_96x96"
    
    try:
        # Load model and classes
        model, class_names = load_model_and_classes(model_dir)
        print(f"Model loaded successfully!")
        print(f"Classes: {class_names}")
        
        # Interactive testing
        while True:
            image_path = input("\nEnter path to test image (or 'quit' to exit): ").strip()
            
            if image_path.lower() in ['quit', 'q', 'exit']:
                break
                
            if not os.path.exists(image_path):
                print(f"Error: File not found - {image_path}")
                continue
            
            # Make prediction
            predicted_class, confidence = predict_image(model, class_names, image_path)
            
            # Test robustness if prediction confidence is low
            if confidence < 0.7:
                print(f"\nLow confidence detected ({confidence:.3f}). Running robustness test...")
                test_model_robustness(model, class_names, image_path)
            
            # Ask for actual class for comparison
            actual_class = input(f"\nWhat is the actual class of this image? (Enter one of {class_names} or 'skip'): ").strip()
            if actual_class.lower() != 'skip' and actual_class in class_names:
                if actual_class == predicted_class:
                    print("✅ CORRECT prediction!")
                else:
                    print(f"❌ INCORRECT prediction. Actual: {actual_class}, Predicted: {predicted_class}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you've trained the model first")
        print("2. Check that the model directory exists")
        print("3. Ensure the image path is correct")

if __name__ == "__main__":
    main()