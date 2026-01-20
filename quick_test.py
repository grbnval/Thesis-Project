import tensorflow as tf
import numpy as np
import cv2
import json
import os

def load_model_and_predict(image_path, model_dir="models/mobilenetv2_efficientnet_style_96x96"):
    """
    Load model and predict on a single image with exact same preprocessing as training
    """
    
    # Load model
    model_path = os.path.join(model_dir, "best_model.h5")
    model = tf.keras.models.load_model(model_path)
    
    # Load class names
    with open(os.path.join(model_dir, "class_info.json"), 'r') as f:
        class_info = json.load(f)
    class_names = class_info["classes"]
    
    # Preprocess image (EXACT same as training)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 96x96
    img = cv2.resize(img, (96, 96))
    
    # Apply CLAHE enhancement (same as training)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Normalize to [0,1] - CRITICAL!
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Predict
    predictions = model.predict(img, verbose=0)[0]
    
    # Get results
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    print(f"\n=== PREDICTION ===")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
    
    print(f"\nAll probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, predictions)):
        print(f"  {name}: {prob:.4f} ({prob*100:.1f}%)")
    
    return predicted_class, confidence

# Example usage:
if __name__ == "__main__":
    # Test with an image from your test dataset
    test_image = "test_dataset/healthy_leaf/H1.jpg"
    
    if os.path.exists(test_image):
        result = load_model_and_predict(test_image)
    else:
        print(f"Please provide a valid image path")
        print("Example: python quick_test.py")
        print("Then modify the test_image variable to point to your image")