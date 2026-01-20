# Google Colab Inference with Warmup - Optimized Performance
# This version eliminates cold start delays for consistent fast inference

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import time
from google.colab import drive, files
import os

class OptimizedTomatoClassifier:
    def __init__(self, model_path, class_info_path):
        """Initialize classifier with warmup for consistent performance"""
        print("🚀 Initializing Optimized Tomato Classifier...")
        
        # Load model
        print("📦 Loading model...")
        start_time = time.time()
        self.model = keras.models.load_model(model_path)
        print(f"   Model loaded in {time.time() - start_time:.2f}s")
        
        # Load class information
        with open(class_info_path, 'r') as f:
            self.class_info = json.load(f)
        
        # Pre-create CLAHE object for consistent performance
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Warmup phase - critical for consistent performance!
        self._warmup()
        
        print("✅ Classifier ready for fast inference!")
    
    def _warmup(self):
        """Warmup preprocessing and model for consistent performance"""
        print("🔥 Warming up classifier...")
        start_time = time.time()
        
        # Create dummy image for warmup
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup preprocessing (3 times to ensure all code paths are warm)
        for i in range(3):
            processed = self._preprocess_image(dummy_image)
            
        # Warmup model (5 predictions to warm up GPU/CPU caches)
        for i in range(5):
            _ = self.model.predict(processed, verbose=0)
            
        warmup_time = time.time() - start_time
        print(f"   Warmup completed in {warmup_time:.2f}s")
        print("   🎯 All subsequent predictions will be fast!")
    
    def _preprocess_image(self, image):
        """Optimized preprocessing pipeline matching training exactly"""
        # Convert BGR to RGB (if needed)
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Apply CLAHE enhancement (using pre-created object)
        if len(image_rgb.shape) == 3:
            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            image_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            image_enhanced = self.clahe.apply(image_rgb)
        
        # Resize to model input size
        resized = cv2.resize(image_enhanced, (96, 96), interpolation=cv2.INTER_LINEAR)
        
        # Normalize and add batch dimension
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)
    
    def predict_image_file(self, image_path, confidence_threshold=0.5):
        """Fast prediction with detailed timing"""
        total_start = time.time()
        
        # Read image
        read_start = time.time()
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        read_time = time.time() - read_start
        
        # Preprocess
        prep_start = time.time()
        processed_image = self._preprocess_image(image)
        prep_time = time.time() - prep_start
        
        # Predict
        pred_start = time.time()
        predictions = self.model.predict(processed_image, verbose=0)
        pred_time = time.time() - pred_start
        
        # Post-process
        post_start = time.time()
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get class name
        class_names = list(self.class_info.keys())
        predicted_class = class_names[predicted_class_idx]
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            predicted_class = "uncertain"
            
        post_time = time.time() - post_start
        total_time = time.time() - total_start
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'all_predictions': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))},
            'timing': {
                'read': read_time,
                'preprocessing': prep_time,
                'prediction': pred_time,
                'post_processing': post_time,
                'total': total_time
            }
        }

# Setup and usage example for Google Colab
def setup_colab_environment():
    """Setup the Colab environment for fast inference"""
    print("🔧 Setting up Google Colab environment...")
    
    # Mount Google Drive
    print("📁 Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # Install OpenCV if needed
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__} already installed")
    except ImportError:
        print("📦 Installing OpenCV...")
        !pip install opencv-python
        import cv2
        print(f"✅ OpenCV {cv2.__version__} installed")
    
    print("✅ Environment setup complete!")

def run_fast_inference():
    """Example usage with warmup for consistent fast performance"""
    
    # Setup environment
    setup_colab_environment()
    
    # Initialize classifier (includes automatic warmup)
    classifier = OptimizedTomatoClassifier(
        model_path='/content/drive/MyDrive/models/mobilenetv2_96x96.keras',
        class_info_path='/content/drive/MyDrive/models/class_info.json'
    )
    
    # Upload and classify images
    print("\n📤 Upload images to classify:")
    uploaded = files.upload()
    
    print("\n🔍 Classifying uploaded images...")
    for filename in uploaded.keys():
        print(f"\n📸 Processing: {filename}")
        try:
            result = classifier.predict_image_file(filename)
            
            print(f"🎯 Prediction: {result['class']} ({result['confidence']:.1%} confidence)")
            print(f"⚡ Timing: {result['timing']['total']:.3f}s total")
            print(f"   ├── Read: {result['timing']['read']:.3f}s")
            print(f"   ├── Preprocessing: {result['timing']['preprocessing']:.3f}s")
            print(f"   ├── Model: {result['timing']['prediction']:.3f}s")
            print(f"   └── Post-processing: {result['timing']['post_processing']:.3f}s")
            
            if result['confidence'] > 0.8:
                print("✅ High confidence prediction")
            elif result['confidence'] > 0.5:
                print("⚠️  Medium confidence prediction")
            else:
                print("❌ Low confidence - check image quality")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

# For direct execution in Colab
if __name__ == "__main__":
    print("🍅 Optimized Tomato Disease Classifier with Warmup")
    print("=" * 50)
    run_fast_inference()