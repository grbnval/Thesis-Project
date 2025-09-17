import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import time
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm import tqdm

# Constants
IMAGE_SIZE = 300
PROCESSED_DATA_DIR = "processed_dataset_300x300"
MODEL_DIR = "models/mobilenetv2_300x300"
BATCH_SIZE = 32
TEST_SPLIT = 0.2  # Should match the value used in training

def load_model(model_path):
    """Load trained model"""
    try:
        # Try loading from .h5 file first (checkpoint)
        h5_path = os.path.join(model_path, "best_model.h5")
        if os.path.exists(h5_path):
            print(f"Loading model from: {h5_path}")
            model = tf.keras.models.load_model(h5_path)
            return model
        
        # Try loading from .keras file
        keras_path = os.path.join(model_path, "model.keras")
        if os.path.exists(keras_path):
            print(f"Loading model from: {keras_path}")
            model = tf.keras.models.load_model(keras_path)
            return model
        
        raise FileNotFoundError(f"No model file found in {model_path}")
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def load_test_data(test_dir):
    """Load test data from preprocessed dataset"""
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    class_names = sorted(os.listdir(test_dir))
    if not class_names:
        raise ValueError(f"No class folders found in {test_dir}")
    
    images = []
    labels = []
    image_paths = []  # Store paths for visualization
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        if not files:
            print(f"Warning: No .npy files found in {class_name}/")
            continue
            
        print(f"Loading {len(files)} test images from {class_name}/")
        for img_file in files:
            img_path = os.path.join(class_dir, img_file)
            img = np.load(img_path)
            images.append(img)
            labels.append(class_idx)
            image_paths.append(img_path)
    
    if not images:
        raise ValueError("No images found in any class folder!")
    
    return np.array(images), np.array(labels), class_names, image_paths

def plot_confusion_matrix(true_labels, pred_labels, class_names, output_dir):
    """Generate and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot with seaborn
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    
    # Save plot
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation", "confusion_matrix.png"))
    plt.close()
    
    # Save raw confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Counts)')
    plt.savefig(os.path.join(output_dir, "evaluation", "confusion_matrix_raw.png"))
    plt.close()

def visualize_predictions(X_test, y_test, predictions, class_names, image_paths, output_dir):
    """Visualize sample predictions (correct and incorrect)"""
    pred_labels = np.argmax(predictions, axis=1)
    
    # Get indices of correct and incorrect predictions
    correct_indices = np.where(pred_labels == y_test)[0]
    incorrect_indices = np.where(pred_labels != y_test)[0]
    
    # Create samples directory
    samples_dir = os.path.join(output_dir, "evaluation", "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Function to save grid of images
    def save_sample_grid(indices, title, filename):
        if len(indices) == 0:
            return
        
        # Take at most 20 samples
        sample_indices = indices[:min(20, len(indices))]
        n_samples = len(sample_indices)
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        rows = cols = grid_size
        
        plt.figure(figsize=(15, 15))
        for i, idx in enumerate(sample_indices):
            if i >= rows * cols:
                break
                
            plt.subplot(rows, cols, i + 1)
            plt.imshow(X_test[idx])
            
            true_class = class_names[y_test[idx]]
            pred_class = class_names[pred_labels[idx]]
            confidence = predictions[idx][pred_labels[idx]]
            
            if true_class == pred_class:
                plt.title(f"{true_class}\nConf: {confidence:.2f}", 
                          color='green', fontsize=10)
            else:
                plt.title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}", 
                          color='red', fontsize=9)
            
            plt.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(samples_dir, filename))
        plt.close()
    
    # Save samples of correct predictions
    save_sample_grid(
        correct_indices, 
        "Correct Predictions",
        "correct_predictions.png"
    )
    
    # Save samples of incorrect predictions
    save_sample_grid(
        incorrect_indices, 
        "Incorrect Predictions",
        "incorrect_predictions.png"
    )
    
    # Save class-specific results
    for class_idx, class_name in enumerate(class_names):
        # Get indices for this class
        class_indices = np.where(y_test == class_idx)[0]
        
        # Correct predictions for this class
        correct_class = np.intersect1d(correct_indices, class_indices)
        
        # Incorrect predictions for this class
        incorrect_class = np.intersect1d(incorrect_indices, class_indices)
        
        # Save correct samples for this class
        save_sample_grid(
            correct_class,
            f"Correct Predictions - {class_name}",
            f"correct_{class_name}.png"
        )
        
        # Save incorrect samples for this class
        save_sample_grid(
            incorrect_class,
            f"Incorrect Predictions - {class_name}",
            f"incorrect_{class_name}.png"
        )

def evaluate_tflite_model(model_dir, X_test, y_test, class_names):
    """Evaluate TFLite model performance"""
    tflite_path = os.path.join(model_dir, "model.tflite")
    quant_path = os.path.join(model_dir, "model_quantized.tflite")
    
    # Create results directory
    results_dir = os.path.join(model_dir, "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # Function to evaluate a single TFLite model
    def evaluate_single_tflite(tflite_path, model_name, is_quantized=False):
        if not os.path.exists(tflite_path):
            print(f"Warning: {model_name} model not found at {tflite_path}")
            return None
        
        print(f"\nEvaluating {model_name} model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Check input type
        input_dtype = input_details[0]['dtype']
        print(f"Model input type: {input_dtype}")
        input_scale = 0
        input_zero_point = 0
        
        if is_quantized:
            # Get quantization parameters
            input_scale, input_zero_point = input_details[0]['quantization']
            print(f"Quantization scale: {input_scale}, zero point: {input_zero_point}")
        
        # Prepare for predictions
        predictions = []
        inference_times = []
        
        # Run inference on all test images
        for i in tqdm(range(len(X_test))):
            # Prepare input based on quantization
            if is_quantized:
                # For quantized model, convert float input to uint8
                input_data = X_test[i]
                # Scale input from [0,1] to [0,255]
                input_data = input_data * 255
                # Convert to uint8
                input_data = input_data.astype(np.uint8)
                # Add batch dimension
                input_data = np.expand_dims(input_data, axis=0)
            else:
                # For regular model, use float32
                input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
            
            try:
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)
                
                # Run inference and measure time
                start_time = time.time()
                interpreter.invoke()
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
                
                # Get output tensor
                output = interpreter.get_tensor(output_details[0]['index'])
                
                # If quantized, dequantize the output
                if is_quantized and output_details[0]['dtype'] != np.float32:
                    scale, zero_point = output_details[0]['quantization']
                    output = (output.astype(np.float32) - zero_point) * scale
                
                predictions.append(output[0])
            
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                if i == 0:  # If first image fails, stop processing
                    print("Failed to process first image, stopping evaluation.")
                    return None
        
        # Convert to numpy array if we have predictions
        if predictions:
            predictions = np.array(predictions)
            
            # Calculate metrics
            pred_labels = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_labels == y_test)
            avg_inference_time = np.mean(inference_times)
            
            # Generate classification report
            report = classification_report(y_test, pred_labels, target_names=class_names)
            
            # Print results
            print(f"{model_name} model accuracy: {accuracy:.4f}")
            print(f"Average inference time: {avg_inference_time:.2f} ms per image")
            print(f"Model size: {os.path.getsize(tflite_path) / (1024 * 1024):.2f} MB")
            
            # Save report to file
            report_filename = f"{model_name.lower().replace(' ', '_')}_report.txt"
            with open(os.path.join(results_dir, report_filename), 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Average inference time: {avg_inference_time:.2f} ms per image\n")
                f.write(f"Model size: {os.path.getsize(tflite_path) / (1024 * 1024):.2f} MB\n\n")
                f.write("Classification Report:\n")
                f.write(report)
            
            return predictions
        
        return None
    
    # Evaluate standard TFLite model
    standard_predictions = evaluate_single_tflite(tflite_path, "Standard TFLite", is_quantized=False)
    
    # Evaluate quantized TFLite model
    quantized_predictions = evaluate_single_tflite(quant_path, "Quantized TFLite", is_quantized=True)
    
    # Return predictions for potential visualization
    return standard_predictions, quantized_predictions

def preprocess_image(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Preprocess a single image for inference"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB (from BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Resize
    resized_img = cv2.resize(enhanced_img, target_size)
    
    # Normalize to [0,1]
    normalized_img = resized_img / 255.0
    
    return normalized_img

def test_on_single_image(model, class_names, image_path):
    """Test model on a single image"""
    try:
        # Preprocess image
        processed_img = preprocess_image(image_path)
        
        # Add batch dimension
        input_img = np.expand_dims(processed_img, axis=0)
        
        # Make prediction
        start_time = time.time()
        prediction = model.predict(input_img)[0]
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get predicted class and confidence
        pred_class_idx = np.argmax(prediction)
        pred_class = class_names[pred_class_idx]
        confidence = prediction[pred_class_idx]
        
        # Print results
        print(f"\nPrediction for {os.path.basename(image_path)}:")
        print(f"Class: {pred_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Inference time: {inference_time:.2f} ms")
        
        # Visualize
        plt.figure(figsize=(8, 8))
        plt.imshow(processed_img)
        plt.title(f"Prediction: {pred_class}\nConfidence: {confidence:.4f}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return pred_class, confidence
    
    except Exception as e:
        print(f"Error testing image: {str(e)}")
        return None, None

if __name__ == "__main__":
    try:
        print("=== EVALUATING MOBILENETV2 300x300 MODEL ===")
        
        # Set up paths
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, PROCESSED_DATA_DIR)
        MODEL_DIR = os.path.join(PROJECT_ROOT, MODEL_DIR)
        
        print(f"Looking for processed data in: {PROCESSED_DIR}")
        print(f"Looking for model in: {MODEL_DIR}")
        
        # Load all processed data
        X, y, class_names, image_paths = load_test_data(PROCESSED_DIR)
        print(f"Loaded {len(X)} processed images from {len(class_names)} classes")
        
        # Split to get the test set (use the same split ratio as training)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
        )
        
        # Get the corresponding image paths for the test set
        _, test_image_paths = train_test_split(
            image_paths, test_size=TEST_SPLIT, random_state=42, stratify=y
        )
        
        # Load model
        model = load_model(MODEL_DIR)
        model.summary()
        
        # Get test data indices
        print(f"Using {TEST_SPLIT*100:.0f}% of data for testing ({len(X_test)} images)")
        
        # Convert labels to categorical for evaluation
        y_test_cat = tf.keras.utils.to_categorical(y_test, len(class_names))
        
        # Evaluate model
        print("\nEvaluating model on test data...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, batch_size=BATCH_SIZE)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Generate predictions
        print("\nGenerating predictions...")
        predictions = model.predict(X_test, batch_size=BATCH_SIZE)
        pred_labels = np.argmax(predictions, axis=1)
        
        # Plot confusion matrix
        print("\nGenerating confusion matrix...")
        plot_confusion_matrix(y_test, pred_labels, class_names, MODEL_DIR)
        
        # Generate classification report
        print("\nClassification Report:")
        report = classification_report(y_test, pred_labels, target_names=class_names)
        print(report)
        
        # Save classification report
        os.makedirs(os.path.join(MODEL_DIR, "evaluation"), exist_ok=True)
        with open(os.path.join(MODEL_DIR, "evaluation", "classification_report.txt"), 'w') as f:
            f.write(report)
        
        # Visualize predictions
        print("\nGenerating prediction visualizations...")
        visualize_predictions(X_test, y_test, predictions, class_names, test_image_paths, MODEL_DIR)
        
        # Evaluate TFLite models
        print("\nEvaluating TFLite models...")
        tflite_preds, quant_preds = evaluate_tflite_model(MODEL_DIR, X_test, y_test, class_names)
        
        # Interactive testing
        while True:
            test_image = input("\nEnter path to test image (or 'q' to quit): ")
            if test_image.lower() == 'q':
                break
            
            if os.path.exists(test_image):
                test_on_single_image(model, class_names, test_image)
            else:
                print(f"Error: File not found: {test_image}")
        
        print("\n=== EVALUATION COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
