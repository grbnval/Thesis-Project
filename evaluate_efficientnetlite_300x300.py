import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import time

# Constants
IMAGE_SIZE = 300
BATCH_SIZE = 32
MODEL_DIR = "models/efficientnetlite_300x300_separate"
PROCESSED_TEST_DIR = "processed_test_300x300"
EVALUATION_DIR = os.path.join(MODEL_DIR, "evaluation")

def load_data(data_dir):
    """Load preprocessed data from directory"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Get class names
    class_names = sorted([d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))])
    
    if not class_names:
        raise ValueError(f"No class directories found in {data_dir}")
    
    # Create label mapping
    label_map = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # Initialize arrays
    images = []
    labels = []
    
    # Load data for each class
    for class_name in class_names:
        print(f"Loading test data for class: {class_name}")
        
        # Get numpy files
        class_dir = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        if not files:
            print(f"Warning: No .npy files found in {class_name}/")
            continue
        
        print(f"Found {len(files)} test images")
        
        # Load each file
        for file in files:
            file_path = os.path.join(class_dir, file)
            img = np.load(file_path)
            images.append(img)
            labels.append(label_map[class_name])
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Convert labels to categorical
    labels_categorical = tf.keras.utils.to_categorical(labels, len(class_names))
    
    print(f"Loaded {len(images)} test images from {len(class_names)} classes")
    
    return images, labels, labels_categorical, class_names

def evaluate_model(model, test_images, test_labels, class_names):
    """Evaluate model and return metrics"""
    # Start time
    start_time = time.time()
    
    # Evaluate model
    print("\nEvaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
    
    # Get predictions
    y_pred_probs = model.predict(test_images, batch_size=BATCH_SIZE)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(test_labels, axis=1)
    
    # Calculate evaluation time
    eval_time = time.time() - start_time
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    
    # Create classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save results
    results = {
        "accuracy": float(test_accuracy),
        "loss": float(test_loss),
        "evaluation_time_seconds": eval_time,
        "num_test_samples": len(test_images),
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return results, report, cm, y_true, y_pred, y_pred_probs

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_predictions(test_images, y_true, y_pred, class_names):
    """Create visualization of correct and incorrect predictions"""
    # Create directories
    os.makedirs(os.path.join(EVALUATION_DIR, "visualizations", "correct"), exist_ok=True)
    os.makedirs(os.path.join(EVALUATION_DIR, "visualizations", "incorrect"), exist_ok=True)
    
    # Sample some images
    indices = np.random.choice(len(test_images), min(25, len(test_images)), replace=False)
    
    # Create grid visualization
    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        plt.imshow(test_images[idx].astype('uint8'))
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        title_color = "green" if true_class == pred_class else "red"
        plt.title(f"True: {true_class}\nPred: {pred_class}", color=title_color)
        plt.axis("off")
        
        # Save individual images
        img_filename = f"sample_{idx}_true_{true_class}_pred_{pred_class}.png"
        correct_dir = "correct" if true_class == pred_class else "incorrect"
        img_save_path = os.path.join(EVALUATION_DIR, "visualizations", correct_dir, img_filename)
        plt.figure(figsize=(5, 5))
        plt.imshow(test_images[idx].astype('uint8'))
        plt.title(f"True: {true_class}\nPred: {pred_class}", color=title_color)
        plt.axis("off")
        plt.savefig(img_save_path)
        plt.close()
    
    # Save the grid visualization
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, "visualizations", "prediction_samples.png"))
    plt.close()

def evaluate_tflite_model(tflite_path, test_images, test_labels, class_names):
    """Evaluate TFLite model performance"""
    # Start time
    start_time = time.time()
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get predictions
    y_true = np.argmax(test_labels, axis=1)
    y_pred = []
    
    for i in range(len(test_images)):
        # Process input data
        input_data = np.expand_dims(test_images[i], axis=0).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(np.argmax(output[0]))
    
    # Convert to numpy array
    y_pred = np.array(y_pred)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)
    
    # Calculate evaluation time
    eval_time = time.time() - start_time
    
    return accuracy, eval_time

def main():
    try:
        print("=== EVALUATING EFFICIENTNET LITE 300x300 MODEL ===")
        
        # Create evaluation directory
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        os.makedirs(os.path.join(EVALUATION_DIR, "tflite"), exist_ok=True)
        os.makedirs(os.path.join(EVALUATION_DIR, "visualizations"), exist_ok=True)
        
        # Load test data
        print(f"Loading test data from: {PROCESSED_TEST_DIR}")
        test_images, test_labels_raw, test_labels, class_names = load_data(PROCESSED_TEST_DIR)
        
        # Load model
        print(f"Loading Keras model from: {os.path.join(MODEL_DIR, 'model.keras')}")
        model = load_model(os.path.join(MODEL_DIR, 'model.keras'))
        
        # Check for phase-specific models
        phase1_path = os.path.join(MODEL_DIR, 'best_model_phase1.h5')
        phase2_path = os.path.join(MODEL_DIR, 'best_model_phase2.h5')
        
        phase_comparison = {}
        
        if os.path.exists(phase1_path):
            print(f"Found Phase 1 model: {phase1_path}")
            
            print("\n=== EVALUATING PHASE 1 MODEL (TOP LAYERS ONLY) ===")
            phase1_model = load_model(phase1_path)
            phase1_loss, phase1_accuracy = phase1_model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
            print(f"Phase 1 model - Test accuracy: {phase1_accuracy:.4f}, Test loss: {phase1_loss:.4f}")
            
            phase_comparison["phase1"] = {
                "accuracy": float(phase1_accuracy),
                "loss": float(phase1_loss)
            }
        
        if os.path.exists(phase2_path):
            print(f"Found Phase 2 model: {phase2_path}")
            
            print("\n=== EVALUATING PHASE 2 MODEL (FINE-TUNED) ===")
            phase2_model = load_model(phase2_path)
            phase2_loss, phase2_accuracy = phase2_model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
            print(f"Phase 2 model - Test accuracy: {phase2_accuracy:.4f}, Test loss: {phase2_loss:.4f}")
            
            phase_comparison["phase2"] = {
                "accuracy": float(phase2_accuracy),
                "loss": float(phase2_loss)
            }
        
        # Save phase comparison
        if phase_comparison:
            with open(os.path.join(EVALUATION_DIR, "phase_comparison.json"), 'w') as f:
                json.dump(phase_comparison, f, indent=4)
        
        print("\n=== EVALUATING FINAL MODEL ===")
        
        # Evaluate model
        results, report, cm, y_true, y_pred, y_pred_probs = evaluate_model(model, test_images, test_labels, class_names)
        
        # Save results
        with open(os.path.join(EVALUATION_DIR, "evaluation_results.json"), 'w') as f:
            json.dump(results, f, indent=4)
        
        with open(os.path.join(EVALUATION_DIR, "classification_report.json"), 'w') as f:
            json.dump(report, f, indent=4)
        
        # Plot and save confusion matrix
        plot_confusion_matrix(cm, class_names, os.path.join(EVALUATION_DIR, "confusion_matrix.png"))
        
        # Visualize predictions
        print("\nSaving individual prediction visualizations...")
        visualize_predictions(test_images, y_true, y_pred, class_names)
        
        # Evaluate TFLite models
        standard_tflite_path = os.path.join(MODEL_DIR, 'model.tflite')
        quantized_tflite_path = os.path.join(MODEL_DIR, 'model_quantized.tflite')
        
        # Standard TFLite model
        print("\nEvaluating standard TFLite model...")
        standard_accuracy, standard_time = evaluate_tflite_model(standard_tflite_path, test_images, test_labels, class_names)
        print(f"Standard TFLite model accuracy: {standard_accuracy:.4f}")
        print(f"Standard TFLite model evaluation time: {standard_time:.2f} seconds")
        
        # Quantized TFLite model
        print("\nEvaluating quantized TFLite model...")
        quantized_accuracy, quantized_time = evaluate_tflite_model(quantized_tflite_path, test_images, test_labels, class_names)
        print(f"Quantized TFLite model accuracy: {quantized_accuracy:.4f}")
        print(f"Quantized TFLite model evaluation time: {quantized_time:.2f} seconds")
        
        # Calculate speedup
        speedup = standard_time / quantized_time
        print(f"Speedup from quantization: {speedup:.1f}x")
        
        # Save TFLite results
        tflite_results = {
            "standard": {
                "accuracy": float(standard_accuracy),
                "evaluation_time_seconds": standard_time
            },
            "quantized": {
                "accuracy": float(quantized_accuracy),
                "evaluation_time_seconds": quantized_time
            },
            "speedup": float(speedup)
        }
        
        with open(os.path.join(EVALUATION_DIR, "tflite", "tflite_evaluation.json"), 'w') as f:
            json.dump(tflite_results, f, indent=4)
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Final model accuracy: {results['accuracy']:.4f}")
        
        if "phase1" in phase_comparison:
            print(f"Phase 1 model accuracy: {phase_comparison['phase1']['accuracy']:.4f}")
        
        if "phase2" in phase_comparison:
            print(f"Phase 2 model accuracy: {phase_comparison['phase2']['accuracy']:.4f}")
        
        print(f"Standard TFLite model accuracy: {standard_accuracy:.4f}")
        print(f"Quantized TFLite model accuracy: {quantized_accuracy:.4f}")
        print(f"Quantization speedup: {speedup:.1f}x")
        
        print("\nEvaluation results saved to:")
        print(f"- {EVALUATION_DIR}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print(f"1. The model has been trained and saved at {MODEL_DIR}")
        print(f"2. The test data exists at {PROCESSED_TEST_DIR}")
        raise

if __name__ == "__main__":
    main()
