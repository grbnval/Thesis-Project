import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import json
from datetime import datetime

# Constants
IMAGE_SIZE = 300
BATCH_SIZE = 32
PROCESSED_TEST_DIR = "processed_test_300x300"
MODEL_DIR = "models/mobilenetv2_300x300_separate"

def load_test_data(data_dir):
    """Load preprocessed test data from directory"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Test directory not found: {data_dir}")
    
    # Get class names from directory
    class_names = sorted([d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))])
    
    if not class_names:
        raise ValueError(f"No class directories found in {data_dir}")
    
    # Create label mapping
    label_map = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # Initialize arrays
    images = []
    labels = []
    filenames = []
    
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
            filenames.append(os.path.join(class_name, file))
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Convert labels to categorical
    labels_categorical = tf.keras.utils.to_categorical(labels, len(class_names))
    
    return images, labels, labels_categorical, class_names, filenames

def load_model_and_metadata():
    """Load trained model and metadata"""
    # Check model directory
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    
    # Load model
    keras_model_path = os.path.join(MODEL_DIR, 'model.keras')
    h5_model_path = os.path.join(MODEL_DIR, 'best_model.h5')
    
    if os.path.exists(keras_model_path):
        print(f"Loading Keras model from: {keras_model_path}")
        model = load_model(keras_model_path)
    elif os.path.exists(h5_model_path):
        print(f"Loading H5 model from: {h5_model_path}")
        model = load_model(h5_model_path)
    else:
        raise FileNotFoundError(f"No model file found in {MODEL_DIR}")
    
    # Check for phase-specific models
    phase1_model_path = os.path.join(MODEL_DIR, 'best_model_phase1.h5')
    phase2_model_path = os.path.join(MODEL_DIR, 'best_model_phase2.h5')
    
    phase_models = {}
    
    if os.path.exists(phase1_model_path):
        print(f"Found Phase 1 model: {phase1_model_path}")
        phase_models['phase1'] = phase1_model_path
    
    if os.path.exists(phase2_model_path):
        print(f"Found Phase 2 model: {phase2_model_path}")
        phase_models['phase2'] = phase2_model_path
    
    # Load class names
    class_names_path = os.path.join(MODEL_DIR, 'class_names.txt')
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        class_names = None
    
    # Load metadata if available
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = None
    
    return model, class_names, metadata, phase_models

def evaluate_model(model, test_images, test_labels, test_labels_raw, class_names):
    """Evaluate model performance"""
    # Create eval directory
    eval_dir = os.path.join(MODEL_DIR, 'evaluation')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    
    # Evaluate model
    print("\nEvaluating model on test data...")
    start_time = time.time()
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
    eval_time = time.time() - start_time
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    
    # Get predictions
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_labels_raw
    
    # Generate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Generate classification report
    report = classification_report(true_classes, predicted_classes, 
                                   target_names=class_names, output_dict=True)
    
    # Save classification report
    with open(os.path.join(eval_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Save evaluation results
    eval_results = {
        "accuracy": float(test_accuracy),
        "loss": float(test_loss),
        "evaluation_time_seconds": float(eval_time),
        "num_test_samples": len(test_images),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(eval_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    return predicted_classes, eval_results

def visualize_predictions(test_images, test_labels_raw, predicted_classes, 
                          class_names, filenames, num_samples=20):
    """Visualize predictions for sample images"""
    # Create visualization directory
    viz_dir = os.path.join(MODEL_DIR, 'evaluation', 'visualizations')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Select random samples
    indices = np.random.choice(len(test_images), 
                              min(num_samples, len(test_images)), 
                              replace=False)
    
    # Create figure
    num_cols = 5
    num_rows = (len(indices) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    axes = axes.flatten()
    
    # Plot each sample
    for i, idx in enumerate(indices):
        img = test_images[idx]
        true_label = test_labels_raw[idx]
        pred_label = predicted_classes[idx]
        
        # Plot image
        axes[i].imshow(img)
        
        # Set title color based on prediction
        color = 'green' if true_label == pred_label else 'red'
        
        # Set title
        axes[i].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", 
                         color=color)
        
        # Remove axis ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'prediction_samples.png'))
    plt.close()
    
    # Create individual prediction images
    correct_dir = os.path.join(viz_dir, 'correct')
    incorrect_dir = os.path.join(viz_dir, 'incorrect')
    
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    
    # Process all test images
    print(f"\nSaving individual prediction visualizations...")
    for i, (img, true_label, pred_label, filename) in enumerate(
        zip(test_images, test_labels_raw, predicted_classes, filenames)):
        
        # Determine if prediction is correct
        is_correct = true_label == pred_label
        
        # Skip most samples to avoid creating too many files
        if is_correct and i % 20 != 0:  # Save only 5% of correct predictions
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Plot image
        ax.imshow(img)
        
        # Set title color based on prediction
        color = 'green' if is_correct else 'red'
        
        # Set title
        ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", 
                    color=color)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save figure
        save_dir = correct_dir if is_correct else incorrect_dir
        save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(filename))[0]}.png")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def evaluate_tflite_models(test_images, test_labels_raw, class_names):
    """Evaluate TFLite models"""
    # Create TFLite eval directory
    tflite_eval_dir = os.path.join(MODEL_DIR, 'evaluation', 'tflite')
    if not os.path.exists(tflite_eval_dir):
        os.makedirs(tflite_eval_dir)
    
    # Check for TFLite models
    tflite_path = os.path.join(MODEL_DIR, 'model.tflite')
    tflite_quant_path = os.path.join(MODEL_DIR, 'model_quantized.tflite')
    
    tflite_results = {}
    
    # Evaluate standard TFLite model
    if os.path.exists(tflite_path):
        print("\nEvaluating standard TFLite model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Check if input quantization is present
        input_scale, input_zero_point = 0, 0
        if input_details[0]['quantization'] != (0.0, 0):
            input_scale, input_zero_point = input_details[0]['quantization']
        
        # Evaluate model
        start_time = time.time()
        predictions = []
        
        for i in range(len(test_images)):
            # Process input data based on model requirements
            input_data = test_images[i:i+1]  # Add batch dimension
            
            # Quantize input if needed
            if input_details[0]['dtype'] == np.uint8:
                input_data = input_data / input_scale + input_zero_point
                input_data = input_data.astype(np.uint8)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Store prediction
            predictions.append(output[0])
        
        eval_time = time.time() - start_time
        
        # Convert predictions to class indices
        predictions = np.array(predictions)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(predicted_classes == test_labels_raw)
        
        print(f"Standard TFLite model accuracy: {accuracy:.4f}")
        print(f"Standard TFLite model evaluation time: {eval_time:.2f} seconds")
        
        # Generate confusion matrix
        cm = confusion_matrix(test_labels_raw, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Standard TFLite Model Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(tflite_eval_dir, 'standard_tflite_confusion_matrix.png'))
        plt.close()
        
        # Store results
        tflite_results["standard"] = {
            "accuracy": float(accuracy),
            "evaluation_time_seconds": float(eval_time),
            "file_size_mb": os.path.getsize(tflite_path) / (1024 * 1024)
        }
    
    # Evaluate quantized TFLite model
    if os.path.exists(tflite_quant_path):
        print("\nEvaluating quantized TFLite model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_quant_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Check if input quantization is present
        input_scale, input_zero_point = 0, 0
        if input_details[0]['quantization'] != (0.0, 0):
            input_scale, input_zero_point = input_details[0]['quantization']
        
        # Evaluate model
        start_time = time.time()
        predictions = []
        
        for i in range(len(test_images)):
            # Process input data based on model requirements
            input_data = test_images[i:i+1]  # Add batch dimension
            
            # Quantize input if needed
            if input_details[0]['dtype'] == np.uint8:
                # Convert to 0-255 range first
                input_data = input_data * 255.0
                # Then apply quantization parameters
                input_data = input_data / input_scale + input_zero_point
                input_data = input_data.astype(np.uint8)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Store prediction
            predictions.append(output[0])
        
        eval_time = time.time() - start_time
        
        # Convert predictions to class indices
        predictions = np.array(predictions)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(predicted_classes == test_labels_raw)
        
        print(f"Quantized TFLite model accuracy: {accuracy:.4f}")
        print(f"Quantized TFLite model evaluation time: {eval_time:.2f} seconds")
        
        # Generate confusion matrix
        cm = confusion_matrix(test_labels_raw, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Quantized TFLite Model Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(tflite_eval_dir, 'quantized_tflite_confusion_matrix.png'))
        plt.close()
        
        # Calculate speedup
        if "standard" in tflite_results:
            speedup = tflite_results["standard"]["evaluation_time_seconds"] / eval_time
            print(f"Speedup from quantization: {speedup:.1f}x")
        
        # Store results
        tflite_results["quantized"] = {
            "accuracy": float(accuracy),
            "evaluation_time_seconds": float(eval_time),
            "file_size_mb": os.path.getsize(tflite_quant_path) / (1024 * 1024)
        }
        
        if "standard" in tflite_results:
            tflite_results["quantized"]["speedup"] = float(speedup)
    
    # Save TFLite evaluation results
    if tflite_results:
        with open(os.path.join(tflite_eval_dir, 'tflite_evaluation_results.json'), 'w') as f:
            json.dump(tflite_results, f, indent=4)
    
    return tflite_results

def main():
    try:
        print("=== EVALUATING MOBILENETV2 300x300 MODEL ===")
        
        # Load test data
        print(f"\nLoading test data from: {PROCESSED_TEST_DIR}")
        test_images, test_labels_raw, test_labels, class_names, filenames = load_test_data(PROCESSED_TEST_DIR)
        print(f"Loaded {len(test_images)} test images from {len(class_names)} classes")
        
        # Load model
        model, model_class_names, metadata, phase_models = load_model_and_metadata()
        
        # Check if class names match
        if model_class_names and set(class_names) != set(model_class_names):
            print("\nWarning: Model and test data have different classes!")
            print(f"Model classes: {model_class_names}")
            print(f"Test data classes: {class_names}")
            
            # Use model's class names if available
            if model_class_names:
                print("Using model's class names for evaluation...")
                class_names = model_class_names
        
        # Evaluate each phase model if available
        phase_results = {}
        
        if 'phase1' in phase_models:
            print("\n=== EVALUATING PHASE 1 MODEL (TOP LAYERS ONLY) ===")
            # Create a copy of the model with phase 1 weights
            phase1_model = tf.keras.models.clone_model(model)
            phase1_model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            phase1_model.load_weights(phase_models['phase1'])
            
            # Evaluate phase 1 model
            phase1_loss, phase1_accuracy = phase1_model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
            print(f"Phase 1 model - Test accuracy: {phase1_accuracy:.4f}, Test loss: {phase1_loss:.4f}")
            
            phase_results['phase1'] = {
                'accuracy': float(phase1_accuracy),
                'loss': float(phase1_loss)
            }
        
        if 'phase2' in phase_models:
            print("\n=== EVALUATING PHASE 2 MODEL (FINE-TUNED) ===")
            # Create a copy of the model with phase 2 weights
            phase2_model = tf.keras.models.clone_model(model)
            phase2_model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            phase2_model.load_weights(phase_models['phase2'])
            
            # Evaluate phase 2 model
            phase2_loss, phase2_accuracy = phase2_model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
            print(f"Phase 2 model - Test accuracy: {phase2_accuracy:.4f}, Test loss: {phase2_loss:.4f}")
            
            phase_results['phase2'] = {
                'accuracy': float(phase2_accuracy),
                'loss': float(phase2_loss)
            }
        
        # Evaluate final keras model
        print("\n=== EVALUATING FINAL MODEL ===")
        predicted_classes, eval_results = evaluate_model(
            model, test_images, test_labels, test_labels_raw, class_names)
        
        # Add phase comparison to results
        if phase_results:
            eval_dir = os.path.join(MODEL_DIR, 'evaluation')
            with open(os.path.join(eval_dir, 'phase_comparison.json'), 'w') as f:
                json.dump(phase_results, f, indent=4)
        
        # Visualize predictions
        visualize_predictions(
            test_images, test_labels_raw, predicted_classes, class_names, filenames)
        
        # Evaluate TFLite models
        tflite_results = evaluate_tflite_models(test_images, test_labels_raw, class_names)
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Final model accuracy: {eval_results['accuracy']:.4f}")
        
        if 'phase1' in phase_results:
            print(f"Phase 1 model accuracy: {phase_results['phase1']['accuracy']:.4f}")
        
        if 'phase2' in phase_results:
            print(f"Phase 2 model accuracy: {phase_results['phase2']['accuracy']:.4f}")
        
        if "standard" in tflite_results:
            print(f"Standard TFLite model accuracy: {tflite_results['standard']['accuracy']:.4f}")
        
        if "quantized" in tflite_results:
            print(f"Quantized TFLite model accuracy: {tflite_results['quantized']['accuracy']:.4f}")
            if "speedup" in tflite_results["quantized"]:
                print(f"Quantization speedup: {tflite_results['quantized']['speedup']:.1f}x")
        
        print("\nEvaluation results saved to:")
        print(f"- {os.path.join(MODEL_DIR, 'evaluation')}")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        print("\nPlease ensure:")
        print(f"1. The model has been trained and saved to {MODEL_DIR}")
        print(f"2. The processed test data exists in {PROCESSED_TEST_DIR}")
        raise

if __name__ == "__main__":
    main()
