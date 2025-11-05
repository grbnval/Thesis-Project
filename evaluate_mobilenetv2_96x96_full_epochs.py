import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import time
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

# NOTE: This script uses preprocessed test data from 'processed_test_96x96'
# The preprocessing includes CLAHE enhancement (clipLimit=3.0, tileGridSize=(8,8))
# which matches the training preprocessing pipeline used in preprocess_dataset_96x96.py

# Constants
IMAGE_SIZE = 96
MODEL_DIR = "models/mobilenetv2_96x96_full_epochs_with_unknown"
TEST_DATA_DIR = "processed_test_96x96"
BATCH_SIZE = 32

def load_model(model_path):
    """Load trained model"""
    try:
        # Try loading best model first (checkpoint)
        h5_path = os.path.join(model_path, "best_model.h5")
        if os.path.exists(h5_path):
            print(f"Loading model from: {h5_path}")
            model = tf.keras.models.load_model(h5_path)
            return model
        
        # Try loading final model
        keras_path = os.path.join(model_path, "final_model.keras")
        if os.path.exists(keras_path):
            print(f"Loading model from: {keras_path}")
            model = tf.keras.models.load_model(keras_path)
            return model
        
        # Try loading regular keras model
        keras_path = os.path.join(model_path, "model.keras")
        if os.path.exists(keras_path):
            print(f"Loading model from: {keras_path}")
            model = tf.keras.models.load_model(keras_path)
            return model
        
        raise FileNotFoundError(f"No model file found in {model_path}")
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def load_class_info(model_dir):
    """Load class information from model directory"""
    class_info_path = os.path.join(model_dir, "class_info.json")
    if not os.path.exists(class_info_path):
        raise FileNotFoundError(f"Class info file not found: {class_info_path}")
        
    with open(class_info_path, 'r') as f:
        class_info = json.load(f)
        
    return class_info

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

def load_and_preprocess_test_data(test_dir, class_names):
    """Load preprocessed test data from .npy files (already CLAHE processed)"""
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Check if all classes from class_info exist in the test directory
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class '{class_name}' not found in test directory")
    
    images = []
    labels = []
    image_paths = []  # Store paths for reference
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Get all .npy files (preprocessed images)
        files = [f for f in os.listdir(class_dir) 
                if f.lower().endswith('.npy')]
        
        if not files:
            print(f"Warning: No .npy files found in {class_name}/")
            continue
            
        print(f"Loading {len(files)} preprocessed test images from {class_name}/")
        
        # Load each preprocessed image
        for img_file in tqdm(files, desc=f"Loading {class_name}"):
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Load the preprocessed numpy array (already CLAHE enhanced and normalized)
                img = np.load(img_path)
                
                # Add to dataset
                images.append(img)
                labels.append(class_idx)
                image_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    if not images:
        raise ValueError("No images found in any class folder!")
    
    print(f"\nLoaded {len(images)} preprocessed images (already CLAHE enhanced)")
    return np.array(images), np.array(labels), image_paths

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

def calculate_class_accuracies(true_labels, pred_labels, class_names):
    """Calculate accuracy for each class"""
    class_accuracies = []
    
    for class_idx, class_name in enumerate(class_names):
        # Get indices for this class
        class_indices = np.where(true_labels == class_idx)[0]
        
        if len(class_indices) > 0:
            # Calculate accuracy for this class
            class_pred = pred_labels[class_indices]
            class_true = true_labels[class_indices]
            accuracy = np.mean(class_pred == class_true)
            class_accuracies.append((class_name, accuracy))
    
    return class_accuracies

def generate_false_negative_report(true_labels, pred_labels, class_names, output_dir):
    """Generate detailed false negative report for each class"""
    print("\n=== FALSE NEGATIVE ANALYSIS ===")
    
    fn_report = []
    
    for class_idx, class_name in enumerate(class_names):
        # Get indices where true label is this class
        true_class_indices = np.where(true_labels == class_idx)[0]
        
        if len(true_class_indices) > 0:
            # Get predictions for this true class
            predictions_for_class = pred_labels[true_class_indices]
            
            # Find false negatives (true class but predicted as other classes)
            false_negatives = np.where(predictions_for_class != class_idx)[0]
            fn_indices = true_class_indices[false_negatives]
            
            total_samples = len(true_class_indices)
            fn_count = len(fn_indices)
            fn_rate = fn_count / total_samples if total_samples > 0 else 0
            
            print(f"\n{class_name.upper()}:")
            print(f"  Total samples: {total_samples}")
            print(f"  False negatives: {fn_count}")
            print(f"  False negative rate: {fn_rate:.4f} ({fn_rate*100:.2f}%)")
            
            # Analyze what these false negatives were predicted as
            if fn_count > 0:
                fn_predictions = predictions_for_class[false_negatives]
                print(f"  Misclassified as:")
                
                misclassified_counts = {}
                for pred_class_idx in fn_predictions:
                    pred_class_name = class_names[pred_class_idx]
                    misclassified_counts[pred_class_name] = misclassified_counts.get(pred_class_name, 0) + 1
                
                for pred_class, count in sorted(misclassified_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / fn_count) * 100
                    print(f"    - {pred_class}: {count} samples ({percentage:.1f}%)")
            
            fn_report.append({
                'true_class': class_name,
                'total_samples': total_samples,
                'false_negatives': fn_count,
                'fn_rate': fn_rate,
                'misclassified_as': misclassified_counts if fn_count > 0 else {}
            })
    
    # Save false negative report
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    with open(os.path.join(output_dir, "evaluation", "false_negative_report.txt"), 'w', encoding='utf-8') as f:
        f.write("FALSE NEGATIVE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for report in fn_report:
            f.write(f"{report['true_class'].upper()}:\n")
            f.write(f"  Total samples: {report['total_samples']}\n")
            f.write(f"  False negatives: {report['false_negatives']}\n")
            f.write(f"  False negative rate: {report['fn_rate']:.4f} ({report['fn_rate']*100:.2f}%)\n")
            
            if report['misclassified_as']:
                f.write(f"  Misclassified as:\n")
                for pred_class, count in sorted(report['misclassified_as'].items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / report['false_negatives']) * 100
                    f.write(f"    - {pred_class}: {count} samples ({percentage:.1f}%)\n")
            f.write("\n")
    
    return fn_report

def generate_false_positive_report(true_labels, pred_labels, class_names, output_dir):
    """Generate detailed false positive report for each class"""
    print("\n=== FALSE POSITIVE ANALYSIS ===")
    
    fp_report = []
    
    for class_idx, class_name in enumerate(class_names):
        # Get indices where prediction is this class
        pred_class_indices = np.where(pred_labels == class_idx)[0]
        
        if len(pred_class_indices) > 0:
            # Get true labels for these predictions
            true_labels_for_pred = true_labels[pred_class_indices]
            
            # Find false positives (predicted as this class but true label is different)
            false_positives = np.where(true_labels_for_pred != class_idx)[0]
            fp_indices = pred_class_indices[false_positives]
            
            total_predictions = len(pred_class_indices)
            fp_count = len(fp_indices)
            fp_rate = fp_count / total_predictions if total_predictions > 0 else 0
            
            # Calculate precision (true positives / (true positives + false positives))
            true_positives = total_predictions - fp_count
            precision = true_positives / total_predictions if total_predictions > 0 else 0
            
            print(f"\n{class_name.upper()}:")
            print(f"  Total predictions: {total_predictions}")
            print(f"  False positives: {fp_count}")
            print(f"  False positive rate: {fp_rate:.4f} ({fp_rate*100:.2f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            
            # Analyze what these false positives actually were
            if fp_count > 0:
                fp_true_labels = true_labels_for_pred[false_positives]
                print(f"  Actually were:")
                
                actual_counts = {}
                for true_class_idx in fp_true_labels:
                    true_class_name = class_names[true_class_idx]
                    actual_counts[true_class_name] = actual_counts.get(true_class_name, 0) + 1
                
                for true_class, count in sorted(actual_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / fp_count) * 100
                    print(f"    - {true_class}: {count} samples ({percentage:.1f}%)")
            
            fp_report.append({
                'predicted_class': class_name,
                'total_predictions': total_predictions,
                'false_positives': fp_count,
                'fp_rate': fp_rate,
                'precision': precision,
                'actually_were': actual_counts if fp_count > 0 else {}
            })
    
    # Save false positive report
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    with open(os.path.join(output_dir, "evaluation", "false_positive_report.txt"), 'w', encoding='utf-8') as f:
        f.write("FALSE POSITIVE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for report in fp_report:
            f.write(f"{report['predicted_class'].upper()}:\n")
            f.write(f"  Total predictions: {report['total_predictions']}\n")
            f.write(f"  False positives: {report['false_positives']}\n")
            f.write(f"  False positive rate: {report['fp_rate']:.4f} ({report['fp_rate']*100:.2f}%)\n")
            f.write(f"  Precision: {report['precision']:.4f} ({report['precision']*100:.2f}%)\n")
            
            if report['actually_were']:
                f.write(f"  Actually were:\n")
                for true_class, count in sorted(report['actually_were'].items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / report['false_positives']) * 100
                    f.write(f"    - {true_class}: {count} samples ({percentage:.1f}%)\n")
            f.write("\n")
    
    return fp_report

def plot_false_negative_positive_rates(fn_report, fp_report, output_dir):
    """Plot false negative and false positive rates for each class"""
    class_names = [report['true_class'] for report in fn_report]
    fn_rates = [report['fn_rate'] for report in fn_report]
    fp_rates = [report['fp_rate'] for report in fp_report]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, fn_rates, width, label='False Negative Rate', color='red', alpha=0.7)
    plt.bar(x + width/2, fp_rates, width, label='False Positive Rate', color='blue', alpha=0.7)
    
    plt.xlabel('Classes')
    plt.ylabel('Rate')
    plt.title('False Negative and False Positive Rates by Class')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.0)
    
    # Add value labels on bars
    for i, (fn_rate, fp_rate) in enumerate(zip(fn_rates, fp_rates)):
        plt.text(i - width/2, fn_rate + 0.01, f'{fn_rate:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, fp_rate + 0.01, f'{fp_rate:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation", "false_negative_positive_rates.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_class_confusion_analysis(true_labels, pred_labels, class_names, output_dir):
    """Generate detailed confusion analysis between specific class pairs"""
    print("\n=== CLASS CONFUSION ANALYSIS ===")
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    confusion_pairs = []
    
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j and cm[i][j] > 0:  # Only consider off-diagonal elements (misclassifications)
                confusion_count = cm[i][j]
                total_true_class = np.sum(cm[i, :])  # Total samples of true class i
                confusion_rate = confusion_count / total_true_class
                
                confusion_pairs.append({
                    'true_class': true_class,
                    'predicted_as': pred_class,
                    'count': confusion_count,
                    'rate': confusion_rate
                })
    
    # Sort by confusion rate (highest first)
    confusion_pairs.sort(key=lambda x: x['rate'], reverse=True)
    
    print("\nMost common misclassifications:")
    for i, pair in enumerate(confusion_pairs[:10]):  # Show top 10
        print(f"{i+1:2d}. {pair['true_class']} -> {pair['predicted_as']}: "
              f"{pair['count']} samples ({pair['rate']*100:.1f}%)")
    
    # Save confusion analysis
    with open(os.path.join(output_dir, "evaluation", "confusion_analysis.txt"), 'w', encoding='utf-8') as f:
        f.write("CLASS CONFUSION ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write("Most common misclassifications (True Class -> Predicted Class):\n\n")
        
        for i, pair in enumerate(confusion_pairs):
            f.write(f"{i+1:2d}. {pair['true_class']} -> {pair['predicted_as']}: "
                   f"{pair['count']} samples ({pair['rate']*100:.1f}%)\n")
    
    return confusion_pairs

def plot_class_accuracies(class_accuracies, output_dir):
    """Plot class-wise accuracies"""
    class_names = [c[0] for c in class_accuracies]
    accuracies = [c[1] for c in class_accuracies]
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, accuracies)
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy')
    
    # Add accuracy values on top of bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation", "class_accuracies.png"))
    plt.close()

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
            with open(os.path.join(results_dir, report_filename), 'w', encoding='utf-8') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Average inference time: {avg_inference_time:.2f} ms per image\n")
                f.write(f"Model size: {os.path.getsize(tflite_path) / (1024 * 1024):.2f} MB\n\n")
                f.write("Classification Report:\n")
                f.write(report)
                
                # Add FN/FP analysis for TFLite models
                print(f"\nGenerating FN/FP analysis for {model_name}...")
                fn_report_tflite = generate_false_negative_report(y_test, pred_labels, class_names, results_dir)
                fp_report_tflite = generate_false_positive_report(y_test, pred_labels, class_names, results_dir)
                
                f.write("\n\n" + "="*50 + "\n")
                f.write("FALSE NEGATIVE SUMMARY:\n")
                for fn in fn_report_tflite:
                    f.write(f"{fn['true_class']}: {fn['false_negatives']}/{fn['total_samples']} "
                           f"({fn['fn_rate']*100:.1f}% FN rate)\n")
                f.write("\n" + "="*50 + "\n")
                f.write("FALSE POSITIVE SUMMARY:\n")
                for fp in fp_report_tflite:
                    f.write(f"{fp['predicted_class']}: {fp['false_positives']}/{fp['total_predictions']} "
                           f"({fp['fp_rate']*100:.1f}% FP rate, {fp['precision']*100:.1f}% precision)\n")
            
            return predictions
        
        return None
    
    # Evaluate standard TFLite model
    standard_predictions = evaluate_single_tflite(tflite_path, "Standard TFLite", is_quantized=False)
    
    # Evaluate quantized TFLite model
    quantized_predictions = evaluate_single_tflite(quant_path, "Quantized TFLite", is_quantized=True)
    
    # Return predictions for potential visualization
    return standard_predictions, quantized_predictions

def test_on_single_image(model, class_names, image_path):
    """Test model on a single image"""
    try:
        # Preprocess image
        processed_img = preprocess_image(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        
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
        print("=== EVALUATING MOBILENETV2 96x96 MODEL (FULL EPOCHS) ===")
        
        # Set up paths
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(PROJECT_ROOT, MODEL_DIR)
        TEST_DIR = os.path.join(PROJECT_ROOT, TEST_DATA_DIR)
        
        print(f"Looking for model in: {MODEL_DIR}")
        print(f"Looking for test data in: {TEST_DIR}")
        
        # Load model
        model = load_model(MODEL_DIR)
        model.summary()
        
        # Load class information
        class_info = load_class_info(MODEL_DIR)
        class_names = class_info["classes"]
        print(f"Classes: {class_names}")
        
        # Load and preprocess test data
        X_test, y_test, test_image_paths = load_and_preprocess_test_data(TEST_DIR, class_names)
        print(f"Loaded and preprocessed {len(X_test)} test images from {len(class_names)} classes")
        
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
        
        # Calculate and plot class accuracies
        class_accuracies = calculate_class_accuracies(y_test, pred_labels, class_names)
        plot_class_accuracies(class_accuracies, MODEL_DIR)
        
        # Generate false negative and false positive reports
        print("\nGenerating false negative and false positive analysis...")
        fn_report = generate_false_negative_report(y_test, pred_labels, class_names, MODEL_DIR)
        fp_report = generate_false_positive_report(y_test, pred_labels, class_names, MODEL_DIR)
        
        # Plot FN/FP rates
        plot_false_negative_positive_rates(fn_report, fp_report, MODEL_DIR)
        
        # Generate class confusion analysis
        confusion_pairs = generate_class_confusion_analysis(y_test, pred_labels, class_names, MODEL_DIR)
        
        # Generate classification report
        print("\nClassification Report:")
        report = classification_report(y_test, pred_labels, target_names=class_names)
        print(report)
        
        # Save classification report
        os.makedirs(os.path.join(MODEL_DIR, "evaluation"), exist_ok=True)
        with open(os.path.join(MODEL_DIR, "evaluation", "classification_report.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Model: MobileNetV2 96x96 (Full Epochs with Unknown)\n")
            f.write(f"Test accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test loss: {test_loss:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\n" + "="*50 + "\n")
            f.write("FALSE NEGATIVE SUMMARY:\n")
            for fn in fn_report:
                f.write(f"{fn['true_class']}: {fn['false_negatives']}/{fn['total_samples']} "
                       f"({fn['fn_rate']*100:.1f}% FN rate)\n")
            f.write("\n" + "="*50 + "\n")
            f.write("FALSE POSITIVE SUMMARY:\n")
            for fp in fp_report:
                f.write(f"{fp['predicted_class']}: {fp['false_positives']}/{fp['total_predictions']} "
                       f"({fp['fp_rate']*100:.1f}% FP rate, {fp['precision']*100:.1f}% precision)\n")
        
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