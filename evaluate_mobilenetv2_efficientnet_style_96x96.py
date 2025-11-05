import tensorflow as tf
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

# NOTE: This script uses preprocessed test data from 'processed_test_96x96'
# The preprocessing includes CLAHE enhancement (clipLimit=3.0, tileGridSize=(8,8))
# which matches the training preprocessing pipeline used in preprocess_dataset_96x96.py

# Constants
IMAGE_SIZE = 96
MODEL_DIR = "models/mobilenetv2_efficientnet_style_96x96"
TEST_DATA_DIR = "processed_test_96x96"
BATCH_SIZE = 32

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_FULL = os.path.join(PROJECT_ROOT, MODEL_DIR)
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, TEST_DATA_DIR)

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
        
    # Return just the classes list
    if 'classes' in class_info:
        return class_info['classes']
    else:
        raise ValueError(f"No 'classes' key found in class_info.json")
    
def load_full_class_info(model_dir):
    """Load full class information dictionary from model directory"""
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
                print(f"Error loading {img_path}: {str(e)}")
                continue
    
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
                color = 'green'
                plt.title(f"✓ {pred_class}\n{confidence:.3f}", color=color, fontsize=10)
            else:
                color = 'red'
                plt.title(f"✗ {true_class}->{pred_class}\n{confidence:.3f}", color=color, fontsize=10)
            
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
            true_class_predictions = pred_labels[true_class_indices]
            
            # Find false negatives (where true class was predicted as something else)
            false_negative_indices = true_class_indices[true_class_predictions != class_idx]
            
            fn_count = len(false_negative_indices)
            total_count = len(true_class_indices)
            fn_rate = fn_count / total_count
            
            # Analyze what this class was confused with
            if fn_count > 0:
                confused_predictions = pred_labels[false_negative_indices]
                unique_predictions, counts = np.unique(confused_predictions, return_counts=True)
                
                confusion_details = []
                for pred_class_idx, count in zip(unique_predictions, counts):
                    confusion_details.append({
                        'predicted_as': class_names[pred_class_idx],
                        'count': count,
                        'percentage': (count / fn_count) * 100
                    })
                
                # Sort by count (highest first)
                confusion_details.sort(key=lambda x: x['count'], reverse=True)
            else:
                confusion_details = []
            
            fn_report.append({
                'true_class': class_name,
                'fn_count': fn_count,
                'total_count': total_count,
                'fn_rate': fn_rate,
                'confusion_details': confusion_details
            })
            
            print(f"{class_name}: {fn_count}/{total_count} false negatives ({fn_rate:.3f})")
            for detail in confusion_details[:3]:  # Show top 3
                print(f"  -> {detail['predicted_as']}: {detail['count']} ({detail['percentage']:.1f}%)")
    
    # Save false negative report
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    with open(os.path.join(output_dir, "evaluation", "false_negative_report.txt"), 'w', encoding='utf-8') as f:
        f.write("FALSE NEGATIVE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for report in fn_report:
            f.write(f"Class: {report['true_class']}\n")
            f.write(f"False Negatives: {report['fn_count']}/{report['total_count']} ({report['fn_rate']:.3f})\n")
            f.write("Most commonly confused with:\n")
            
            for detail in report['confusion_details']:
                f.write(f"  -> {detail['predicted_as']}: {detail['count']} samples ({detail['percentage']:.1f}%)\n")
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
            pred_class_true_labels = true_labels[pred_class_indices]
            
            # Find false positives (where something else was predicted as this class)
            false_positive_indices = pred_class_indices[pred_class_true_labels != class_idx]
            
            fp_count = len(false_positive_indices)
            total_pred_count = len(pred_class_indices)
            fp_rate = fp_count / total_pred_count if total_pred_count > 0 else 0
            
            # Analyze what was confused as this class
            if fp_count > 0:
                confused_true_labels = true_labels[false_positive_indices]
                unique_labels, counts = np.unique(confused_true_labels, return_counts=True)
                
                confusion_details = []
                for true_class_idx, count in zip(unique_labels, counts):
                    confusion_details.append({
                        'actually_was': class_names[true_class_idx],
                        'count': count,
                        'percentage': (count / fp_count) * 100
                    })
                
                # Sort by count (highest first)
                confusion_details.sort(key=lambda x: x['count'], reverse=True)
            else:
                confusion_details = []
            
            fp_report.append({
                'predicted_class': class_name,
                'fp_count': fp_count,
                'total_pred_count': total_pred_count,
                'fp_rate': fp_rate,
                'confusion_details': confusion_details
            })
            
            print(f"{class_name}: {fp_count}/{total_pred_count} false positives ({fp_rate:.3f})")
            for detail in confusion_details[:3]:  # Show top 3
                print(f"  <- {detail['actually_was']}: {detail['count']} ({detail['percentage']:.1f}%)")
    
    # Save false positive report
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    with open(os.path.join(output_dir, "evaluation", "false_positive_report.txt"), 'w', encoding='utf-8') as f:
        f.write("FALSE POSITIVE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for report in fp_report:
            f.write(f"Predicted Class: {report['predicted_class']}\n")
            f.write(f"False Positives: {report['fp_count']}/{report['total_pred_count']} ({report['fp_rate']:.3f})\n")
            f.write("Actually was:\n")
            
            for detail in report['confusion_details']:
                f.write(f"  <- {detail['actually_was']}: {detail['count']} samples ({detail['percentage']:.1f}%)\n")
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
            if i != j and cm[i, j] > 0:  # Only non-diagonal (error) entries
                total_true = cm[i, :].sum()
                confusion_pairs.append({
                    'true_class': true_class,
                    'predicted_as': pred_class,
                    'count': cm[i, j],
                    'rate': cm[i, j] / total_true if total_true > 0 else 0
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
            print(f"{model_name} model not found: {tflite_path}")
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
            input_scale = input_details[0]['quantization'][0]
            input_zero_point = input_details[0]['quantization'][1]
            print(f"Quantization parameters - Scale: {input_scale}, Zero point: {input_zero_point}")
        
        # Prepare for predictions
        predictions = []
        inference_times = []
        
        # Run inference on all test images
        for i in tqdm(range(len(X_test))):
            input_data = X_test[i:i+1].copy()
            
            # Prepare input based on quantization
            if is_quantized and input_dtype == np.uint8:
                # For quantized models, convert float32 to uint8
                input_data = input_data / input_scale + input_zero_point
                input_data = np.clip(input_data, 0, 255).astype(np.uint8)
            else:
                input_data = input_data.astype(np.float32)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            start_time = time.time()
            interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output_data[0])
        
        # Convert to numpy array if we have predictions
        if predictions:
            predictions = np.array(predictions)
            pred_labels = np.argmax(predictions, axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(pred_labels == y_test)
            
            # Calculate average inference time
            avg_inference_time = np.mean(inference_times)
            
            print(f"{model_name} Accuracy: {accuracy:.4f}")
            print(f"Average inference time: {avg_inference_time:.2f} ms")
            
            # Get model size
            model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
            print(f"Model size: {model_size:.2f} MB")
            
            # Save detailed results
            results = {
                'model_name': model_name,
                'accuracy': float(accuracy),
                'avg_inference_time_ms': float(avg_inference_time),
                'model_size_mb': float(model_size),
                'total_samples': len(y_test)
            }
            
            # Classification report
            report = classification_report(y_test, pred_labels, target_names=class_names, output_dict=True)
            
            # Save results to file
            results_file = os.path.join(results_dir, f"{model_name.lower().replace(' ', '_')}_results.txt")
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write(f"{model_name} EVALUATION RESULTS\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Average Inference Time: {avg_inference_time:.2f} ms\n")
                f.write(f"Model Size: {model_size:.2f} MB\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(y_test, pred_labels, target_names=class_names))
            
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
        
        return pred_class, confidence, inference_time
    
    except Exception as e:
        print(f"Error testing image: {str(e)}")
        return None, None, None

def plot_roc_curve(y_test_cat, y_pred_prob, class_names, output_dir):
    """Plot and save ROC curves for each class"""
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i, class_name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_test_cat[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
    
    # Plot the diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    # Save figure
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "evaluation", "roc_curves.png")
    plt.savefig(roc_path)
    print(f"ROC curves saved to: {roc_path}")
    plt.close()

def plot_precision_recall_curve(y_test_cat, y_pred_prob, class_names, output_dir):
    """Plot and save precision-recall curves for each class"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    plt.figure(figsize=(10, 8))
    
    # Compute precision-recall curve for each class
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_cat[:, i], y_pred_prob[:, i])
        avg_precision = average_precision_score(y_test_cat[:, i], y_pred_prob[:, i])
        
        plt.plot(recall, precision, lw=2,
                 label=f'{class_name} (AP = {avg_precision:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    
    # Save figure
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(output_dir, "evaluation", "precision_recall_curves.png")
    plt.savefig(pr_path)
    print(f"Precision-recall curves saved to: {pr_path}")
    plt.close()

def save_evaluation_summary(metrics, class_names, model_info, output_dir):
    """Save evaluation metrics summary to a text file"""
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    summary_path = os.path.join(output_dir, "evaluation", "evaluation_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=== MODEL EVALUATION SUMMARY ===\n\n")
        
        # Overall model score summary (concise version at the top)
        f.write("OVERALL MODEL SCORE: {:.2f}%\n".format(metrics['accuracy'] * 100))
        f.write("=======================\n\n")
        
        # Model information
        f.write("MODEL INFORMATION:\n")
        f.write(f"Model Type: {model_info.get('model_type', 'Unknown')}\n")
        f.write(f"Resolution: {model_info.get('resolution', 'Unknown')}\n")
        f.write(f"Training Type: {model_info.get('training_type', 'Unknown')}\n")
        f.write(f"Date Trained: {model_info.get('date_trained', 'Unknown')}\n")
        f.write(f"Date Evaluated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Get overall metrics from classification report
        overall_metrics = {}
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in metrics['class_report']:
                overall_metrics[avg_type] = metrics['class_report'][avg_type]
        
        # Overall metrics section with more comprehensive metrics
        f.write("OVERALL METRICS:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (Weighted): {metrics['precision']:.4f}\n")
        f.write(f"Recall (Weighted): {metrics['recall']:.4f}\n")
        f.write(f"F1 Score (Weighted): {metrics['f1']:.4f}\n\n")
        
        # Additional overall metrics from classification report
        f.write("DETAILED OVERALL METRICS:\n")
        
        if 'macro avg' in overall_metrics:
            macro = overall_metrics['macro avg']
            f.write("Macro Average (arithmetic mean, each class has equal weight):\n")
            f.write(f"  Precision: {macro['precision']:.4f}\n")
            f.write(f"  Recall: {macro['recall']:.4f}\n")
            f.write(f"  F1 Score: {macro['f1-score']:.4f}\n")
            f.write(f"  Support: {macro['support']}\n\n")
        
        if 'weighted avg' in overall_metrics:
            weighted = overall_metrics['weighted avg']
            f.write("Weighted Average (accounts for class imbalance):\n")
            f.write(f"  Precision: {weighted['precision']:.4f}\n")
            f.write(f"  Recall: {weighted['recall']:.4f}\n")
            f.write(f"  F1 Score: {weighted['f1-score']:.4f}\n")
            f.write(f"  Support: {weighted['support']}\n\n")
        
        # Class-specific metrics
        f.write("CLASS-SPECIFIC METRICS:\n")
        for i, class_name in enumerate(class_names):
            class_metrics = metrics['class_report'][class_name]
            f.write(f"Class: {class_name}\n")
            f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {class_metrics['f1-score']:.4f}\n")
            f.write(f"  Support: {class_metrics['support']}\n\n")
    
    print(f"Evaluation summary saved to: {summary_path}")
    
    # Also save as JSON for easier programmatic access
    json_path = os.path.join(output_dir, "evaluation", "evaluation_metrics.json")
    
    # Convert NumPy arrays to lists for JSON serialization
    json_metrics = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'class_report': metrics['class_report'],
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"Evaluation metrics (JSON) saved to: {json_path}")

if __name__ == "__main__":
    try:
        print("=== EVALUATING MOBILENETV2 (EFFICIENTNET-STYLE) 96x96 MODEL ===")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Model directory: {MODEL_DIR_FULL}")
        print(f"Test dataset directory: {TEST_DATA_DIR}")
        
        # Check directories
        if not os.path.exists(PROJECT_ROOT):
            raise FileNotFoundError(f"Project root directory not found: {PROJECT_ROOT}")
        
        if not os.path.exists(TEST_DATA_DIR):
            print(f"Test dataset directory not found: {TEST_DATA_DIR}")
            print("Please ensure the test_dataset folder exists with class subdirectories.")
            sys.exit(1)
        
        # Check model directory
        if not os.path.exists(MODEL_DIR_FULL):
            raise FileNotFoundError(f"Model directory not found: {MODEL_DIR_FULL}")
            
        print(f"\nStarting evaluation for model in: {MODEL_DIR_FULL}")
        
        # Load model
        model = load_model(MODEL_DIR_FULL)
        
        # Load class names
        class_names = load_class_info(MODEL_DIR_FULL)
        print(f"Classes: {class_names}")
        
        # Also load full class info for metadata
        class_info = load_full_class_info(MODEL_DIR_FULL)
        
        # Load and preprocess test data
        print("\nLoading and preprocessing test data...")
        X_test, y_test, image_paths = load_and_preprocess_test_data(TEST_DATA_DIR, class_names)
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Number of test samples: {len(y_test)}")
        
        print("\nEvaluating model...")
        start_time = time.time()
        
        # Get predictions
        y_pred_prob = model.predict(X_test, batch_size=32, verbose=1)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calculate overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nOverall Test Accuracy: {accuracy:.4f}")
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Create evaluation directory
        eval_dir = os.path.join(MODEL_DIR_FULL, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Calculate class-wise accuracies
        class_accuracies = calculate_class_accuracies(y_test, y_pred, class_names)
        print("\nClass-wise Accuracies:")
        for class_name, acc in class_accuracies:
            print(f"{class_name}: {acc:.4f}")
        
        # Generate confusion matrix
        plot_confusion_matrix(y_test, y_pred, class_names, MODEL_DIR_FULL)
        
        # Generate and visualize predictions
        visualize_predictions(X_test, y_test, y_pred_prob, class_names, image_paths, MODEL_DIR_FULL)
        
        # Generate false negative analysis
        fn_report = generate_false_negative_report(y_test, y_pred, class_names, MODEL_DIR_FULL)
        
        # Generate false positive analysis
        fp_report = generate_false_positive_report(y_test, y_pred, class_names, MODEL_DIR_FULL)
        
        # Plot false negative and false positive rates
        plot_false_negative_positive_rates(fn_report, fp_report, MODEL_DIR_FULL)
        
        # Generate confusion analysis between classes
        confusion_analysis = generate_class_confusion_analysis(y_test, y_pred, class_names, MODEL_DIR_FULL)
        
        # Plot class accuracies
        plot_class_accuracies(class_accuracies, MODEL_DIR_FULL)
        
        # Evaluate TFLite models if they exist
        print("\n" + "="*50)
        print("EVALUATING TFLITE MODELS")
        print("="*50)
        evaluate_tflite_model(MODEL_DIR_FULL, X_test, y_test, class_names)
        
        # Save comprehensive evaluation report
        with open(os.path.join(eval_dir, "comprehensive_evaluation.txt"), 'w', encoding='utf-8') as f:
            f.write("MOBILENETV2 EFFICIENTNET STYLE MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model Directory: {MODEL_DIR_FULL}\n")
            f.write(f"Test Dataset: {TEST_DATA_DIR}\n")
            f.write(f"Model Type: {class_info.get('model_type', 'MobileNetV2 EfficientNet-style')}\n")
            f.write(f"Training Date: {class_info.get('date_trained', 'Unknown')}\n")
            f.write(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")
            f.write(f"Number of Classes: {len(class_names)}\n")
            f.write(f"Total Test Samples: {len(y_test)}\n\n")
            
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            
            f.write("Class-wise Accuracies:\n")
            for class_name, acc in class_accuracies:
                f.write(f"{class_name}: {acc:.4f}\n")
            f.write("\n")
            
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred, target_names=class_names))
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        evaluation_time = time.time() - start_time
        print(f"\nEvaluation time: {evaluation_time:.2f} seconds")
        print(f"\nDetailed results saved in: {MODEL_DIR_FULL}/evaluation/")
        
        # Example of testing on a single image (commented out)
        # print("\n" + "="*50)
        # print("SINGLE IMAGE TEST EXAMPLE")
        # print("="*50)
        # # Uncomment and modify path to test specific image
        # # test_image_path = "path/to/your/test/image.jpg"
        # # if os.path.exists(test_image_path):
        # #     test_on_single_image(model, class_names, test_image_path)
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("\nPlease ensure the following:")
        print(f"1. The model directory exists: {MODEL_DIR_FULL}")
        print(f"2. The model files exist in the model directory")
        print(f"3. The test dataset folder exists: {TEST_DATA_DIR}")
        print("4. The test dataset has proper class subdirectories")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()