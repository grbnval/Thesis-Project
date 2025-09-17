import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd
import time

# Constants
IMAGE_SIZE = 96
BATCH_SIZE = 32
PROCESSED_DATA_DIR = "processed_dataset_96x96"
MODEL_DIR = os.path.join("models", "mobilenetv2_efficientnet_style_96x96")

def load_preprocessed_data(data_dir):
    """Load preprocessed numpy arrays"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    class_names = sorted(os.listdir(data_dir))
    if not class_names:
        raise ValueError(f"No class folders found in {data_dir}")
    
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        if not files:
            print(f"Warning: No .npy files found in {class_name}/")
            continue
            
        print(f"Loading {len(files)} images from {class_name}/")
        for img_file in tqdm(files, desc=f"Loading {class_name}"):
            img_path = os.path.join(class_dir, img_file)
            img = np.load(img_path)
            images.append(img)
            labels.append(class_idx)
    
    if not images:
        raise ValueError("No images found in any class folder!")
    
    return np.array(images), np.array(labels), class_names

def load_model_and_info(model_dir):
    """Load the trained model and class information"""
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Try to load the model from different potential formats
    model_path = None
    for model_file in ["best_model.h5", "final_model.keras", "model.keras"]:
        potential_path = os.path.join(model_dir, model_file)
        if os.path.exists(potential_path):
            model_path = potential_path
            break
    
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {model_dir}")
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    # Load class information
    class_info_path = os.path.join(model_dir, "class_info.json")
    if not os.path.exists(class_info_path):
        raise FileNotFoundError(f"Class information file not found: {class_info_path}")
    
    with open(class_info_path, 'r') as f:
        class_info = json.load(f)
    
    return model, class_info

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate the model and calculate metrics"""
    # Predict classes
    y_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate class-specific metrics
    class_report = classification_report(y_test, y_pred, 
                                        target_names=class_names, 
                                        output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_report': class_report,
        'confusion_matrix': cm,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    
    # Save figure
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "evaluation", "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    plt.close()
    
    # Also save the raw counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Raw Counts)')
    
    plt.tight_layout()
    cm_raw_path = os.path.join(output_dir, "evaluation", "confusion_matrix_raw.png")
    plt.savefig(cm_raw_path)
    plt.close()

def plot_class_metrics(class_report, output_dir):
    """Plot and save class-specific metrics"""
    # Extract class metrics
    classes = []
    precision = []
    recall = []
    f1 = []
    
    for class_name, metrics in class_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(class_name)
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1.append(metrics['f1-score'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar chart
    bar_width = 0.25
    x = np.arange(len(classes))
    
    plt.bar(x - bar_width, precision, width=bar_width, label='Precision')
    plt.bar(x, recall, width=bar_width, label='Recall')
    plt.bar(x + bar_width, f1, width=bar_width, label='F1-Score')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Class-Specific Metrics')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, "evaluation", "class_metrics.png")
    plt.savefig(metrics_path)
    print(f"Class metrics plot saved to: {metrics_path}")
    plt.close()
    
    # Save as CSV for reference
    csv_path = os.path.join(output_dir, "evaluation", "class_metrics.csv")
    df.to_csv(csv_path, index=False)

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
    start_time = time.time()
    
    try:
        print("=== EVALUATING MOBILENETV2 (EFFICIENTNET-STYLE) 96x96 MODEL ===")
        
        # Set up paths
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, PROCESSED_DATA_DIR)
        MODEL_DIR_FULL = os.path.join(PROJECT_ROOT, MODEL_DIR)
        
        print(f"Looking for processed data in: {PROCESSED_DIR}")
        print(f"Looking for model in: {MODEL_DIR_FULL}")
        
        # Check if directories exist
        if not os.path.exists(PROCESSED_DIR):
            print(f"\nError: Preprocessed data directory not found: {PROCESSED_DIR}")
            print("\nPlease run preprocess_mobilenetv2_96x96.py first to prepare the dataset.")
            exit(1)
        
        if not os.path.exists(MODEL_DIR_FULL):
            print(f"\nError: Model directory not found: {MODEL_DIR_FULL}")
            print("\nPlease run train_mobilenetv2_efficientnet_style_96x96.py first to train the model.")
            exit(1)
        
        # Load processed data
        X, y, classes = load_preprocessed_data(PROCESSED_DIR)
        print(f"\nLoaded {len(X)} images from {len(classes)} classes")
        
        # Convert labels to categorical for ROC curve calculation
        y_cat = tf.keras.utils.to_categorical(y, len(classes))
        
        # Load model and class information
        model, class_info = load_model_and_info(MODEL_DIR_FULL)
        print("\nModel loaded successfully")
        
        # Verify that class names match
        if set(classes) != set(class_info.get('classes', [])):
            print("Warning: Class names in processed data don't match model's class info")
            print(f"Data classes: {classes}")
            print(f"Model classes: {class_info.get('classes', [])}")
            print("Using classes from the processed data for evaluation")
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = evaluate_model(model, X, y, classes)
        
        # Print initial results
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Overall Precision: {metrics['precision']:.4f}")
        print(f"Overall Recall: {metrics['recall']:.4f}")
        print(f"Overall F1 Score: {metrics['f1']:.4f}")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        plot_confusion_matrix(metrics['confusion_matrix'], classes, MODEL_DIR_FULL)
        plot_class_metrics(metrics['class_report'], MODEL_DIR_FULL)
        plot_roc_curve(y_cat, metrics['y_pred_prob'], classes, MODEL_DIR_FULL)
        plot_precision_recall_curve(y_cat, metrics['y_pred_prob'], classes, MODEL_DIR_FULL)
        
        # Save evaluation summary
        save_evaluation_summary(metrics, classes, class_info, MODEL_DIR_FULL)
        
        # Calculate evaluation time
        end_time = time.time()
        eval_time = end_time - start_time
        minutes, seconds = divmod(eval_time, 60)
        
        print(f"\nEvaluation completed in {int(minutes)}m {int(seconds)}s")
        print("=== EVALUATION COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print(f"1. You have run preprocess_mobilenetv2_96x96.py first")
        print(f"2. You have run train_mobilenetv2_efficientnet_style_96x96.py first")
        print(f"3. The processed dataset folder exists: {PROCESSED_DATA_DIR}")
        print(f"4. The model directory exists: {MODEL_DIR}")
        raise