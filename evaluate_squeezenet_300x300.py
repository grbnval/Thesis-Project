import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

# Constants
MODEL_DIR = "models/squeezenet_300x300"
TEST_DATA_DIR = "processed_test_300x300"
BATCH_SIZE = 32

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
        for img_file in files:
            img_path = os.path.join(class_dir, img_file)
            img = np.load(img_path)
            images.append(img)
            labels.append(class_idx)
    
    if not images:
        raise ValueError("No images found in any class folder!")
    
    return np.array(images), np.array(labels), class_names

def load_model_and_classes():
    """Load the trained model and class names"""
    # Load model
    model_path = os.path.join(MODEL_DIR, "model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "best_model_phase2.h5")
    
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Load class names
    class_names_path = os.path.join(MODEL_DIR, "class_names.txt")
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Try to load from metadata
        metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                class_names = metadata.get("classes", [])
        else:
            raise FileNotFoundError("No class names or metadata file found")
    
    return model, class_names

def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - SqueezeNet 300x300')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "evaluation", "confusion_matrix.png"))
    plt.close()

def plot_class_accuracy(class_accuracy, class_names, output_dir):
    """Plot and save per-class accuracy"""
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_names), y=list(class_accuracy.values()))
    plt.title('Class-wise Accuracy - SqueezeNet 300x300')
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "evaluation", "class_accuracy.png"))
    plt.close()

def save_classification_report(report, output_dir):
    """Save classification report to CSV"""
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, "evaluation", "classification_report.csv")
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")

def save_evaluation_metrics(metrics, output_dir):
    """Save evaluation metrics to JSON"""
    metrics_path = os.path.join(output_dir, "evaluation", "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to {metrics_path}")

def evaluate_model():
    """Evaluate the model on test data"""
    print("=== EVALUATING SQUEEZENET 300x300 MODEL ===")
    
    # Load model and class names
    model, class_names = load_model_and_classes()
    
    # Load test data
    test_data_path = os.path.join(TEST_DATA_DIR)
    X_test, y_test, test_classes = load_preprocessed_data(test_data_path)
    
    # Check that test class names match model class names
    if len(test_classes) != len(class_names):
        print("Warning: Number of classes in test data doesn't match model")
        print(f"Test classes: {test_classes}")
        print(f"Model classes: {class_names}")
    
    # Convert labels to categorical
    num_classes = len(class_names)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, batch_size=BATCH_SIZE)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Get predictions
    y_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, MODEL_DIR)
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for i in range(len(class_names)):
        class_indices = np.where(y_test == i)[0]
        if len(class_indices) > 0:
            class_acc = accuracy_score(y_test[class_indices], y_pred[class_indices])
            class_accuracy[class_names[i]] = class_acc
    
    # Plot class accuracy
    plot_class_accuracy(class_accuracy, class_names, MODEL_DIR)
    
    # Generate classification report
    cr = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    save_classification_report(cr, MODEL_DIR)
    
    # Save evaluation metrics
    metrics = {
        "accuracy": float(test_accuracy),
        "loss": float(test_loss),
        "per_class_accuracy": class_accuracy,
        "f1_scores": {class_name: cr[class_name]['f1-score'] for class_name in class_names},
        "precision": {class_name: cr[class_name]['precision'] for class_name in class_names},
        "recall": {class_name: cr[class_name]['recall'] for class_name in class_names},
    }
    save_evaluation_metrics(metrics, MODEL_DIR)
    
    print("\n=== EVALUATION COMPLETED ===")
    
    return metrics

if __name__ == "__main__":
    metrics = evaluate_model()
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nPer-class F1 Scores:")
    for class_name, f1 in metrics['f1_scores'].items():
        print(f"- {class_name}: {f1:.4f}")
