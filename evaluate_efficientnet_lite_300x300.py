import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# Constants
MODEL_DIR = "models/efficientnet_lite_300x300"
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
    filenames = []
    
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
            filenames.append(os.path.join(class_name, img_file))
    
    if not images:
        raise ValueError("No images found in any class folder!")
    
    return np.array(images), np.array(labels), class_names, filenames

def load_model(model_path):
    """Load the trained model"""
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate the model and print metrics"""
    print("\nEvaluating model...")
    # Convert to categorical
    y_test_cat = tf.keras.utils.to_categorical(y_test, len(class_names))
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Get predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report)
    
    return y_pred, cm, report

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

def plot_sample_predictions(X_test, y_test, y_pred, class_names, filenames, save_dir):
    """Plot some sample predictions (correct and incorrect)"""
    # Find correct and incorrect predictions
    correct_indices = np.where(y_test == y_pred)[0]
    incorrect_indices = np.where(y_test != y_pred)[0]
    
    # Create a directory for saving sample predictions
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot some correct predictions
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(correct_indices[:5]):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_test[idx])
        plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}")
        plt.axis('off')
    
    # Plot some incorrect predictions
    if len(incorrect_indices) > 0:
        for i, idx in enumerate(incorrect_indices[:5]):
            plt.subplot(2, 5, i+6)
            plt.imshow(X_test[idx])
            plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sample_predictions.png"))
    print(f"Sample predictions saved to {os.path.join(save_dir, 'sample_predictions.png')}")

def save_evaluation_results(y_test, y_pred, class_names, filenames, save_path):
    """Save detailed evaluation results to CSV"""
    results = []
    for i in range(len(y_test)):
        results.append({
            'filename': filenames[i],
            'true_class': class_names[y_test[i]],
            'predicted_class': class_names[y_pred[i]],
            'correct': y_test[i] == y_pred[i]
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Detailed results saved to {save_path}")

def main():
    print("=== EVALUATING EFFICIENTNET LITE 300x300 MODEL ===")
    
    # Set up paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(PROJECT_ROOT, TEST_DATA_DIR)
    MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_DIR, "best_model_phase1.h5")
    EVAL_DIR = os.path.join(PROJECT_ROOT, MODEL_DIR, "evaluation")
    
    # Create evaluation directory
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from: {TEST_DIR}")
    X_test, y_test, class_names, filenames = load_preprocessed_data(TEST_DIR)
    print(f"Loaded {len(X_test)} test images from {len(class_names)} classes")
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Evaluate model
    y_pred, cm, report = evaluate_model(model, X_test, y_test, class_names)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, class_names, os.path.join(EVAL_DIR, "confusion_matrix.png"))
    
    # Plot sample predictions
    plot_sample_predictions(X_test, y_test, y_pred, class_names, filenames, EVAL_DIR)
    
    # Save detailed results
    save_evaluation_results(y_test, y_pred, class_names, filenames, os.path.join(EVAL_DIR, "detailed_results.csv"))
    
    # Save classification report
    with open(os.path.join(EVAL_DIR, "classification_report.txt"), 'w') as f:
        f.write(report)
    
    print("\n=== EVALUATION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
