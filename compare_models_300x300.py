import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import seaborn as sns

# Constants
MODELS_TO_COMPARE = [
    {
        "name": "MobileNetV2 300x300",
        "model_path": "models/mobilenetv2_300x300_separate/best_model.h5",
        "color": "blue"
    },
    {
        "name": "EfficientNet Lite 300x300",
        "model_path": "models/efficientnet_lite_300x300/best_model_phase1.h5",
        "color": "green"
    }
]
TEST_DATA_DIR = "processed_test_300x300"
OUTPUT_DIR = "model_comparison"

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

def load_model(model_path):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    print(f"Loading model from {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate the model and return metrics"""
    # Convert to categorical
    y_test_cat = tf.keras.utils.to_categorical(y_test, len(class_names))
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=1)
    
    # Get predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4, output_dict=True)
    
    # Class-wise accuracy
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_indices = np.where(y_test == i)[0]
        class_predictions = y_pred[class_indices]
        class_accuracy[class_name] = accuracy_score(y_test[class_indices], class_predictions)
    
    return {
        "accuracy": test_accuracy,
        "loss": test_loss,
        "confusion_matrix": cm,
        "report": report,
        "class_accuracy": class_accuracy,
        "predictions": y_pred
    }

def compare_models(models_data, X_test, y_test, class_names):
    """Compare multiple models and return their evaluation metrics"""
    results = {}
    
    for model_info in models_data:
        model_name = model_info["name"]
        model_path = model_info["model_path"]
        
        model = load_model(model_path)
        if model is None:
            print(f"Skipping {model_name} due to loading error")
            continue
        
        print(f"\nEvaluating {model_name}...")
        results[model_name] = evaluate_model(model, X_test, y_test, class_names)
        print(f"Accuracy: {results[model_name]['accuracy']:.4f}")
        print(f"Loss: {results[model_name]['loss']:.4f}")
    
    return results

def plot_comparison_charts(results, class_names, output_dir):
    """Plot comparison charts for the models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall accuracy comparison
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = [results[model]["accuracy"] for model in models]
    
    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0.5, 1.0)  # Set a reasonable y-axis limit
    
    # Add values on top of bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    
    # 2. Class-wise accuracy comparison
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    class_data = {model_name: results[model_name]["class_accuracy"] for model_name in models}
    df = pd.DataFrame(class_data)
    
    # Plot
    df.plot(kind='bar', figsize=(12, 8))
    plt.title('Class-wise Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.ylim(0.5, 1.0)
    plt.legend(title='Models')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_accuracy_comparison.png"))
    
    # 3. Confusion matrices
    for model_name in models:
        plt.figure(figsize=(10, 8))
        cm = results[model_name]["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png"))
    
    # 4. Generate a summary table
    summary = {
        "Model": [],
        "Accuracy": [],
        "Loss": []
    }
    
    # Add class-wise F1 scores
    for class_name in class_names:
        summary[f"F1-{class_name}"] = []
    
    for model_name in models:
        summary["Model"].append(model_name)
        summary["Accuracy"].append(f"{results[model_name]['accuracy']:.4f}")
        summary["Loss"].append(f"{results[model_name]['loss']:.4f}")
        
        for class_name in class_names:
            class_idx = class_names.index(class_name)
            f1 = results[model_name]["report"][class_name]["f1-score"]
            summary[f"F1-{class_name}"].append(f"{f1:.4f}")
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "model_comparison_summary.csv"), index=False)
    
    # Print summary to console
    print("\nModel Comparison Summary:")
    print(summary_df.to_string(index=False))

def main():
    print("=== COMPARING 300x300 MODELS ===")
    
    # Set up paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(PROJECT_ROOT, TEST_DATA_DIR)
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, OUTPUT_DIR)
    
    # Update model paths to full paths
    for model_info in MODELS_TO_COMPARE:
        model_info["model_path"] = os.path.join(PROJECT_ROOT, model_info["model_path"])
    
    # Load test data
    print(f"Loading test data from: {TEST_DIR}")
    X_test, y_test, class_names = load_preprocessed_data(TEST_DIR)
    print(f"Loaded {len(X_test)} test images from {len(class_names)} classes")
    
    # Compare models
    results = compare_models(MODELS_TO_COMPARE, X_test, y_test, class_names)
    
    # Plot comparison charts
    plot_comparison_charts(results, class_names, OUTPUT_PATH)
    
    print(f"\nComparison charts and data saved to: {OUTPUT_PATH}")
    print("\n=== COMPARISON COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
