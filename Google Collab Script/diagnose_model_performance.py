import tensorflow as tf
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def analyze_model_performance(model_dir="models/mobilenetv2_efficientnet_style_96x96"):
    """Analyze model performance and identify potential issues"""
    
    print("=== MODEL PERFORMANCE ANALYSIS ===")
    
    # Load evaluation results
    eval_path = os.path.join(model_dir, "evaluation", "evaluation_metrics.json")
    
    if not os.path.exists(eval_path):
        print(f"Error: Evaluation metrics not found at {eval_path}")
        print("Please run the evaluation script first:")
        print("python evaluate_mobilenetv2_efficientnet_style_96x96.py")
        return
    
    with open(eval_path, 'r') as f:
        metrics = json.load(f)
    
    # Load class info
    class_info_path = os.path.join(model_dir, "class_info.json")
    with open(class_info_path, 'r') as f:
        class_info = json.load(f)
    class_names = class_info["classes"]
    
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Overall Precision: {metrics['precision']:.4f}")
    print(f"Overall Recall: {metrics['recall']:.4f}")
    print(f"Overall F1-Score: {metrics['f1']:.4f}")
    
    # Analyze per-class performance
    print(f"\n=== PER-CLASS ANALYSIS ===")
    class_report = metrics['class_report']
    
    problem_classes = []
    for class_name in class_names:
        if class_name in class_report:
            precision = class_report[class_name]['precision']
            recall = class_report[class_name]['recall']
            f1 = class_report[class_name]['f1-score']
            support = class_report[class_name]['support']
            
            print(f"\n{class_name}:")
            print(f"  Precision: {precision:.3f} (how often predictions are correct)")
            print(f"  Recall: {recall:.3f} (how often actual cases are found)")
            print(f"  F1-Score: {f1:.3f} (balanced measure)")
            print(f"  Test samples: {support}")
            
            # Flag problematic classes
            if precision < 0.8 or recall < 0.8:
                problem_classes.append(class_name)
                print(f"  ⚠️  LOW PERFORMANCE detected!")
    
    # Analyze confusion matrix
    print(f"\n=== CONFUSION MATRIX ANALYSIS ===")
    cm = np.array(metrics['confusion_matrix'])
    
    # Find most confused class pairs
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                confusion_rate = cm[i][j] / cm[i].sum()
                if confusion_rate > 0.1:  # More than 10% confusion
                    confusion_pairs.append((class_names[i], class_names[j], confusion_rate))
    
    if confusion_pairs:
        print("Most confused class pairs:")
        for true_class, pred_class, rate in sorted(confusion_pairs, key=lambda x: x[2], reverse=True):
            print(f"  {true_class} → {pred_class}: {rate:.1%} confusion rate")
    else:
        print("No significant class confusion detected in test set")
    
    # Check for overfitting indicators
    print(f"\n=== OVERFITTING CHECK ===")
    
    # Load training history if available
    history_path = os.path.join(model_dir, "plots", "training_history.png")
    if os.path.exists(history_path):
        print(f"Training history plot available at: {history_path}")
        print("Check if:")
        print("- Training accuracy >> Validation accuracy (overfitting)")
        print("- Validation loss increases while training loss decreases")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    
    if metrics['accuracy'] > 0.95:
        print("🚨 Very high test accuracy detected!")
        print("This might indicate:")
        print("- Data leakage (same images in train/test)")
        print("- Test set too similar to training set")
        print("- Need to test on completely new, real-world images")
    
    if problem_classes:
        print(f"\n📊 Problem classes detected: {problem_classes}")
        print("Possible solutions:")
        print("- Collect more training data for these classes")
        print("- Use more aggressive data augmentation")
        print("- Check if images are properly labeled")
        print("- Consider class balancing techniques")
    
    if confusion_pairs:
        print(f"\n🔄 Class confusion detected")
        print("Possible solutions:")
        print("- Use more discriminative features")
        print("- Collect more diverse training examples")
        print("- Use techniques like focal loss for hard examples")
    
    # Real-world testing recommendations
    print(f"\n🌍 REAL-WORLD TESTING TIPS:")
    print("1. Test with images taken in different lighting conditions")
    print("2. Test with images from different cameras/phones")
    print("3. Test with images at different angles and distances")
    print("4. Test with partially visible leaves or multiple leaves")
    print("5. Test with images that have different backgrounds")
    
    return metrics

if __name__ == "__main__":
    analyze_model_performance()