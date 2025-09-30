import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

MODEL_DIR = "models/efficientnet_lite_96x96_full_epochs"
BATCH_SIZE = 32
TEST_DATASET_DIR = "test_dataset"

# Load class names from model_metadata.json
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r") as f:
    class_info = json.load(f)
class_names = class_info["classes"]

# Load model
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.h5"))

# Load test data from test_dataset
X_test = []
y_test = []
from PIL import Image
for idx, class_name in enumerate(class_names):
    class_dir = os.path.join(TEST_DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for file in tqdm(files, desc=f"Loading {class_name}"):
        img_path = os.path.join(class_dir, file)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((96, 96))
        img = np.array(img) / 255.0
        X_test.append(img)
        y_test.append(idx)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Predict
y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - EfficientNet Lite 96x96")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_efficientnet_lite_96x96.png"))
plt.show()

# Per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(10, 5))
plt.bar(class_names, class_accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Class")
plt.ylim(0, 1)
plt.title("Class-wise Accuracy - EfficientNet Lite 96x96")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "class_accuracy_efficientnet_lite_96x96.png"))
plt.show()

# Overall accuracy
overall_acc = np.mean(y_pred == y_test)
print(f"\nOverall Test Accuracy: {overall_acc:.4f}")
