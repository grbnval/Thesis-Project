# Training and Preprocessing Guide

This document summarizes how the dataset is preprocessed and how the models in this repository were trained, exported, and evaluated. It includes the exact preprocessing steps (so you can reproduce the `.npy` processed dataset), the training flow, model saving conventions, and TFLite conversion notes.

## Project layout (relevant files)

- `raw_dataset/` — original input images organized by class subfolders:
  - `healthy_leaf/`, `early_blight_leaf/`, `late_blight_leaf/`, `septoria_leaf/`, `unknown/`
- `preprocess_dataset_96x96.py` — preprocessing script that converts images to 96x96 `.npy` arrays using CLAHE and normalization.
- `processed_dataset_96x96/` — output folder produced by the preprocessing script (contains `.npy` files per class).
- `processed_train_96x96/`, `processed_test_96x96/` — train/test split directories created from the processed dataset.
- `train_*.py` — training scripts (MobileNetV2, EfficientNet-style, SqueezeNet variants).
- `models/` — trained model folders; each contains `evaluation/`, model files (`best_model.h5`, `model.keras`, `model.tflite`, `model_quantized.tflite`) and `class_info.json`.

---

## Preprocessing pipeline (exact steps)

Preprocessing is implemented in `preprocess_dataset_96x96.py`. The canonical steps used during training and evaluation are:

1. Read image with OpenCV (`cv2.imread`). If the image cannot be read it is skipped.
2. Convert BGR → RGB (`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`).
3. Resize to 96×96 (`cv2.resize(img, (96,96))`).
4. Apply CLAHE on the L channel in LAB color space:
   - Convert to LAB: `lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)`
   - Split channels: `l, a, b = cv2.split(lab)`
   - Create CLAHE: `clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))`
   - Apply to L channel: `cl = clahe.apply(l)`
   - Merge and convert back: `enhanced = cv2.merge((cl, a, b))` then `img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)`
5. Convert to `float32` and normalize to [0, 1]: `img = img.astype('float32') / 255.0`.
6. Save as NumPy array: `np.save(output_path, img)` where `output_path` ends with `.npy`.

Notes:
- CLAHE parameters (clipLimit=3.0, tileGridSize=(8,8)) were chosen experimentally to boost contrast while avoiding noise amplification.
- Using preprocessed `.npy` files during evaluation ensures deterministic input and much faster runs (no on-the-fly preprocessing overhead).

---

## Train / evaluation workflow

Core steps used to train the reported models:

1. Prepare dataset
   - Run the preprocessing script to create `processed_dataset_96x96/` and then create train/test splits into `processed_train_96x96/` and `processed_test_96x96/` (default test split 20%).

2. Configure model and training
   - Choose a training script (e.g. `train_mobilenetv2_96x96.py` or `train_mobilenetv2_efficientnet_style_96x96.py` or `train_squeezenet_96x96_full_epochs.py`).
   - Each script loads `class_info.json` (or builds class indices from the training folders), defines the model backbone (MobileNetV2-based, EfficientNet-style head, Squeeze-like head), prepares training callbacks, and compiles the model with appropriate optimizer and loss (usually `Adam` and `sparse_categorical_crossentropy` / `categorical_crossentropy` depending on label format).

3. Training
   - Models were trained for the configured number of epochs (see training script headers for default `epochs` values). Typical training used `ImageDataGenerator`-style loading from `.npy` batches or custom dataset pipelines reading `.npy` files and applying minimal runtime augmentations (random flips, small rotations) if enabled in the training script.
   - Callbacks used: ModelCheckpoint (save best weights by validation accuracy), ReduceLROnPlateau, EarlyStopping (optional), CSVLogger.

4. Model saving and metadata
   - Best-performing checkpoint saved as `best_model.h5` in the model folder.
   - Final model may be saved as `final_model.keras` or `model.keras` depending on the script.
   - `class_info.json` stores class names and any mapping used (keeps evaluation consistent).

5. Evaluation
   - Evaluation scripts (e.g. `evaluate_mobilenetv2_96x96_full_epochs.py`, `evaluate_squeezenet_96x96.py`, `evaluate_mobilenetv2_efficientnet_style_96x96.py`) load `best_model.h5` (or `model.keras`) and the preprocessed test set in `processed_test_96x96/`.
   - They compute predictions, confusion matrices, classification reports (precision, recall, f1, support), per-class false negative/positive analysis, save figures and text reports under `models/<model_name>/evaluation/`.

---

## TFLite conversion and quantization

After training and evaluation with Keras, models are converted to TFLite for on-device testing:

1. Standard TFLite (float32)
   - Use `tf.lite.TFLiteConverter.from_keras_model(model)` and convert to `model.tflite`.
   - Evaluate the TFLite model with the same preprocessed `.npy` test data; record accuracy and inference time (measured per image across test set).

2. Post-training quantized TFLite (int8/uint8)
   - Use `converter.optimizations = [tf.lite.Optimize.DEFAULT]` and provide a representative dataset generator to calibrate activations for quantization.
   - Save as `model_quantized.tflite`.
   - Quantization can reduce model size and inference latency but may reduce accuracy — in these evaluations, quantized accuracy dropped by ~5–9 percentage points depending on the model.

Tips:
- If quantized accuracy loss is large, try Quantization-Aware Training (QAT) using TensorFlow Model Optimization (TF-MOT). QAT often preserves accuracy after quantization but requires additional training.

---

## Reproducible commands

Preprocess dataset (interactive):

```powershell
python preprocess_dataset_96x96.py
```

Generate train/test split (non-interactive example):

```python
# inside a Python session
from preprocess_dataset_96x96 import create_train_test_split
create_train_test_split('processed_dataset_96x96', 'processed_train_96x96', 'processed_test_96x96', test_split=0.2)
```

Train (example):

```powershell
python train_mobilenetv2_96x96.py --data_dir processed_train_96x96 --epochs 100 --batch_size 32
```

Convert to TFLite (example snippet inside a script):

```python
import tensorflow as tf
# load model
model = tf.keras.models.load_model('models/my_model/best_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('models/my_model/model.tflite', 'wb').write(tflite_model)
```

Quantized conversion (post-training quantization with representative dataset):

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_gen():
    for file in representative_files:
        arr = np.load(file)
        arr = arr.reshape(1,96,96,3).astype(np.float32)
        yield [arr]
converter.representative_dataset = representative_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant = converter.convert()
open('models/my_model/model_quantized.tflite', 'wb').write(tflite_quant)
```

---

## Notes and caveats

- The preprocessing steps (CLAHE, resizing, normalization) must match exactly between training and evaluation to get valid comparisons.
- The repository contains multiple training variants; check the corresponding training script headers for hyperparameters and augmentation settings used for each model.
- For production deployment, run additional tests on device hardware to measure real-world latency, memory usage, and accuracy on representative images.

If you want, I can also:
- Add a short `README.md` in `processed_dataset_96x96/` documenting the CLAHE parameters and data layout.
- Add a small script to generate a representative dataset list for quantization calibration.
- Scaffold a QAT workflow for the model you choose.
