# Models: Training, Preprocessing and Comparison

This document summarizes the data processing and normalization used across the project, the architecture / layer decisions for each trained model, how models were converted to TFLite, the training pipeline used, and a concise comparison of the three evaluated models.

Files and scripts referenced in this doc:
- `preprocess_dataset_96x96.py` — preprocessing pipeline (CLAHE, resize, normalize, save `.npy`).
- `processed_test_96x96/`, `processed_train_96x96/` — preprocessed `.npy` dataset used for training and evaluation.
- `train_*.py` — various training scripts (MobileNetV2, EfficientNet-style, SqueezeNet variants).
- `models/<model_name>/evaluation/` — evaluation outputs (classification reports, FP/FN analyses, plots).
- `scripts/compare_class_metrics.py`, `scripts/plot_fp_fn.py`, `scripts/plot_three_models.py` — scripts used to aggregate and plot metrics.

---

## 1) Data processing and normalization

All reported experiments use the same deterministic preprocessing pipeline implemented by `preprocess_dataset_96x96.py` so training and evaluation inputs match exactly.

Key steps (exact):

- Read image with OpenCV: `cv2.imread(path)`; skip if unreadable.
- Convert from BGR → RGB: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
- Resize to 96×96: `cv2.resize(img, (96, 96))`.
- CLAHE contrast enhancement applied on L channel in LAB space:
  - `lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)`
  - `l, a, b = cv2.split(lab)`
  - `clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))`
  - `cl = clahe.apply(l)`
  - `enhanced = cv2.merge((cl, a, b))`
  - `img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)`
- Normalize to [0,1]: `img = img.astype('float32') / 255.0`.
- Save processed sample as `.npy` using `np.save(output_path, img)`.

Notes:
- CLAHE parameters are fixed across preprocessing and match the training pipeline (clipLimit=3.0, tileGridSize=(8,8)).
- Using `.npy` files removes runtime preprocessing variability and speeds up evaluation.

---

## 2) Models, architectures and layer-level notes

This project evaluates three model variants (per-folder names):

- `squeezenet_96x96_full_epochs_with_unknown` — a SqueezeNet-inspired architecture trained to convergence (full epochs)
- `mobilenetv2_96x96_full_epochs_with_unknown` — MobileNetV2 backbone adapted to 96×96 inputs
- `mobilenetv2_efficientnet_style_96x96` — MobileNetV2 with an EfficientNet-style head/regularisation design

Below are the layer-level design notes and differences used in the codebase. Exact architectures live in the training scripts; the notes below summarize the important differences and tunable parameters.

Common inputs and final layer
- Input shape: (96, 96, 3) for all models (matches preprocessed `.npy` files).
- Final dense/classifier layer: `Dense(num_classes, activation='softmax')` with `num_classes` taken from `class_info.json` (includes `unknown` class).

MobileNetV2 (baseline)
- Backbone: standard MobileNetV2 (Keras), adjusted input shape.
- Typical head after backbone: GlobalAveragePooling2D → Dropout (tunable, e.g. 0.2–0.4) → Dense(num_classes).
- Optimizer: Adam (default used in training scripts); loss: categorical or sparse categorical cross-entropy depending on label format.
- Regularization: dropout on head; potential L2 weight decay in some variants.

MobileNetV2 (EfficientNet-style)
- Backbone: MobileNetV2 or MobileNetV2-derived blocks.
- Head inspired by EfficientNet design choices: e.g. squeeze-and-excitation style bottlenecks or additional BatchNormalization / dropout ordering, small modifications to number of filters in top layers, possibly stronger regularization and a slightly deeper head.
- Training hyperparameters: similar optimizer/loss, but typically trained for a full 100-epoch schedule (script metadata indicates `full_100_epochs` for one run). See the script header for precise learning-rate schedules.

SqueezeNet-inspired model
- A light-weight SqueezeNet-like model tuned for 96×96 input (fire modules / squeeze-expand patterns).
- Typically fewer parameters than MobileNetV2 but the reported run achieved the highest Keras accuracy on this dataset.

Assumptions / where to find exact layers
- The exact number of filters, dropout rates, and block-level changes are defined in each `train_*.py` file. If you want, I can extract exact layer sequences (Keras model.summary()) from each saved model and include them here.

### Framework and common choices
- Framework: TensorFlow 2.x with Keras API is used across all training scripts (imports use `tensorflow` and `tf.keras`).
- Pretrained weights: base models are initialized with `weights='imagenet'` (ImageNet pretraining) where applicable (MobileNetV2 base used for MobileNetV2 variants and the SqueezeNet-inspired script uses MobileNetV2 as its base as well). This is transfer learning from ImageNet.

---

## 3) Training pipeline (how models were trained)

General flow (shared across training scripts):

1. Data source: `processed_train_96x96/` (batches reading `.npy` files) and `processed_test_96x96/` for validation/evaluation.
2. Data augmentation: optional small augmentations (random horizontal flips, small rotations) applied at train-time by the training script (check script header for `augmentation=True/False`).
3. Optimizer and loss: typically `Adam` with default parameters; loss `sparse_categorical_crossentropy` or `categorical_crossentropy` depending on labels.
4. Callbacks: `ModelCheckpoint` (save best by val_acc), `ReduceLROnPlateau`, optional `EarlyStopping`, and `CSVLogger` for logging.
5. Epochs: full training runs used a long schedule (examples reference `full_100_epochs`). The exact `epochs`, `batch_size`, and learning-rate schedule are specified in the training script header or command-line args.

Training outputs
- Best checkpoint: `best_model.h5` (saved in `models/<model_name>/`).
- Final model (optional): `final_model.keras` or `model.keras`.
- Metadata: `class_info.json` (class order and metadata) and `evaluation/` folder containing evaluation artifacts.

---

### Exact training hyperparameters used (common across scripts)
- Optimizer: Adam (tf.keras.optimizers.Adam)
- Initial learning rate: 0.001
- LR schedule used for fine-tuning: two-phase training
  1. Phase 1: train newly-added top layers for 20 epochs at LR=0.001
  2. Phase 2: unfreeze the last ~30 layers of the base model and continue training for the remainder of the schedule at LR=0.0001 (i.e., LR/10)
- Batch size: 32
- Epochs: total 100 (phase1 20 + phase2 80 in these scripts)
- Loss: `categorical_crossentropy` (labels are converted to one-hot via `to_categorical`)
- Data augmentation: `ImageDataGenerator` with rotation_range=20, width/height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'. Validation and test sets use no augmentation.

Notes on transfer learning / fine-tuning:
- All training scripts freeze the base (pretrained) backbone initially and train a custom classification head. After an initial stage of head-only training they unfreeze the last ~30 layers for fine-tuning with a reduced LR. This standard transfer-learning strategy gives faster convergence and better final accuracy compared to training from scratch.
- In particular, the SqueezeNet-inspired script uses a MobileNetV2 base with `weights='imagenet'` and follows the same two-phase strategy. Because it leverages ImageNet pretraining and fine-tuning, it converges faster and reaches higher accuracy compared to training an equivalent architecture from random initialization — this is emphasized in the results where the SqueezeNet-inspired run achieved the highest Keras test accuracy.


## 4) Conversion to TFLite and quantization

Conversion steps used in this work:

- Standard TFLite (float32): `tf.lite.TFLiteConverter.from_keras_model(model)` → `model.tflite`.
- Post-training quantization: `converter.optimizations = [tf.lite.Optimize.DEFAULT]` with a `representative_dataset` generator that yields batches of preprocessed `.npy` arrays. Saved as `model_quantized.tflite`.

Quantization details and observed effects:
- Quantization reduces model size and inference latency but often reduces accuracy. In these evaluations the quantized models showed a measurable accuracy drop:
  - SqueezeNet: 0.9793 → 0.8970 (approx. −8.8 percentage points)
  - MobileNetV2 baseline: 0.9583 → 0.9035 (approx. −5.5 pp)
  - MobileNetV2 Eff-style: 0.9692 → 0.9041 (approx. −6.5 pp)
- When quantization causes a large drop, recommended mitigations:
  - Use Quantization-Aware Training (QAT) via TF Model Optimization and fine-tune the model with simulated quantization.
  - Add a larger, representative calibration/representative dataset for post-training quantization.

---

## 5) Results summary and per-model comparison

High-level numbers (from model `evaluation/` reports):

- SqueezeNet (folder `squeezenet_96x96_full_epochs_with_unknown`)
  - Keras (best checkpoint) accuracy: **0.9793**
  - Standard TFLite accuracy: **0.9790** (avg inference ~26.25 ms, size ~2.84 MB)
  - Quantized TFLite accuracy: **0.8970** (avg inference ~7.47 ms, size ~3.04 MB)

- MobileNetV2 baseline (`mobilenetv2_96x96_full_epochs_with_unknown`)
  - Keras accuracy: **0.9583**
  - Standard TFLite: **0.9570** (avg inference ~11.25 ms, size ~3.15 MB)
  - Quantized TFLite: **0.9035** (avg inference ~1.64 ms, size ~3.36 MB)

- MobileNetV2 EfficientNet-style (`mobilenetv2_efficientnet_style_96x96`)
  - Keras accuracy: **0.9692**
  - Standard TFLite: **0.9651** (avg inference ~10.07 ms, size ~3.15 MB)
  - Quantized TFLite: **0.9041** (avg inference ~2.01 ms, size ~3.36 MB)

Per-class behaviour (selected highlights from FP/FN reports):

- early_blight_leaf (most problematic class in many runs)
  - False negatives (Keras): SqueezeNet 38/596 (6.4%), MobileNetV2 baseline 83/596 (13.9%), Eff-style 74/596 (12.4%).
  - Quantized models increase FN dramatically for some models — e.g., SqueezeNet quantized had early_blight_leaf FN = 149/596 (25%).

- septoria_leaf
  - Highest false positive rates across models. Baseline MobileNetV2 showed a septoria FP rate ~12.2%, Eff-style ~9.4%, SqueezeNet ~6.7% (Keras evaluations).

- unknown & healthy
  - `unknown` predictions are consistently precise (rare FP), and `healthy` typically has very low FN in all models.

Trade-offs and interpretable summary

- Best raw accuracy: **SqueezeNet** (Keras) — highest overall test accuracy and lower FN for early_blight/ septoria in the Keras runs.
- Best inference latency (TFLite standard): **MobileNetV2 EfficientNet-style** (~10 ms) and **MobileNetV2 baseline** (~11 ms) — much faster than SqueezeNet (~26 ms) in the tested environment.
- Best quantized retention: both MobileNetV2 variants preserved more accuracy after quantization (~0.903–0.904) than SqueezeNet (~0.897), suggesting MobileNetV2 variants are more robust to post-training quantization here.

Practical recommendation

- If top test-set accuracy is the single priority and you can accept slower on-device inference: use **SqueezeNet**.
- If on-device latency and smaller inference times are priorities while retaining good accuracy after quantization: use **MobileNetV2 EfficientNet-style** (best tradeoff in our runs).
- For quantized deployment: consider QAT for the chosen model. MobileNetV2 variants appear to require less remediation after quantization than SqueezeNet in these runs.

---

## 5.a) Model efficiency (size, latency, FPS)

The evaluation scripts measured average inference times (ms per image) and model file sizes for both standard TFLite and quantized TFLite. Below are the measured values (reported by the evaluation scripts and saved in each model's `evaluation/` folder). Note: these are approximate and depend on the host hardware where the evaluation was performed — treat them as relative comparisons rather than absolute benchmarks.

- SqueezeNet (folder `squeezenet_96x96_full_epochs_with_unknown`)
  - Standard TFLite: size = 2.84 MB, avg inference time = 26.25 ms → FPS ≈ 38.1
  - Quantized TFLite: size = 3.04 MB, avg inference time = 7.47 ms → FPS ≈ 133.9

- MobileNetV2 baseline (`mobilenetv2_96x96_full_epochs_with_unknown`)
  - Standard TFLite: size = 3.15 MB, avg inference time = 11.25 ms → FPS ≈ 88.9
  - Quantized TFLite: size = 3.36 MB, avg inference time = 1.64 ms → FPS ≈ 609.8

- MobileNetV2 EfficientNet-style (`mobilenetv2_efficientnet_style_96x96`)
  - Standard TFLite: size = 3.15 MB, avg inference time = 10.07 ms → FPS ≈ 99.3
  - Quantized TFLite: size = 3.36 MB, avg inference time = 2.01 ms → FPS ≈ 497.5

Observations:
- The MobileNetV2 variants provide a substantially faster standard TFLite inference time than the SqueezeNet-inspired run on the measured platform.
- Quantization dramatically improves per-image latency (ms) and therefore FPS for all models; MobileNetV2 baseline shows the highest FPS after quantization in these measurements.
- Model file sizes reported here are small (2.8–3.4 MB) — differences may come from conversion settings and whether metadata is stored in the TFLite file. The quantized TFLite files can sometimes be larger on disk depending on how arrays are stored; focus on measured latency and accuracy trade-offs for deployment.

Caveats:
- FPS/ms numbers should be re-measured on target hardware (ESP32, mobile CPU, or edge accelerator) before making deployment choices. The evaluation results here were gathered by the project's evaluation scripts on the development machine and serve as relative guidance.


---

## 6) Reproducible artifacts and scripts

You can regenerate the comparison artifacts used in this document with the scripts in `scripts/`:

- `scripts/aggregate_evaluation_metrics.py` → writes `models/evaluation_metrics.csv` (summary metrics per model).
- `scripts/plot_three_models.py` → `models/comparison_three_models.csv`, `models/comparison_plot.png` (accuracy + inference time).
- `scripts/plot_fp_fn.py` → `models/fp_fn_comparison.csv`, `models/fn_comparison.png`, `models/fp_comparison.png`.
- `scripts/compare_class_metrics.py` → `models/class_metrics_comparison.csv`, `models/class_precision_bar_comparison.png`, `models/class_recall_bar_comparison.png`, `models/class_f1_score_bar_comparison.png`.

Run these from project root, e.g.:

```powershell
python scripts/aggregate_evaluation_metrics.py
python scripts/plot_three_models.py
python scripts/plot_fp_fn.py
python scripts/compare_class_metrics.py
```

---

## 7) Next steps (optional suggestions)

1. Quantization-Aware Training (QAT): scaffold QAT for the chosen model and fine-tune for 10–30 epochs; this typically recovers quantized accuracy.
2. Focused augmentation / rebalancing: add class-targeted augmentation for `early_blight_leaf` and `septoria_leaf` (they are either confused for each other or cause higher FP rates).
3. Run per-class calibration or thresholding for a high-precision production mode (raise decision threshold for classes with high FP cost).
4. Export high-resolution figures and a single multipanel PDF for reporting.

If you want, I can: extract `model.summary()` for each saved model, create a short `README` inside `processed_dataset_96x96/` documenting the preprocessing exactly, or scaffold QAT scripts for the model you prefer.
