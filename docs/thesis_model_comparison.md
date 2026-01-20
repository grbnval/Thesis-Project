# Thesis-ready comparison: SqueezeNet-inspired vs MobileNetV2 (baseline) vs MobileNetV2 (Eff-style)

This document is a compact, thesis-ready comparison of the three fine-tuned models evaluated in this project. It summarizes dataset and preprocessing, training protocol, numeric results (Keras, TFLite, quantized TFLite), per-class failure modes, efficiency trade-offs (size / latency / FPS), and recommendations for deployment and follow-up work.

## Executive summary

- Best test accuracy (Keras checkpoint): SqueezeNet-inspired (0.9793).
- Best inference latency (standard TFLite): MobileNetV2 EfficientNet-style (~10 ms per image measured here).
- Best retention of accuracy after post-training quantization: MobileNetV2 variants (~0.903–0.904) outperform SqueezeNet (0.897).
- Practical recommendation: For top accuracy use SqueezeNet; for on-device latency and better quantized behavior use MobileNetV2 EfficientNet-style or baseline.

## 1. Dataset and preprocessing (brief)

- Raw images are organized by class inside `raw_dataset/` (directories: `healthy_leaf`, `early_blight_leaf`, `late_blight_leaf`, `septoria_leaf`, `unknown`).
- All images are preprocessed by `preprocess_dataset_96x96.py` and saved as per-class `.npy` images in `processed_dataset_96x96/`.
- Preprocessing steps (exact): BGR→RGB, resize to 96×96, CLAHE on L channel (clipLimit=3.0, tileGridSize=(8,8)), normalize to [0,1], save as `float32` `.npy` arrays.
- Train/test split: `create_train_test_split(..., test_split=0.2)` produces an approximately 80/20 per-class stratified split saved to `processed_train_96x96/` and `processed_test_96x96/`. The split uses `random.shuffle()` (no fixed seed by default).

## 2. Models and training protocol (shared)

- Input shape: (96, 96, 3). Final layer: `Dense(num_classes, activation='softmax')`.
- Transfer learning: base models initialized with ImageNet weights where applicable.
- Two-phase fine-tuning schedule used across runs:
  1. Phase 1: train top/head layers for 20 epochs at LR=1e-3.
  2. Phase 2: unfreeze the last ~30 base layers and continue training (total ≈100 epochs) at LR=1e-4.
- Optimizer: Adam; Batch size: 32; Loss: categorical_crossentropy (one-hot labels). Data augmentation during training uses ImageDataGenerator (rotations, shifts, shear, zoom, horizontal flip) as specified in training scripts.

## 3. Numeric results (key metrics)

### Model comparison bar chart

Below is a grouped bar chart comparing the three models on Keras accuracy, standard TFLite accuracy, and quantized TFLite accuracy. This figure makes it easy to see which model performs best in each deployment format and which retains the most accuracy after quantization.

![Grouped bar chart comparing three models on Keras, standard TFLite, and quantized TFLite accuracy. SqueezeNet-inspired achieves the highest Keras accuracy, but MobileNetV2 variants retain more accuracy after quantization, making them preferable for low-latency or quantized deployment.](../models/model_comparison_bar.png)

**Caption:**
Grouped bar chart comparing SqueezeNet-inspired, MobileNetV2-baseline, and MobileNetV2-Eff-style models on Keras, standard TFLite, and quantized TFLite accuracy. SqueezeNet-inspired achieves the highest Keras accuracy, but MobileNetV2 variants retain more accuracy after quantization, making them preferable for low-latency or quantized deployment.

All metrics below are taken from each model's evaluation artifacts (`models/<model_name>/evaluation/`). Reported values are the evaluation-script outputs used for the project figures and the `models/evaluation_metrics.csv` aggregation.

- SqueezeNet (folder `squeezenet_96x96_full_epochs_with_unknown`)
  - Keras (best checkpoint) accuracy: 0.9793
  - Standard TFLite accuracy: 0.9790 (avg inference ≈ 26.25 ms, size ≈ 2.84 MB)
  - Quantized TFLite accuracy: 0.8970 (avg inference ≈ 7.47 ms, size ≈ 3.04 MB)

- MobileNetV2 baseline (`mobilenetv2_96x96_full_epochs_with_unknown`)
  - Keras accuracy: 0.9583
  - Standard TFLite: 0.9570 (avg inference ≈ 11.25 ms, size ≈ 3.15 MB)
  - Quantized TFLite: 0.9035 (avg inference ≈ 1.64 ms, size ≈ 3.36 MB)

- MobileNetV2 EfficientNet-style (`mobilenetv2_efficientnet_style_96x96`)
  - Keras accuracy: 0.9692
  - Standard TFLite: 0.9651 (avg inference ≈ 10.07 ms, size ≈ 3.15 MB)
  - Quantized TFLite: 0.9041 (avg inference ≈ 2.01 ms, size ≈ 3.36 MB)

### Observations
- SqueezeNet achieves the highest raw Keras accuracy but shows the largest relative drop when post-training quantized.
- MobileNetV2 variants offer substantially faster standard TFLite latency and better quantized retention.

## 4. Per-class failure modes (highlights)

- `early_blight_leaf` is the most problematic class across runs (higher false negative rates). Examples from Keras evaluation:
  - SqueezeNet: 38 / 596 FN (≈6.4%)
  - MobileNetV2 baseline: 83 / 596 FN (≈13.9%)
  - MobileNetV2 Eff-style: 74 / 596 FN (≈12.4%)
- Quantized models amplify FN for some classes; e.g., SqueezeNet quantized `early_blight_leaf` FN increased to 149/596 (≈25%).
- `septoria_leaf` exhibits comparatively higher false positive rates in some runs (model-dependent).

Recommendation from error analysis: use targeted data augmentation or class rebalancing for `early_blight_leaf` and `septoria_leaf` (augment real examples, use focal loss or class-weighting, or thresholding during inference when FP cost is high).

## 5. Efficiency / deployment trade-offs

- Measured standard TFLite avg inference times (development host): SqueezeNet ≈ 26 ms, MobileNetV2 baseline ≈ 11 ms, MobileNetV2 Eff-style ≈ 10 ms.
- Quantization reduces latency significantly for all models (quantized TFLite avg inference times reported in evaluations), but can reduce accuracy. MobileNetV2 variants retained more accuracy after post-training quantization.

Practical selection guidance:
- If on-device latency and quantized robustness are primary constraints: MobileNetV2 EfficientNet-style is the best trade-off.
- If maximum Keras accuracy on the test set is the priority and you can accept slower inference or additional work to improve quantized accuracy (e.g., QAT): SqueezeNet-inspired is a good choice.

## 6. Reproducibility (how to reproduce key artifacts)

From project root (PowerShell examples):

```powershell
python scripts/aggregate_evaluation_metrics.py
python scripts/plot_three_models.py
python scripts/plot_fp_fn.py
python scripts/compare_class_metrics.py
```

To recreate preprocessed datasets and the 80/20 split interactively:

```powershell
python preprocess_dataset_96x96.py
# follow prompts (default: process images and create test_split=0.2)
```

To make the split deterministic, edit `preprocess_dataset_96x96.py` and set a seed (for example `random.seed(42)`) before calls to `random.shuffle()`.

## 7. Recommended next steps for thesis / deployment

1. Run Quantization-Aware Training (QAT) for the chosen model and re-evaluate quantized accuracy and per-class behavior.
2. Extract `model.summary()` and parameter counts for each saved model and include them in the thesis appendix (I can automate this and append the outputs to this file).
3. Add calibrated decision thresholds or class-specific thresholds if certain FP costs are more severe than FN costs for target deployment.
4. Re-run latency benchmarks on representative target hardware (ESP32 / mobile CPU) for final deployment decisions.

## 8. Appendix — exact artifact locations

- Preprocessed data (per-class `.npy`): `processed_dataset_96x96/`
- Train/test split: `processed_train_96x96/`, `processed_test_96x96/`
- Model folders: `models/squeezenet_96x96_full_epochs_with_unknown/`, `models/mobilenetv2_96x96_full_epochs_with_unknown/`, `models/mobilenetv2_efficientnet_style_96x96/`
- Model evaluation outputs: `models/<model_name>/evaluation/` (contains `classification_report.txt`, confusion matrices, FP/FN reports, TFLite evaluation files, and plots)
- Aggregated CSVs and comparison plots: `models/evaluation_metrics.csv`, `models/comparison_three_models.csv`, `models/comparison_plot.png`, `models/class_metrics_comparison.csv` etc.

---

If you want, I will append `model.summary()` outputs and exact parameter counts for each saved model to this document (automated), and/or add a one-page printable figure showing Keras vs quantized accuracy + latency for the three models. Which follow-up would you like first? (QAT scaffold, summaries, or printable figure.)

## How to read the parameter plots and model summaries (plain language)

Below are short, plain-language explanations for the plots and model summaries generated in `models/` so you can include them in your thesis with minimal extra interpretation.

- What "total parameters" means:
  - "Total parameters" is the total number of numeric weights the model stores. More parameters generally means a larger file and more compute needed to run the model, but not always proportionally slower.

- Trainable vs non-trainable parameters:
  - "Trainable" parameters are those that are updated during training (weights and biases in layers that are not frozen). "Non-trainable" parameters are fixed during fine-tuning (for example, the frozen part of a pretrained backbone) or are bookkeeping tensors (e.g., BatchNorm moving averages).

- The per-layer bar charts (what they show):
  - Each per-model plot (`*_layer_param_counts.png`) shows how many parameters each layer holds. The plots use a log scale on the x-axis so both very large layers and small layers are visible on the same chart.
  - We show the top ~30 layers by parameter count to keep the chart readable. Dense (fully connected) and large convolutional layers usually appear at the top.

- How to use these plots when choosing a model:
  - If you need the smallest model file and lowest memory use, pick the model with the fewest total parameters.
  - If you need the fastest inference on-device, look at the evaluated latency numbers (in `models/<model>/evaluation/`) — fewer parameters often helps latency but hardware and layer type matter (some small models are slow if they have unfriendly memory access patterns).
  - If quantized performance is important, use the quantized accuracy numbers as well; a model with slightly more parameters but better quantized retention may be the better deployment choice.

- Simple interpretation example (from this project):
  - SqueezeNet-inspired has fewer total parameters than the MobileNetV2 runs in some counts but showed the highest Keras accuracy; however it lost more accuracy after post-training quantization. That trade-off is visible in the totals plot (`model_param_totals.png`) and the quantized evaluation reports.

Use the generated CSVs (`models/*_layer_params.csv` and `models/model_param_totals.csv`) to build tables in your thesis; the images are already suitable for insertion into slides or a printed appendix.
